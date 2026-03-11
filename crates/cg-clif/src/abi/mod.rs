//! ABI-driven call/return lowering helpers.

pub(crate) mod pass_mode;
pub(crate) mod returning;

use cranelift_codegen::ir::Signature;
use cranelift_codegen::isa::TargetIsa;
use hir_def::DefWithBodyId;
use hir_def::FunctionId;
use hir_def::attrs::AttrFlags;
use hir_ty::codegen_abi::{self, CodegenCx};
use hir_ty::db::HirDatabase;
use hir_ty::mir::MirBody;
use hir_ty::next_solver::{DbInterner, GenericArgs, StoredTy};
use hir_ty::traits::StoredParamEnvAndCrate;
use intern::sym;
use rac_abi::callconv::{ArgAttributes, PassMode};
use rac_abi::spec::{Abi, Arch, Env as TargetEnv, Os, RustcAbi, Target};
use rustc_abi::{BackendRepr, TargetDataLayout};
use rustc_type_ir::inherent::Tys;
use target_lexicon::{Architecture, Environment, OperatingSystem};
use typed_arena::Arena;

use crate::LayoutArc;

#[derive(Clone)]
pub(crate) struct ArgAbi {
    pub(crate) mode: PassMode,
    pub(crate) layout: Option<LayoutArc>,
}

#[derive(Clone)]
pub(crate) struct FnAbi {
    pub(crate) args: Vec<ArgAbi>,
    pub(crate) ret: ArgAbi,
    pub(crate) c_variadic: bool,
    pub(crate) requires_caller_location: bool,
    pub(crate) sig: Signature,
}

pub(crate) fn function_requires_caller_location(db: &dyn HirDatabase, func_id: FunctionId) -> bool {
    let def_db: &dyn hir_def::db::DefDatabase = db;
    AttrFlags::query(def_db, func_id.into()).contains(AttrFlags::TRACK_CALLER)
}

pub(crate) fn body_requires_caller_location(db: &dyn HirDatabase, owner: DefWithBodyId) -> bool {
    match owner {
        DefWithBodyId::FunctionId(func_id) => function_requires_caller_location(db, func_id),
        DefWithBodyId::StaticId(_) | DefWithBodyId::ConstId(_) | DefWithBodyId::VariantId(_) => {
            false
        }
    }
}

fn arg_abi_for_layout(layout: LayoutArc) -> ArgAbi {
    let mode = match layout.backend_repr {
        _ if layout.is_zst() => PassMode::Ignore,
        BackendRepr::Scalar(_) => PassMode::Direct(ArgAttributes::new()),
        BackendRepr::ScalarPair(_, _) => PassMode::Pair(ArgAttributes::new(), ArgAttributes::new()),
        _ => PassMode::Indirect {
            attrs: ArgAttributes::new(),
            meta_attrs: (!layout.is_sized()).then_some(ArgAttributes::new()),
            on_stack: false,
        },
    };
    ArgAbi { mode, layout: Some(layout) }
}

fn ignore_arg_abi() -> ArgAbi {
    ArgAbi { mode: PassMode::Ignore, layout: None }
}

fn local_arg_abi_from_rac_abi<'a>(
    arg_abi: &rac_abi::callconv::ArgAbi<'a, codegen_abi::AbiTy<'_>>,
) -> ArgAbi {
    ArgAbi {
        mode: arg_abi.mode.clone(),
        layout: Some(triomphe::Arc::new(arg_abi.layout.layout.0.0.clone())),
    }
}

fn signature_from_fn_abi(isa: &dyn TargetIsa, dl: &TargetDataLayout, fn_abi: &FnAbi) -> Signature {
    let mut sig = Signature::new(isa.default_call_conv());
    let (ret_ptr, returns) = pass_mode::abi_return_for_arg(dl, &fn_abi.ret);
    sig.params.extend(ret_ptr);
    for arg in &fn_abi.args {
        sig.params.extend(pass_mode::abi_params_for_arg(dl, arg));
    }
    if fn_abi.requires_caller_location {
        sig.params.push(cranelift_codegen::ir::AbiParam::new(crate::pointer_ty(dl)));
    }
    sig.returns.extend(returns);
    sig
}

fn target_spec_from_isa(isa: &dyn TargetIsa, dl: &TargetDataLayout) -> Target {
    let triple = isa.triple();
    let arch = match triple.architecture {
        Architecture::X86_64 => Arch::X86_64,
        Architecture::X86_32(_) => Arch::X86,
        Architecture::Aarch64(_) => Arch::AArch64,
        Architecture::Arm(_) => Arch::Arm,
        Architecture::Riscv64(_) => Arch::RiscV64,
        Architecture::Riscv32(_) => Arch::RiscV32,
        _ => Arch::Other(triple.architecture.to_string().into()),
    };
    let os = match triple.operating_system {
        OperatingSystem::Linux => Os::Linux,
        OperatingSystem::Darwin(_) | OperatingSystem::MacOSX { .. } => Os::Other("darwin".into()),
        OperatingSystem::Windows => Os::Other("windows".into()),
        OperatingSystem::Freebsd => Os::FreeBsd,
        OperatingSystem::Aix => Os::Aix,
        _ => Os::Other(triple.operating_system.to_string().into()),
    };
    let env = match triple.environment {
        Environment::Gnu | Environment::Gnueabi | Environment::Gnueabihf | Environment::Gnux32 => {
            TargetEnv::Gnu
        }
        Environment::Musl | Environment::Musleabi | Environment::Musleabihf => TargetEnv::Musl,
        Environment::Uclibc => TargetEnv::Uclibc,
        _ => TargetEnv::Other(triple.environment.to_string().into()),
    };
    let abi = match triple.binary_format {
        _ => Abi::Other(triple.binary_format.to_string().into()),
    };
    let is_x86_sse2 = matches!(arch, Arch::X86_64)
        || matches!(arch, Arch::X86) && triple.environment.to_string().contains("sse2");

    Target {
        arch,
        os: os.clone(),
        env: env.clone(),
        abi,
        rustc_abi: is_x86_sse2.then_some(RustcAbi::X86Sse2),
        llvm_target: triple.to_string(),
        llvm_abiname: String::new(),
        pointer_width: dl.pointer_size().bits().try_into().expect("pointer width exceeds u32"),
        abi_return_struct_as_int: false,
        is_like_darwin: matches!(os, Os::Other(ref os) if os == "darwin"),
        is_like_windows: matches!(os, Os::Other(ref os) if os == "windows"),
        is_like_msvc: matches!(env, TargetEnv::Other(ref env) if env == "msvc"),
        simd_types_indirect: true,
    }
}

fn rust_fn_abi_from_tys(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    arg_tys: &[StoredTy],
    ret_ty: StoredTy,
    c_variadic: bool,
) -> Result<FnAbi, String> {
    let arena = Arena::new();
    let cx = CodegenCx {
        db,
        env: env.clone(),
        target_data: dl,
        target_spec: target_spec_from_isa(isa, dl),
        arena: &arena,
    };
    let arg_tys: Vec<_> = arg_tys.iter().map(|ty| ty.as_ref()).collect();
    let rac_fn_abi = hir_ty::codegen_abi::compute_fn_abi(&cx, &arg_tys, ret_ty.as_ref())
        .ok_or_else(|| "compute_fn_abi failed".to_owned())?;
    let args = rac_fn_abi.args.iter().map(local_arg_abi_from_rac_abi).collect::<Vec<_>>();
    let ret = local_arg_abi_from_rac_abi(&rac_fn_abi.ret);
    let mut fn_abi = FnAbi {
        args,
        ret,
        c_variadic,
        requires_caller_location: false,
        sig: Signature::new(isa.default_call_conv()),
    };
    fn_abi.sig = signature_from_fn_abi(isa, dl, &fn_abi);
    Ok(fn_abi)
}

fn fn_abi_from_arg_and_ret_tys(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    arg_tys: &[StoredTy],
    ret_ty: Option<StoredTy>,
    c_variadic: bool,
) -> Result<FnAbi, String> {
    let args = arg_tys
        .iter()
        .map(|ty| {
            db.layout_of_ty(ty.clone(), env.clone())
                .map(arg_abi_for_layout)
                .map_err(|e| format!("param layout error: {:?}", e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let ret = if let Some(ret_ty) = ret_ty {
        let ret_layout = db
            .layout_of_ty(ret_ty, env.clone())
            .map_err(|e| format!("return type layout error: {:?}", e))?;
        arg_abi_for_layout(ret_layout)
    } else {
        ignore_arg_abi()
    };

    let mut fn_abi = FnAbi {
        args,
        ret,
        c_variadic,
        requires_caller_location: false,
        sig: Signature::new(isa.default_call_conv()),
    };
    fn_abi.sig = signature_from_fn_abi(isa, dl, &fn_abi);
    Ok(fn_abi)
}

pub(crate) fn fn_abi_for_body(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
) -> Result<FnAbi, String> {
    let ret_ty = body.locals[hir_ty::mir::return_slot()].ty.clone();
    let arg_tys = body
        .param_locals
        .iter()
        .map(|&param_local| body.locals[param_local].ty.clone())
        .collect::<Vec<_>>();
    let mut fn_abi = rust_fn_abi_from_tys(isa, db, dl, env, &arg_tys, ret_ty, false)?;
    fn_abi.requires_caller_location = body_requires_caller_location(db, body.owner);
    fn_abi.sig = signature_from_fn_abi(isa, dl, &fn_abi);
    Ok(fn_abi)
}

pub(crate) fn fn_abi_for_fn_item_from_ty(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    func_id: FunctionId,
    generic_args: GenericArgs<'_>,
) -> Result<FnAbi, String> {
    let interner = DbInterner::new_no_crate(db);
    let fn_sig = if generic_args.is_empty() {
        db.callable_item_signature(func_id.into()).skip_binder().skip_binder()
    } else {
        db.callable_item_signature(func_id.into()).instantiate(interner, generic_args).skip_binder()
    };

    let ret_ty = {
        let output = *fn_sig.inputs_and_output.as_slice().split_last().unwrap().0;
        (!output.is_never()).then(|| output.store())
    };
    let arg_tys = fn_sig.inputs_and_output.inputs().iter().map(|ty| ty.store()).collect::<Vec<_>>();
    let fn_sig_data = db.function_signature(func_id);
    let c_variadic = fn_sig_data.is_varargs();
    let is_rust_abi = fn_sig_data.abi.is_none() || fn_sig_data.abi == Some(sym::rust_dash_call);
    let mut fn_abi = if is_rust_abi {
        let ret_ty = ret_ty.unwrap_or_else(|| fn_sig.inputs_and_output.output().store());
        rust_fn_abi_from_tys(isa, db, dl, env, &arg_tys, ret_ty, c_variadic)
    } else {
        fn_abi_from_arg_and_ret_tys(isa, db, dl, env, &arg_tys, ret_ty, c_variadic)
    }?;
    fn_abi.requires_caller_location = function_requires_caller_location(db, func_id);
    fn_abi.sig = signature_from_fn_abi(isa, dl, &fn_abi);
    Ok(fn_abi)
}

pub(crate) fn fn_abi_for_fn_ptr(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    sig_tys: &rustc_type_ir::Binder<DbInterner, rustc_type_ir::FnSigTys<DbInterner>>,
    is_rust_call_abi: bool,
    c_variadic: bool,
) -> Result<FnAbi, String> {
    let sig_tys_inner = sig_tys.clone().skip_binder();
    let arg_tys = sig_tys_inner.inputs().iter().map(|ty| ty.store()).collect::<Vec<_>>();
    if is_rust_call_abi {
        rust_fn_abi_from_tys(isa, db, dl, env, &arg_tys, sig_tys_inner.output().store(), c_variadic)
    } else {
        fn_abi_from_arg_and_ret_tys(
            isa,
            db,
            dl,
            env,
            &arg_tys,
            Some(sig_tys_inner.output().store()),
            c_variadic,
        )
    }
}

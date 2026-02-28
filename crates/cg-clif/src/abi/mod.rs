//! ABI-driven call/return lowering helpers.

pub(crate) mod pass_mode;
pub(crate) mod returning;

use cranelift_codegen::ir::Signature;
use cranelift_codegen::isa::TargetIsa;
use hir_def::FunctionId;
use hir_ty::db::HirDatabase;
use hir_ty::mir::MirBody;
use hir_ty::next_solver::{DbInterner, GenericArgs, IntoKind, StoredTy, TyKind};
use hir_ty::traits::StoredParamEnvAndCrate;
use rac_abi::callconv::{ArgAttributes, PassMode};
use rustc_abi::{BackendRepr, TargetDataLayout};

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
    pub(crate) sig: Signature,
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

fn signature_from_fn_abi(isa: &dyn TargetIsa, dl: &TargetDataLayout, fn_abi: &FnAbi) -> Signature {
    let mut sig = Signature::new(isa.default_call_conv());
    let (ret_ptr, returns) = pass_mode::abi_return_for_arg(dl, &fn_abi.ret);
    sig.params.extend(ret_ptr);
    for arg in &fn_abi.args {
        sig.params.extend(pass_mode::abi_params_for_arg(dl, arg));
    }
    sig.returns.extend(returns);
    sig
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

    let mut fn_abi = FnAbi { args, ret, c_variadic, sig: Signature::new(isa.default_call_conv()) };
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
    fn_abi_from_arg_and_ret_tys(isa, db, dl, env, &arg_tys, Some(ret_ty), false)
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
    let c_variadic = db.function_signature(func_id).is_varargs();
    fn_abi_from_arg_and_ret_tys(isa, db, dl, env, &arg_tys, ret_ty, c_variadic)
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

    let ret_ty = {
        let output = sig_tys_inner.output();
        (!output.is_never()).then(|| output.store())
    };

    let mut arg_tys = sig_tys_inner.inputs().iter().map(|ty| ty.store()).collect::<Vec<_>>();
    if is_rust_call_abi {
        if let Some((packed_tuple_ty, prefix_tys)) = arg_tys.split_last() {
            if let TyKind::Tuple(tuple_fields) = packed_tuple_ty.as_ref().kind() {
                let mut flattened = prefix_tys.to_vec();
                flattened.extend(tuple_fields.iter().map(|field_ty| field_ty.store()));
                arg_tys = flattened;
            }
        }
    }

    fn_abi_from_arg_and_ret_tys(isa, db, dl, env, &arg_tys, ret_ty, c_variadic)
}

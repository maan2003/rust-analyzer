//! MIR → Cranelift IR codegen for rust-analyzer.
//!
//! Translates r-a's MIR representation to Cranelift IR and emits object files
//! via cranelift-object. Based on patterns from cg_clif (rustc's Cranelift backend).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, ArgumentPurpose, Block, InstBuilder, MemFlags, Signature, Type, Value, types};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{DataDescription, FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use hir_def::{AssocItemId, CallableDefId, HasModule, ItemContainerId, Lookup, TraitId, VariantId};
use hir_def::signatures::FunctionSignature;
use hir_ty::db::HirDatabase;
use hir_ty::mir::{
    BasicBlockId, BinOp, CastKind, LocalId, MirBody, Operand, OperandKind, Place,
    ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp,
};
use either::Either;
use hir_ty::PointerCast;
use hir_ty::method_resolution::TraitImpls;
use hir_ty::next_solver::{Const, ConstKind, DbInterner, GenericArgs, IntoKind, StoredGenericArgs, StoredTy, TyKind};
use rustc_type_ir::inherent::GenericArgs as _;
use hir_ty::traits::StoredParamEnvAndCrate;
use rac_abi::VariantIdx;
use rustc_abi::{BackendRepr, Primitive, Scalar, Size, TargetDataLayout};
use triomphe::Arc as TArc;

pub mod layout;
pub mod link;
pub mod mirdata_codegen;
mod pointer;
pub mod symbol_mangling;
mod value_and_place;

use hir_ty::layout::Layout;
use value_and_place::{CPlace, CValue};

/// Layout Arc type alias (triomphe::Arc from hir-ty's layout_of_ty).
type LayoutArc = TArc<Layout>;

// ---------------------------------------------------------------------------
// Type mapping: r-a Ty → Cranelift Type
// ---------------------------------------------------------------------------

fn scalar_to_clif_type(dl: &TargetDataLayout, scalar: &Scalar) -> Type {
    use rustc_abi::Primitive;
    match scalar.primitive() {
        Primitive::Int(int, _signed) => match int.size().bits() {
            8 => types::I8,
            16 => types::I16,
            32 => types::I32,
            64 => types::I64,
            128 => types::I128,
            _ => unreachable!("unsupported int size: {}", int.size().bits()),
        },
        Primitive::Float(float) => match float {
            rustc_abi::Float::F16 => types::F16,
            rustc_abi::Float::F32 => types::F32,
            rustc_abi::Float::F64 => types::F64,
            rustc_abi::Float::F128 => types::F128,
        },
        Primitive::Pointer(_) => pointer_ty(dl),
    }
}

fn pointer_ty(dl: &TargetDataLayout) -> Type {
    match dl.pointer_size().bits() {
        16 => types::I16,
        32 => types::I32,
        64 => types::I64,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// MirSource: backend-specific fields for FunctionCx
// ---------------------------------------------------------------------------

pub(crate) enum MirSource<'a> {
    Ra {
        db: &'a dyn HirDatabase,
        env: StoredParamEnvAndCrate,
        body: &'a MirBody,
        local_crate: base_db::Crate,
        ext_crate_disambiguators: &'a HashMap<String, u64>,
    },
    Mirdata {
        #[allow(dead_code)]
        body: &'a ra_mir_types::Body,
        #[allow(dead_code)]
        layouts: &'a [LayoutArc],
    },
}

// ---------------------------------------------------------------------------
// FunctionCx: per-function codegen state
// ---------------------------------------------------------------------------

pub(crate) struct FunctionCx<'a, M: Module> {
    pub(crate) bcx: FunctionBuilder<'a>,
    pub(crate) module: &'a mut M,
    pub(crate) isa: &'a dyn TargetIsa,
    pub(crate) pointer_type: Type,
    pub(crate) dl: &'a TargetDataLayout,
    pub(crate) mir: MirSource<'a>,
    /// MIR basic block → Cranelift block (indexed by raw block id)
    pub(crate) block_map: Vec<Block>,
    /// MIR local → CPlace (indexed by raw local id)
    pub(crate) local_map: Vec<CPlace>,
}

impl<'a, M: Module> FunctionCx<'a, M> {
    fn db(&self) -> &'a dyn HirDatabase {
        match &self.mir {
            MirSource::Ra { db, .. } => *db,
            _ => panic!("db() called on non-Ra FunctionCx"),
        }
    }

    fn env(&self) -> &StoredParamEnvAndCrate {
        match &self.mir {
            MirSource::Ra { env, .. } => env,
            _ => panic!("env() called on non-Ra FunctionCx"),
        }
    }

    fn ra_body(&self) -> &'a MirBody {
        match &self.mir {
            MirSource::Ra { body, .. } => body,
            _ => panic!("ra_body() called on non-Ra FunctionCx"),
        }
    }

    fn local_crate(&self) -> base_db::Crate {
        match &self.mir {
            MirSource::Ra { local_crate, .. } => *local_crate,
            _ => panic!("local_crate() called on non-Ra FunctionCx"),
        }
    }

    fn ext_crate_disambiguators(&self) -> &'a HashMap<String, u64> {
        match &self.mir {
            MirSource::Ra { ext_crate_disambiguators, .. } => ext_crate_disambiguators,
            _ => panic!("ext_crate_disambiguators() called on non-Ra FunctionCx"),
        }
    }

    fn clif_block(&self, bb: BasicBlockId) -> Block {
        self.block_map[bb.into_raw().into_u32() as usize]
    }

    fn clif_block_idx(&self, idx: usize) -> Block {
        self.block_map[idx]
    }

    fn local_place(&self, local: LocalId) -> &CPlace {
        &self.local_map[local.into_raw().into_u32() as usize]
    }

    fn local_place_idx(&self, idx: usize) -> &CPlace {
        &self.local_map[idx]
    }
}

// ---------------------------------------------------------------------------
// Constant extraction
// ---------------------------------------------------------------------------

fn const_to_i64(konst: Const<'_>, size: Size) -> i64 {
    match konst.kind() {
        ConstKind::Value(val) => {
            let bytes = &val.value.inner().memory;
            let n = size.bytes() as usize;
            let mut buf = [0u8; 8];
            let len = n.min(8);
            buf[..len].copy_from_slice(&bytes[..len]);
            i64::from_le_bytes(buf)
        }
        _ => panic!("const_to_i64: unsupported const kind: {:?}", konst),
    }
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Extract the stored type from an operand.
fn operand_ty(db: &dyn HirDatabase, body: &MirBody, kind: &OperandKind) -> StoredTy {
    match kind {
        OperandKind::Constant { ty, .. } => ty.clone(),
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            place_ty(db, body, place)
        }
        OperandKind::Static(_) => todo!("static operand type"),
    }
}

fn place_ty(db: &dyn HirDatabase, body: &MirBody, place: &Place) -> StoredTy {
    let mut ty = body.locals[place.local].ty.clone();
    let projections = place.projection.lookup(&body.projection_store);
    for proj in projections {
        ty = match proj {
            ProjectionElem::Field(field) => field_type(db, &ty, field),
            ProjectionElem::Deref => {
                ty.as_ref().builtin_deref(true).expect("deref on non-pointer").store()
            }
            ProjectionElem::Downcast(_) => ty, // Downcast doesn't change the Rust type
            ProjectionElem::ClosureField(idx) => closure_field_type(db, &ty, *idx),
            ProjectionElem::Index(_) => match ty.as_ref().kind() {
                TyKind::Array(elem, _) | TyKind::Slice(elem) => elem.store(),
                _ => panic!("Index on non-array/slice type"),
            },
            _ => todo!("place_ty for {:?}", proj),
        };
    }
    ty
}

fn bin_op_to_intcc(op: &BinOp, signed: bool) -> IntCC {
    match op {
        BinOp::Eq => IntCC::Equal,
        BinOp::Ne => IntCC::NotEqual,
        BinOp::Lt if signed => IntCC::SignedLessThan,
        BinOp::Lt => IntCC::UnsignedLessThan,
        BinOp::Le if signed => IntCC::SignedLessThanOrEqual,
        BinOp::Le => IntCC::UnsignedLessThanOrEqual,
        BinOp::Gt if signed => IntCC::SignedGreaterThan,
        BinOp::Gt => IntCC::UnsignedGreaterThan,
        BinOp::Ge if signed => IntCC::SignedGreaterThanOrEqual,
        BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
        _ => unreachable!("not a comparison: {:?}", op),
    }
}

fn bin_op_to_floatcc(op: &BinOp) -> FloatCC {
    match op {
        BinOp::Eq => FloatCC::Equal,
        BinOp::Ne => FloatCC::NotEqual,
        BinOp::Lt => FloatCC::LessThan,
        BinOp::Le => FloatCC::LessThanOrEqual,
        BinOp::Gt => FloatCC::GreaterThan,
        BinOp::Ge => FloatCC::GreaterThanOrEqual,
        _ => unreachable!("not a float comparison: {:?}", op),
    }
}

fn ty_is_signed_int(ty: StoredTy) -> bool {
    matches!(ty.as_ref().kind(), TyKind::Int(_))
}

fn codegen_intcast(fx: &mut FunctionCx<'_, impl Module>, val: Value, to_ty: Type, signed: bool) -> Value {
    let from_ty = fx.bcx.func.dfg.value_type(val);
    match (from_ty, to_ty) {
        (_, _) if from_ty == to_ty => val,
        (_, _) if to_ty.wider_or_equal(from_ty) => {
            if signed {
                fx.bcx.ins().sextend(to_ty, val)
            } else {
                fx.bcx.ins().uextend(to_ty, val)
            }
        }
        (_, _) => fx.bcx.ins().ireduce(to_ty, val),
    }
}

fn codegen_libcall1(
    fx: &mut FunctionCx<'_, impl Module>,
    name: &str,
    params: &[Type],
    ret: Type,
    args: &[Value],
) -> Value {
    let mut sig = Signature::new(fx.isa.default_call_conv());
    sig.params.extend(params.iter().copied().map(AbiParam::new));
    sig.returns.push(AbiParam::new(ret));

    let func_id = fx
        .module
        .declare_function(name, Linkage::Import, &sig)
        .expect("declare libcall");
    let func_ref = fx.module.declare_func_in_func(func_id, fx.bcx.func);
    let call = fx.bcx.ins().call(func_ref, args);
    fx.bcx.inst_results(call)[0]
}

fn codegen_cast(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: &CastKind,
    operand: &Operand,
    target_ty: &StoredTy,
    result_layout: &LayoutArc,
) -> CValue {
    let body = fx.ra_body();
    // Handle Unsize coercion separately — it produces a fat pointer (ScalarPair)
    if let CastKind::PointerCoercion(PointerCast::Unsize) = kind {
        return codegen_unsize_coercion(fx, operand, target_ty, result_layout);
    }

    // Handle ReifyFnPointer: FnDef (ZST) \x{2192} fn pointer.
    // Must be before the generic scalar path because FnDef is ZST and has no scalar value.
    if let CastKind::PointerCoercion(PointerCast::ReifyFnPointer) = kind {
        let from_ty = operand_ty(fx.db(), body, &operand.kind);
        let TyKind::FnDef(def, generic_args) = from_ty.as_ref().kind() else {
            panic!("ReifyFnPointer on non-FnDef type: {:?}", from_ty);
        };
        let CallableDefId::FunctionId(callee_func_id) = def.0 else {
            panic!("ReifyFnPointer on non-function: {:?}", def);
        };
        // Declare the function in the module and get its address
        let is_extern = matches!(
            callee_func_id.loc(fx.db()).container,
            ItemContainerId::ExternBlockId(_)
        );
        let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();
        let (callee_sig, callee_name) = if is_extern {
            let sig = build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id)
                .expect("extern fn sig");
            let name = fx.db().function_signature(callee_func_id).name.as_str().to_owned();
            (sig, name)
        } else if is_cross_crate {
            let sig = build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id)
                .expect("cross-crate fn sig");
            let name = symbol_mangling::mangle_function(
                fx.db(), callee_func_id, generic_args, fx.ext_crate_disambiguators(),
            );
            (sig, name)
        } else {
            let callee_body = fx.db()
                .monomorphized_mir_body(callee_func_id.into(), generic_args.store(), fx.env().clone())
                .expect("failed to get callee MIR for ReifyFnPointer");
            let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body, &[])
                .expect("callee sig");
            let name = symbol_mangling::mangle_function(
                fx.db(), callee_func_id, generic_args, fx.ext_crate_disambiguators(),
            );
            (sig, name)
        };
        let callee_id = fx.module
            .declare_function(&callee_name, Linkage::Import, &callee_sig)
            .expect("declare callee for ReifyFnPointer");
        let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
        let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, callee_ref);
        return CValue::by_val(func_addr, result_layout.clone());
    }

    let from_ty = operand_ty(fx.db(), body, &operand.kind);
    let from_cval = codegen_operand(fx, &operand.kind);
    let from_val = from_cval.load_scalar(fx);
    let from_clif_ty = fx.bcx.func.dfg.value_type(from_val);

    let BackendRepr::Scalar(target_scalar) = result_layout.backend_repr else {
        todo!("cast target must be scalar")
    };
    let target_clif_ty = scalar_to_clif_type(fx.dl, &target_scalar);

    let val = match kind {
        CastKind::IntToInt => {
            let from_signed = ty_is_signed_int(from_ty);
            codegen_intcast(fx, from_val, target_clif_ty, from_signed)
        }
        CastKind::FloatToInt => {
            if ty_is_signed_int(target_ty.clone()) {
                fx.bcx.ins().fcvt_to_sint_sat(target_clif_ty, from_val)
            } else {
                fx.bcx.ins().fcvt_to_uint_sat(target_clif_ty, from_val)
            }
        }
        CastKind::IntToFloat => {
            let from_signed = ty_is_signed_int(from_ty);
            if from_signed {
                fx.bcx.ins().fcvt_from_sint(target_clif_ty, from_val)
            } else {
                fx.bcx.ins().fcvt_from_uint(target_clif_ty, from_val)
            }
        }
        CastKind::FloatToFloat => match (from_clif_ty, target_clif_ty) {
            (from, to) if from == to => from_val,
            (from, to) if to.wider_or_equal(from) => fx.bcx.ins().fpromote(to, from_val),
            (to, from) if to.wider_or_equal(from) => fx.bcx.ins().fdemote(from, from_val),
            _ => unreachable!("invalid float cast from {from_clif_ty:?} to {target_clif_ty:?}"),
        },
        CastKind::PtrToPtr
        | CastKind::FnPtrToPtr
        | CastKind::PointerExposeProvenance
        | CastKind::PointerWithExposedProvenance
        | CastKind::PointerCoercion(_) => {
            // Remaining PointerCoercion variants (MutToConstPointer, UnsafeFnPointer, etc.)
            // are thin ptr → thin ptr, handled as intcast.
            let from_signed = ty_is_signed_int(from_ty);
            codegen_intcast(fx, from_val, target_clif_ty, from_signed)
        }
        CastKind::Transmute => {
            assert_eq!(
                from_cval.layout.size, result_layout.size,
                "transmute between differently-sized types"
            );
            if from_clif_ty == target_clif_ty {
                from_val
            } else if from_clif_ty.bits() == target_clif_ty.bits() {
                fx.bcx.ins().bitcast(target_clif_ty, MemFlags::new(), from_val)
            } else {
                unreachable!(
                    "transmute between mismatched clif types: {from_clif_ty:?} -> {target_clif_ty:?}"
                );
            }
        }
        CastKind::DynStar => todo!("dyn* cast"),
    };
    CValue::by_val(val, result_layout.clone())
}

/// Handle `PointerCoercion(Unsize)`: `&T → &dyn Trait`.
/// Produces a fat pointer `(data_ptr, vtable_ptr)`.
/// Reference: cg_clif/src/unsize.rs `coerce_unsized_into`
fn codegen_unsize_coercion(
    fx: &mut FunctionCx<'_, impl Module>,
    operand: &Operand,
    target_ty: &StoredTy,
    result_layout: &LayoutArc,
) -> CValue {
    let body = fx.ra_body();
    let from_cval = codegen_operand(fx, &operand.kind);
    let data_ptr = from_cval.load_scalar(fx);

    // Extract the trait from the target type (&dyn Trait → Dyn → trait_id)
    let pointee_ty = target_ty.as_ref().builtin_deref(true)
        .expect("Unsize target must be a pointer/reference type");
    let trait_id = pointee_ty.dyn_trait()
        .expect("Unsize target pointee must be dyn Trait");

    // Extract the concrete source type (what's behind the thin pointer)
    let from_ty = operand_ty(fx.db(), body, &operand.kind);
    let source_pointee = from_ty.as_ref().builtin_deref(true)
        .expect("Unsize source must be a pointer/reference type");

    let vtable_ptr = get_or_create_vtable(fx, source_pointee.store(), trait_id);

    CValue::by_val_pair(data_ptr, vtable_ptr, result_layout.clone())
}

/// Build or retrieve a vtable for `concrete_ty` implementing `trait_id`.
///
/// Vtable layout (matches rustc):
/// - Slot 0: drop_in_place fn ptr (null for now)
/// - Slot 1: size of concrete type (usize)
/// - Slot 2: alignment of concrete type (usize)
/// - Slot 3+: trait methods in declaration order
///
/// Reference: cg_clif/src/vtable.rs `get_vtable` + cg_clif/src/constant.rs `data_id_for_vtable`
fn get_or_create_vtable(
    fx: &mut FunctionCx<'_, impl Module>,
    concrete_ty: StoredTy,
    trait_id: TraitId,
) -> Value {
    let ptr_size = fx.dl.pointer_size().bytes() as usize;

    // Get concrete type layout for size/align
    let concrete_layout = fx.db()
        .layout_of_ty(concrete_ty.clone(), fx.env().clone())
        .expect("layout error for vtable concrete type");
    let concrete_size = concrete_layout.size.bytes();
    let concrete_align = concrete_layout.align.abi.bytes();

    // Get trait methods in declaration order
    let trait_items = trait_id.trait_items(fx.db());
    let method_func_ids: Vec<hir_def::FunctionId> = trait_items.items.iter()
        .filter_map(|(_name, item)| match item {
            AssocItemId::FunctionId(fid) => Some(*fid),
            _ => None,
        })
        .collect();
    let num_methods = method_func_ids.len();

    // Find the impl for this concrete type
    let krate = fx.local_crate();
    let trait_impls = TraitImpls::for_crate(fx.db(), krate);
    let interner = DbInterner::new_no_crate(fx.db());

    // Simplify the concrete type for lookup (same approach as hir-ty's method_resolution)
    use rustc_type_ir::fast_reject::{TreatParams, simplify_type};
    let simplified = simplify_type(interner, concrete_ty.as_ref(), TreatParams::InstantiateWithInfer)
        .expect("cannot simplify concrete type for vtable lookup");
    let (impl_ids, _) = trait_impls.for_trait_and_self_ty(trait_id, &simplified);
    assert!(!impl_ids.is_empty(), "no impl found for vtable");
    let impl_id = impl_ids[0]; // Take first matching impl

    // Build unique vtable name
    let vtable_name = format!(
        "__vtable_{}_for_{:?}",
        trait_id.trait_items(fx.db()).items.first().map(|(n, _)| n.as_str().to_string()).unwrap_or_default(),
        simplified,
    );

    // Declare the vtable data object
    let data_id = fx.module
        .declare_data(&vtable_name, Linkage::Local, false, false)
        .expect("declare vtable data");

    // Build vtable data
    let total_size = ptr_size * (3 + num_methods);
    let mut data = DataDescription::new();
    let mut vtable_bytes = vec![0u8; total_size];

    // Slot 0: drop_in_place — null for now (no drop glue)
    // (already zeroed)

    // Slot 1: size
    vtable_bytes[ptr_size..ptr_size * 2].copy_from_slice(&(concrete_size as u64).to_le_bytes()[..ptr_size]);

    // Slot 2: alignment
    vtable_bytes[ptr_size * 2..ptr_size * 3].copy_from_slice(&(concrete_align as u64).to_le_bytes()[..ptr_size]);

    data.define(vtable_bytes.into_boxed_slice());

    // Slot 3+: trait method fn ptrs — emit as relocations
    let impl_items = impl_id.impl_items(fx.db());
    for (method_idx, trait_method_func_id) in method_func_ids.iter().enumerate() {
        // Find the corresponding impl method by name
        let trait_method_name = fx.db().function_signature(*trait_method_func_id).name.clone();
        let impl_func_id = impl_items.items.iter()
            .find_map(|(name, item)| {
                if *name == trait_method_name {
                    match item {
                        AssocItemId::FunctionId(fid) => Some(*fid),
                        _ => None,
                    }
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("impl method `{}` not found for vtable", trait_method_name.as_str()));

        // Declare/import the impl function
        let impl_body = fx.db()
            .monomorphized_mir_body(
                impl_func_id.into(),
                GenericArgs::empty(interner).store(),
                fx.env().clone(),
            )
            .expect("failed to get impl method MIR for vtable");
        let impl_sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &impl_body, &[])
            .expect("impl method sig for vtable");
        let impl_fn_name = symbol_mangling::mangle_function(
            fx.db(),
            impl_func_id,
            GenericArgs::empty(interner),
            fx.ext_crate_disambiguators(),
        );

        let func_id = fx.module
            .declare_function(&impl_fn_name, Linkage::Import, &impl_sig)
            .expect("declare vtable method");
        let func_ref = fx.module.declare_func_in_data(func_id, &mut data);
        data.write_function_addr(((3 + method_idx) * ptr_size) as u32, func_ref);
    }

    // Define the vtable data — ignore duplicate definition errors (vtable may be
    // created more than once if multiple unsizing coercions target the same pair)
    match fx.module.define_data(data_id, &data) {
        Ok(()) => {}
        Err(cranelift_module::ModuleError::DuplicateDefinition(_)) => {}
        Err(e) => panic!("define vtable data: {e}"),
    }

    // Get a pointer to the vtable in the current function
    let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
    fx.bcx.ins().symbol_value(fx.pointer_type, local_data_id)
}

/// MemFlags for vtable loads: always aligned, readonly, notrap.
/// Reference: cg_clif/src/vtable.rs `vtable_memflags`
fn vtable_memflags() -> MemFlags {
    let mut flags = MemFlags::trusted();
    flags.set_readonly();
    flags
}

// ---------------------------------------------------------------------------
// Place codegen
// ---------------------------------------------------------------------------

fn codegen_place(fx: &mut FunctionCx<'_, impl Module>, place: &Place) -> CPlace {
    let body = fx.ra_body();
    let projections = place.projection.lookup(&body.projection_store);
    if projections.is_empty() {
        return fx.local_place(place.local).clone();
    }

    let mut cplace = fx.local_place(place.local).clone();
    let mut cur_ty = body.locals[place.local].ty.clone();

    for proj in projections {
        match proj {
            ProjectionElem::Field(field) => {
                let field_idx = match field {
                    Either::Left(field_id) => {
                        field_id.local_id.into_raw().into_u32() as usize
                    }
                    Either::Right(tuple_field_id) => tuple_field_id.index as usize,
                };

                // Determine the field type from the current type
                let field_ty = field_type(fx.db(), &cur_ty, field);
                let field_layout = fx.db()
                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                    .expect("field layout error");

                cplace = cplace.place_field(fx, field_idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Deref => {
                let inner_ty = cur_ty
                    .as_ref()
                    .builtin_deref(true)
                    .expect("deref on non-pointer type");
                let inner_layout = fx.db()
                    .layout_of_ty(inner_ty.store(), fx.env().clone())
                    .expect("deref layout error");

                // Load the pointer value from the current place.
                // For fat pointers (e.g. &dyn Trait = ScalarPair), extract both
                // data ptr and metadata, and carry the metadata in the CPlace.
                let cval = cplace.to_cvalue(fx);
                cplace = match cval.layout.backend_repr {
                    BackendRepr::ScalarPair(_, _) => {
                        let (data_ptr, meta) = cval.load_scalar_pair(fx);
                        CPlace::for_ptr_with_extra(
                            pointer::Pointer::new(data_ptr),
                            meta,
                            inner_layout,
                        )
                    }
                    _ => {
                        let ptr_val = cval.load_scalar(fx);
                        CPlace::for_ptr(pointer::Pointer::new(ptr_val), inner_layout)
                    }
                };
                cur_ty = inner_ty.store();
            }
            ProjectionElem::Downcast(variant_idx) => {
                // For multi-variant enums in registers, spill to memory so
                // field projections use correct offsets (tag vs payload).
                if cplace.is_register() {
                    use rustc_abi::Variants;
                    if matches!(&cplace.layout.variants, Variants::Multiple { .. }) {
                        let cval = cplace.to_cvalue(fx);
                        let ptr = cval.force_stack(fx);
                        cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                    }
                }
                let variant_layout = variant_layout(
                    fx.db(),
                    &cur_ty,
                    &cplace.layout,
                    *variant_idx,
                    fx.env(),
                );
                cplace = cplace.downcast_variant(variant_layout);
                // cur_ty stays the same (Downcast is just a type assertion)
            }
            ProjectionElem::ClosureField(idx) => {
                // Closure captures are stored as fields of the closure struct
                let field_ty = closure_field_type(fx.db(), &cur_ty, *idx);
                let field_layout = fx.db()
                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                    .expect("closure field layout error");
                cplace = cplace.place_field(fx, *idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Index(index_local) => {
                let index_place = fx.local_place(*index_local).clone();
                let index_val = index_place.to_cvalue(fx).load_scalar(fx);
                let elem_ty = match cur_ty.as_ref().kind() {
                    TyKind::Array(elem, _) | TyKind::Slice(elem) => elem.store(),
                    _ => panic!("Index on non-array/slice type"),
                };
                let elem_layout = fx.db()
                    .layout_of_ty(elem_ty.clone(), fx.env().clone())
                    .expect("elem layout");
                let offset = fx.bcx.ins().imul_imm(index_val, elem_layout.size.bytes() as i64);
                // Arrays in registers → spill to memory
                if cplace.is_register() {
                    let cval = cplace.to_cvalue(fx);
                    let ptr = cval.force_stack(fx);
                    cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                }
                cplace = CPlace::for_ptr(
                    cplace.to_ptr().offset_value(&mut fx.bcx, fx.pointer_type, offset),
                    elem_layout,
                );
                cur_ty = elem_ty;
            }
            ProjectionElem::ConstantIndex { .. } => todo!("ConstantIndex projection"),
            ProjectionElem::Subslice { .. } => todo!("Subslice projection"),
            ProjectionElem::OpaqueCast(_) => todo!("OpaqueCast projection"),
        }
    }
    cplace
}

/// Get the type of a field from a parent type.
fn field_type(
    db: &dyn HirDatabase,
    parent_ty: &StoredTy,
    field: &Either<hir_def::FieldId, hir_def::TupleFieldId>,
) -> StoredTy {
    match parent_ty.as_ref().kind() {
        TyKind::Tuple(tys) => {
            let idx = match field {
                Either::Right(tuple_field) => tuple_field.index as usize,
                Either::Left(field_id) => {
                    field_id.local_id.into_raw().into_u32() as usize
                }
            };
            tys.as_slice()[idx].store()
        }
        TyKind::Adt(_adt_id, args) => {
            let Either::Left(field_id) = field else {
                panic!("TupleFieldId on ADT type");
            };
            let interner = DbInterner::new_no_crate(db);
            db.field_types(field_id.parent)[field_id.local_id]
                .get()
                .instantiate(interner, args)
                .store()
        }
        _ => todo!("field_type for {:?}", parent_ty.as_ref().kind()),
    }
}

/// Get the type of a closure field (captured variable).
fn closure_field_type(
    db: &dyn HirDatabase,
    closure_ty: &StoredTy,
    idx: usize,
) -> StoredTy {
    let TyKind::Closure(closure_id, args) = closure_ty.as_ref().kind() else {
        panic!("closure_field_type on non-closure: {:?}", closure_ty);
    };
    let interned = closure_id.0;
    let def = db.lookup_intern_closure(interned);
    let infer = hir_ty::InferenceResult::for_body(db, def.0);
    let (captures, _) = infer.closure_info(interned);
    captures[idx].ty(db, args).store()
}

/// Compute a variant layout for Downcast projection.
fn variant_layout(
    _db: &dyn HirDatabase,
    _ty: &StoredTy,
    parent_layout: &TArc<Layout>,
    variant_idx: VariantIdx,
    _env: &StoredParamEnvAndCrate,
) -> TArc<Layout> {
    use rustc_abi::Variants;
    match &parent_layout.variants {
        Variants::Single { .. } => {
            // Single variant — layout stays the same, just bump refcount
            parent_layout.clone()
        }
        Variants::Multiple { variants, .. } => {
            let variant = &variants[variant_idx];
            TArc::new(variant.clone())
        }
        Variants::Empty => panic!("downcast on empty variants"),
    }
}

/// Build a layout for a tag scalar (used for discriminant field access).
fn tag_scalar_layout(dl: &TargetDataLayout, tag: &Scalar) -> TArc<Layout> {
    let mut ly = Layout::scalar(dl, *tag);
    ly.size = tag.size(dl);
    TArc::new(ly)
}

/// Compare a Cranelift value against an i128 immediate.
/// Ported from upstream cg_clif/src/common.rs `codegen_icmp_imm`.
fn codegen_icmp_imm(
    fx: &mut FunctionCx<'_, impl Module>,
    intcc: IntCC,
    lhs: Value,
    rhs: i128,
) -> Value {
    let lhs_ty = fx.bcx.func.dfg.value_type(lhs);
    if lhs_ty == types::I128 {
        let (lhs_lsb, lhs_msb) = fx.bcx.ins().isplit(lhs);
        let (rhs_lsb, rhs_msb) = (rhs as u128 as u64 as i64, (rhs as u128 >> 64) as u64 as i64);
        match intcc {
            IntCC::Equal => {
                let lsb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_lsb, rhs_lsb);
                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                fx.bcx.ins().band(lsb_eq, msb_eq)
            }
            IntCC::NotEqual => {
                let lsb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_lsb, rhs_lsb);
                let msb_ne = fx.bcx.ins().icmp_imm(IntCC::NotEqual, lhs_msb, rhs_msb);
                fx.bcx.ins().bor(lsb_ne, msb_ne)
            }
            _ => {
                let msb_eq = fx.bcx.ins().icmp_imm(IntCC::Equal, lhs_msb, rhs_msb);
                let lsb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_lsb, rhs_lsb);
                let msb_cc = fx.bcx.ins().icmp_imm(intcc, lhs_msb, rhs_msb);
                fx.bcx.ins().select(msb_eq, lsb_cc, msb_cc)
            }
        }
    } else {
        fx.bcx.ins().icmp_imm(intcc, lhs, rhs as i64)
    }
}

/// Convert a `VariantId` to a `VariantIdx`.
fn variant_id_to_idx(db: &dyn HirDatabase, variant_id: VariantId) -> VariantIdx {
    match variant_id {
        VariantId::EnumVariantId(ev) => {
            let lookup = ev.lookup(db);
            VariantIdx::from_u32(lookup.index)
        }
        VariantId::StructId(_) | VariantId::UnionId(_) => VariantIdx::from_u32(0),
    }
}

// ---------------------------------------------------------------------------
// Statement codegen
// ---------------------------------------------------------------------------

fn codegen_statement(fx: &mut FunctionCx<'_, impl Module>, stmt: &StatementKind) {
    match stmt {
        StatementKind::Assign(place, rvalue) => {
            codegen_assign(fx, place, rvalue);
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
        StatementKind::Nop | StatementKind::FakeRead(_) => {}
        StatementKind::Deinit(_) => {}
        StatementKind::SetDiscriminant { place, variant_index } => {
            let dest = codegen_place(fx, place);
            codegen_set_discriminant(fx, &dest, *variant_index);
        }
    }
}

fn codegen_assign(fx: &mut FunctionCx<'_, impl Module>, place: &Place, rvalue: &Rvalue) {
    let dest = codegen_place(fx, place);
    if dest.layout.is_zst() {
        return;
    }

    // Some rvalues need to write directly to the destination place
    match rvalue {
        Rvalue::Aggregate(kind, operands) => {
            codegen_aggregate(fx, kind, operands, dest);
            return;
        }
        Rvalue::Ref(_, ref_place) | Rvalue::AddressOf(_, ref_place) => {
            let place = codegen_place(fx, ref_place);
            let ref_val = place.place_ref(fx, dest.layout.clone());
            dest.write_cvalue(fx, ref_val);
            return;
        }
        Rvalue::Discriminant(disc_place) => {
            let disc_cplace = codegen_place(fx, disc_place);
            let disc_val = codegen_get_discriminant(fx, &disc_cplace, &dest.layout);
            dest.write_cvalue(fx, CValue::by_val(disc_val, dest.layout.clone()));
            return;
        }
        _ => {}
    }

    let val = codegen_rvalue(fx, rvalue, &dest.layout);
    dest.write_cvalue(fx, val);
}

fn codegen_rvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    rvalue: &Rvalue,
    result_layout: &LayoutArc,
) -> CValue {
    match rvalue {
        Rvalue::Use(operand) => codegen_operand(fx, &operand.kind),
        Rvalue::BinaryOp(op, lhs, rhs) => codegen_binop(fx, op, lhs, rhs, result_layout),
        Rvalue::UnaryOp(op, operand) => codegen_unop(fx, op, operand, result_layout),
        Rvalue::Cast(kind, operand, target_ty) => {
            codegen_cast(fx, kind, operand, target_ty, result_layout)
        }
        Rvalue::Len(place) => {
            // For fixed-size arrays, length is a constant
            let body = fx.ra_body();
            let place_ty = place_ty(fx.db(), body, place);
            match place_ty.as_ref().kind() {
                TyKind::Array(_, len) => {
                    let len_val = match len.kind() {
                        ConstKind::Value(val) => {
                            let bytes = &val.value.inner().memory;
                            let mut buf = [0u8; 8];
                            let n = bytes.len().min(8);
                            buf[..n].copy_from_slice(&bytes[..n]);
                            u64::from_le_bytes(buf) as i64
                        }
                        _ => todo!("non-value const in array length"),
                    };
                    CValue::by_val(
                        fx.bcx.ins().iconst(fx.pointer_type, len_val),
                        result_layout.clone(),
                    )
                }
                _ => todo!("Len on non-array type"),
            }
        }
        Rvalue::Aggregate(_, _)
        | Rvalue::Ref(_, _)
        | Rvalue::AddressOf(_, _)
        | Rvalue::Discriminant(_) => {
            unreachable!("handled in codegen_assign")
        }
        _ => todo!("rvalue: {:?}", rvalue),
    }
}

fn codegen_aggregate(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: &hir_ty::mir::AggregateKind,
    operands: &[Operand],
    dest: CPlace,
) {
    use hir_ty::mir::AggregateKind;
    match kind {
        AggregateKind::Tuple(_) | AggregateKind::Array(_) | AggregateKind::Closure(_) | AggregateKind::Coroutine(_) | AggregateKind::CoroutineClosure(_) => {
            // For ScalarPair tuples, construct directly as a pair
            if let BackendRepr::ScalarPair(_, _) = dest.layout.backend_repr {
                assert_eq!(operands.len(), 2, "ScalarPair aggregate expects 2 operands");
                let val0 = codegen_operand(fx, &operands[0].kind).load_scalar(fx);
                let val1 = codegen_operand(fx, &operands[1].kind).load_scalar(fx);
                dest.write_cvalue(
                    fx,
                    CValue::by_val_pair(val0, val1, dest.layout.clone()),
                );
                return;
            }

            // For single scalar, construct directly
            if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
                assert_eq!(operands.len(), 1, "Scalar aggregate expects 1 operand");
                let val = codegen_operand(fx, &operands[0].kind);
                dest.write_cvalue(fx, val);
                return;
            }

            // General case: write each field to the destination place
            for (i, operand) in operands.iter().enumerate() {
                let field_cval = codegen_operand(fx, &operand.kind);
                let field_layout = field_cval.layout.clone();
                let field_place = dest.place_field(fx, i, field_layout);
                field_place.write_cvalue(fx, field_cval);
            }
        }
        AggregateKind::Adt(variant_id, _subst) => {
            use rustc_abi::Variants;
            let variant_idx = variant_id_to_idx(fx.db(), *variant_id);
            let is_single_variant = matches!(&dest.layout.variants, Variants::Single { .. });

            // Fast path: Scalar ADT with single variant (wrapper struct)
            if is_single_variant {
                if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
                    let non_zst: Vec<_> = operands
                        .iter()
                        .map(|op| codegen_operand(fx, &op.kind))
                        .filter(|cv| !cv.layout.is_zst())
                        .collect();
                    assert_eq!(non_zst.len(), 1, "Scalar ADT aggregate expects 1 non-ZST operand");
                    dest.write_cvalue(fx, non_zst.into_iter().next().unwrap());
                    codegen_set_discriminant(fx, &dest, variant_idx);
                    return;
                }

                // Fast path: ScalarPair ADT with single variant (two-field struct)
                if let BackendRepr::ScalarPair(_, _) = dest.layout.backend_repr {
                    let non_zst: Vec<_> = operands
                        .iter()
                        .map(|op| codegen_operand(fx, &op.kind))
                        .filter(|cv| !cv.layout.is_zst())
                        .collect();
                    assert_eq!(non_zst.len(), 2, "ScalarPair ADT aggregate expects 2 non-ZST operands");
                    let val0 = non_zst[0].load_scalar(fx);
                    let val1 = non_zst[1].load_scalar(fx);
                    dest.write_cvalue(
                        fx,
                        CValue::by_val_pair(val0, val1, dest.layout.clone()),
                    );
                    codegen_set_discriminant(fx, &dest, variant_idx);
                    return;
                }
            }

            // General case: for multi-variant enums on register places,
            // spill to memory so field projections use correct offsets.
            let use_temp = matches!(&dest.layout.variants, Variants::Multiple { .. })
                && dest.is_register();
            let (work_dest, original_dest) = if use_temp {
                let tmp = CPlace::new_stack_slot(fx, dest.layout.clone());
                (tmp, Some(dest))
            } else {
                (dest, None)
            };

            let variant_ly = match &work_dest.layout.variants {
                Variants::Single { .. } => work_dest.layout.clone(),
                Variants::Multiple { variants, .. } => TArc::new(variants[variant_idx].clone()),
                Variants::Empty => panic!("aggregate on empty variants"),
            };
            let variant_dest = work_dest.downcast_variant(variant_ly);

            for (i, operand) in operands.iter().enumerate() {
                let field_cval = codegen_operand(fx, &operand.kind);
                if field_cval.layout.is_zst() {
                    continue;
                }
                let field_layout = field_cval.layout.clone();
                let field_place = variant_dest.place_field(fx, i, field_layout);
                field_place.write_cvalue(fx, field_cval);
            }

            codegen_set_discriminant(fx, &work_dest, variant_idx);

            if let Some(orig) = original_dest {
                let cval = work_dest.to_cvalue(fx);
                orig.write_cvalue(fx, cval);
            }
        }
        AggregateKind::Union(_, _) => {
            todo!("Union aggregate")
        }
        AggregateKind::RawPtr(_, _) => {
            todo!("RawPtr aggregate")
        }
    }
}

/// Read the discriminant of an enum place.
/// Follows upstream cg_clif/src/discriminant.rs `codegen_get_discriminant`.
fn codegen_get_discriminant(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &CPlace,
    dest_layout: &LayoutArc,
) -> Value {
    use rustc_abi::Variants;
    let BackendRepr::Scalar(dest_scalar) = dest_layout.backend_repr else {
        panic!("discriminant destination must be scalar");
    };
    let dest_clif_ty = scalar_to_clif_type(fx.dl, &dest_scalar);

    match &place.layout.variants {
        Variants::Single { index } => {
            // TODO: Use db.const_eval_discriminant() for explicit discriminant values
            let discr_val = index.as_u32();
            fx.bcx.ins().iconst(dest_clif_ty, i64::from(discr_val))
        }
        Variants::Multiple { tag, tag_field, tag_encoding, .. } => {
            use rustc_abi::TagEncoding;
            let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);

            // Read the tag value — handle register and memory places
            let tag_val = match place.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    place.to_cvalue(fx).load_scalar(fx)
                }
                BackendRepr::ScalarPair(_, _) => {
                    let (a, b) = place.to_cvalue(fx).load_scalar_pair(fx);
                    if tag_field.as_usize() == 0 { a } else { b }
                }
                _ => {
                    let tag_offset = place.layout.fields.offset(tag_field.as_usize());
                    let tag_ptr = place.to_ptr().offset_i64(
                        &mut fx.bcx,
                        fx.pointer_type,
                        i64::try_from(tag_offset.bytes()).unwrap(),
                    );
                    let mut flags = MemFlags::new();
                    flags.set_notrap();
                    tag_ptr.load(&mut fx.bcx, tag_clif_ty, flags)
                }
            };

            match tag_encoding {
                TagEncoding::Direct => {
                    let signed = match tag.primitive() {
                        Primitive::Int(_, signed) => signed,
                        _ => false,
                    };
                    codegen_intcast(fx, tag_val, dest_clif_ty, signed)
                }
                TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                    let relative_max =
                        niche_variants.end().as_u32() - niche_variants.start().as_u32();

                    // Algorithm (from upstream):
                    // relative_tag = tag - niche_start
                    // is_niche = relative_tag <= (ule) relative_max
                    // discr = if is_niche {
                    //     cast(relative_tag) + niche_variants.start()
                    // } else {
                    //     untagged_variant
                    // }

                    let (is_niche, tagged_discr, delta) = if relative_max == 0 {
                        // Single niche variant: just compare tag == niche_start
                        let is_niche = codegen_icmp_imm(
                            fx,
                            IntCC::Equal,
                            tag_val,
                            *niche_start as i128,
                        );
                        let tagged_discr = fx.bcx.ins().iconst(
                            dest_clif_ty,
                            niche_variants.start().as_u32() as i64,
                        );
                        (is_niche, tagged_discr, 0)
                    } else {
                        // General case: compute relative_tag, check range
                        let niche_start_val =
                            fx.bcx.ins().iconst(tag_clif_ty, *niche_start as i64);
                        let relative_discr = fx.bcx.ins().isub(tag_val, niche_start_val);
                        let cast_tag =
                            codegen_intcast(fx, relative_discr, dest_clif_ty, false);
                        let is_niche = codegen_icmp_imm(
                            fx,
                            IntCC::UnsignedLessThanOrEqual,
                            relative_discr,
                            i128::from(relative_max),
                        );
                        (is_niche, cast_tag, niche_variants.start().as_u32() as u128)
                    };

                    let tagged_discr = if delta == 0 {
                        tagged_discr
                    } else {
                        let delta_val = fx.bcx.ins().iconst(dest_clif_ty, delta as i64);
                        fx.bcx.ins().iadd(tagged_discr, delta_val)
                    };

                    let untagged_variant_val = fx.bcx.ins().iconst(
                        dest_clif_ty,
                        i64::from(untagged_variant.as_u32()),
                    );
                    fx.bcx.ins().select(is_niche, tagged_discr, untagged_variant_val)
                }
            }
        }
        Variants::Empty => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
            fx.bcx.ins().iconst(dest_clif_ty, 0)
        }
    }
}

/// Set the discriminant for an enum place.
/// Follows upstream cg_clif/src/discriminant.rs `codegen_set_discriminant`.
fn codegen_set_discriminant(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &CPlace,
    variant_index: VariantIdx,
) {
    use rustc_abi::{TagEncoding, Variants};
    match &place.layout.variants {
        Variants::Single { index } => {
            assert_eq!(*index, variant_index);
        }
        Variants::Multiple { tag, tag_field, tag_encoding, .. } => {
            let tag_layout = tag_scalar_layout(fx.dl, tag);

            match tag_encoding {
                TagEncoding::Direct => {
                    let ptr = place.place_field(fx, tag_field.as_usize(), tag_layout);
                    let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);
                    // TODO: Use db.const_eval_discriminant() for explicit discriminant values
                    let discr_val = variant_index.as_u32();
                    let to = fx.bcx.ins().iconst(tag_clif_ty, i64::from(discr_val));
                    ptr.write_cvalue(fx, CValue::by_val(to, ptr.layout.clone()));
                }
                TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                    if variant_index != *untagged_variant {
                        let niche = place.place_field(fx, tag_field.as_usize(), tag_layout);
                        let niche_type = scalar_to_clif_type(fx.dl, tag);
                        let niche_value =
                            variant_index.as_u32() - niche_variants.start().as_u32();
                        let niche_value = (niche_value as u128).wrapping_add(*niche_start);
                        let niche_value =
                            fx.bcx.ins().iconst(niche_type, niche_value as i64);
                        niche.write_cvalue(
                            fx,
                            CValue::by_val(niche_value, niche.layout.clone()),
                        );
                    }
                }
            }
        }
        Variants::Empty => unreachable!("SetDiscriminant on empty variants"),
    }
}

fn codegen_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &BinOp,
    lhs: &Operand,
    rhs: &Operand,
    result_layout: &LayoutArc,
) -> CValue {
    let body = fx.ra_body();
    let lhs_cval = codegen_operand(fx, &lhs.kind);
    let rhs_cval = codegen_operand(fx, &rhs.kind);
    let lhs_val = lhs_cval.load_scalar(fx);
    let rhs_val = rhs_cval.load_scalar(fx);

    let BackendRepr::Scalar(scalar) = lhs_cval.layout.backend_repr else {
        panic!("expected scalar type for binop lhs");
    };

    // Overflow binops return (T, bool) as a ScalarPair
    if matches!(op, BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow) {
        let Primitive::Int(_, signed) = scalar.primitive() else {
            panic!("overflow binop on non-integer type");
        };
        let (res, has_overflow) = codegen_checked_int_binop(fx, op, lhs_val, rhs_val, signed);
        return CValue::by_val_pair(res, has_overflow, result_layout.clone());
    }

    let val = match scalar.primitive() {
        Primitive::Int(_, signed) => codegen_int_binop(fx, op, lhs_val, rhs_val, signed),
        Primitive::Float(_) => codegen_float_binop(fx, op, lhs_val, rhs_val),
        Primitive::Pointer(_) => match op {
            BinOp::Offset => {
                let lhs_ty = operand_ty(fx.db(), body, &lhs.kind);
                let pointee_ty = lhs_ty
                    .as_ref()
                    .builtin_deref(true)
                    .expect("Offset lhs must be a pointer/reference");
                let pointee_layout = fx.db()
                    .layout_of_ty(pointee_ty.store(), fx.env().clone())
                    .expect("layout error for pointee type");
                let ptr_ty = fx.bcx.func.dfg.value_type(lhs_val);
                let rhs_ty = fx.bcx.func.dfg.value_type(rhs_val);
                let rhs_signed = ty_is_signed_int(operand_ty(fx.db(), body, &rhs.kind));
                let rhs = if rhs_ty == ptr_ty {
                    rhs_val
                } else if ptr_ty.wider_or_equal(rhs_ty) {
                    if rhs_signed {
                        fx.bcx.ins().sextend(ptr_ty, rhs_val)
                    } else {
                        fx.bcx.ins().uextend(ptr_ty, rhs_val)
                    }
                } else {
                    fx.bcx.ins().ireduce(ptr_ty, rhs_val)
                };
                let byte_offset = fx.bcx.ins().imul_imm(rhs, pointee_layout.size.bytes() as i64);
                fx.bcx.ins().iadd(lhs_val, byte_offset)
            }
            BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
                let cc = bin_op_to_intcc(op, false);
                fx.bcx.ins().icmp(cc, lhs_val, rhs_val)
            }
            _ => todo!("pointer binop: {:?}", op),
        },
    };
    CValue::by_val(val, result_layout.clone())
}

fn codegen_int_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &BinOp,
    lhs: Value,
    rhs: Value,
    signed: bool,
) -> Value {
    let b = &mut fx.bcx;
    match op {
        BinOp::Add | BinOp::AddUnchecked => b.ins().iadd(lhs, rhs),
        BinOp::Sub | BinOp::SubUnchecked => b.ins().isub(lhs, rhs),
        BinOp::Mul | BinOp::MulUnchecked => b.ins().imul(lhs, rhs),
        BinOp::Div => {
            if signed {
                b.ins().sdiv(lhs, rhs)
            } else {
                b.ins().udiv(lhs, rhs)
            }
        }
        BinOp::Rem => {
            if signed {
                b.ins().srem(lhs, rhs)
            } else {
                b.ins().urem(lhs, rhs)
            }
        }
        BinOp::BitXor => b.ins().bxor(lhs, rhs),
        BinOp::BitAnd => b.ins().band(lhs, rhs),
        BinOp::BitOr => b.ins().bor(lhs, rhs),
        BinOp::Shl | BinOp::ShlUnchecked => b.ins().ishl(lhs, rhs),
        BinOp::Shr | BinOp::ShrUnchecked => {
            if signed {
                b.ins().sshr(lhs, rhs)
            } else {
                b.ins().ushr(lhs, rhs)
            }
        }
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
            let cc = bin_op_to_intcc(op, signed);
            b.ins().icmp(cc, lhs, rhs)
        }
        BinOp::Cmp => {
            // Three-way comparison: (lhs > rhs) as i8 - (lhs < rhs) as i8
            let (gt_cc, lt_cc) = if signed {
                (IntCC::SignedGreaterThan, IntCC::SignedLessThan)
            } else {
                (IntCC::UnsignedGreaterThan, IntCC::UnsignedLessThan)
            };
            let gt = b.ins().icmp(gt_cc, lhs, rhs);
            let lt = b.ins().icmp(lt_cc, lhs, rhs);
            b.ins().isub(gt, lt)
        }
        BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow => {
            unreachable!("overflow binops handled by codegen_checked_int_binop")
        }
        BinOp::Offset => unreachable!("Offset on integer type"),
    }
}

fn codegen_checked_int_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &BinOp,
    lhs: Value,
    rhs: Value,
    signed: bool,
) -> (Value, Value) {
    let b = &mut fx.bcx;
    match op {
        BinOp::AddWithOverflow => {
            let val = b.ins().iadd(lhs, rhs);
            let has_overflow = if !signed {
                b.ins().icmp(IntCC::UnsignedLessThan, val, lhs)
            } else {
                let rhs_is_negative = b.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let slt = b.ins().icmp(IntCC::SignedLessThan, val, lhs);
                b.ins().bxor(rhs_is_negative, slt)
            };
            (val, has_overflow)
        }
        BinOp::SubWithOverflow => {
            let val = b.ins().isub(lhs, rhs);
            let has_overflow = if !signed {
                b.ins().icmp(IntCC::UnsignedGreaterThan, val, lhs)
            } else {
                let rhs_is_negative = b.ins().icmp_imm(IntCC::SignedLessThan, rhs, 0);
                let sgt = b.ins().icmp(IntCC::SignedGreaterThan, val, lhs);
                b.ins().bxor(rhs_is_negative, sgt)
            };
            (val, has_overflow)
        }
        BinOp::MulWithOverflow => {
            let ty = b.func.dfg.value_type(lhs);
            match ty {
                types::I8 | types::I16 | types::I32 if !signed => {
                    let wide = ty.double_width().unwrap();
                    let lhs_w = b.ins().uextend(wide, lhs);
                    let rhs_w = b.ins().uextend(wide, rhs);
                    let val_w = b.ins().imul(lhs_w, rhs_w);
                    let has_overflow = b.ins().icmp_imm(
                        IntCC::UnsignedGreaterThan,
                        val_w,
                        (1i64 << ty.bits()) - 1,
                    );
                    let val = b.ins().ireduce(ty, val_w);
                    (val, has_overflow)
                }
                types::I8 | types::I16 | types::I32 if signed => {
                    let wide = ty.double_width().unwrap();
                    let lhs_w = b.ins().sextend(wide, lhs);
                    let rhs_w = b.ins().sextend(wide, rhs);
                    let val_w = b.ins().imul(lhs_w, rhs_w);
                    let has_underflow = b.ins().icmp_imm(
                        IntCC::SignedLessThan,
                        val_w,
                        -(1i64 << (ty.bits() - 1)),
                    );
                    let has_overflow = b.ins().icmp_imm(
                        IntCC::SignedGreaterThan,
                        val_w,
                        (1i64 << (ty.bits() - 1)) - 1,
                    );
                    let val = b.ins().ireduce(ty, val_w);
                    (val, b.ins().bor(has_underflow, has_overflow))
                }
                types::I64 => {
                    let val = b.ins().imul(lhs, rhs);
                    let has_overflow = if !signed {
                        let val_hi = b.ins().umulhi(lhs, rhs);
                        b.ins().icmp_imm(IntCC::NotEqual, val_hi, 0)
                    } else {
                        let val_hi = b.ins().smulhi(lhs, rhs);
                        let val_sign = b.ins().sshr_imm(val, i64::from(ty.bits() - 1));
                        let xor = b.ins().bxor(val_hi, val_sign);
                        b.ins().icmp_imm(IntCC::NotEqual, xor, 0)
                    };
                    (val, has_overflow)
                }
                _ => todo!("checked mul for {ty:?}"),
            }
        }
        _ => unreachable!("not a checked binop: {:?}", op),
    }
}

fn codegen_float_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &BinOp,
    lhs: Value,
    rhs: Value,
) -> Value {
    let b = &mut fx.bcx;
    match op {
        BinOp::Add | BinOp::AddUnchecked => b.ins().fadd(lhs, rhs),
        BinOp::Sub | BinOp::SubUnchecked => b.ins().fsub(lhs, rhs),
        BinOp::Mul | BinOp::MulUnchecked => b.ins().fmul(lhs, rhs),
        BinOp::Div => b.ins().fdiv(lhs, rhs),
        BinOp::Rem => {
            let ty = b.func.dfg.value_type(lhs);
            let name = match ty {
                types::F32 => "fmodf",
                types::F64 => "fmod",
                _ => todo!("float rem for {ty:?}"),
            };
            codegen_libcall1(fx, name, &[ty, ty], ty, &[lhs, rhs])
        }
        BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
            let cc = bin_op_to_floatcc(op);
            b.ins().fcmp(cc, lhs, rhs)
        }
        _ => todo!("float binop: {:?}", op),
    }
}

fn codegen_unop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &UnOp,
    operand: &Operand,
    result_layout: &LayoutArc,
) -> CValue {
    let body = fx.ra_body();
    let cval = codegen_operand(fx, &operand.kind);
    let val = cval.load_scalar(fx);
    let result = match op {
        UnOp::Not => {
            let ty = operand_ty(fx.db(), body, &operand.kind);
            if ty.as_ref().kind() == TyKind::Bool {
                fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0)
            } else {
                fx.bcx.ins().bnot(val)
            }
        }
        UnOp::Neg => {
            let BackendRepr::Scalar(scalar) = cval.layout.backend_repr else {
                panic!("neg on non-scalar");
            };
            match scalar.primitive() {
                Primitive::Int(_, _) => fx.bcx.ins().ineg(val),
                Primitive::Float(_) => fx.bcx.ins().fneg(val),
                Primitive::Pointer(_) => unreachable!("neg on pointer"),
            }
        }
    };
    CValue::by_val(result, result_layout.clone())
}

fn codegen_operand(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: &OperandKind,
) -> CValue {
    match kind {
        OperandKind::Constant { konst, ty } => {
            let layout = fx.db()
                .layout_of_ty(ty.clone(), fx.env().clone())
                .expect("layout error for constant type");
            if layout.is_zst() {
                return CValue::zst(layout);
            }
            match layout.backend_repr {
                BackendRepr::Scalar(scalar) => {
                    let raw = const_to_i64(konst.as_ref(), scalar.size(fx.dl));
                    let val = match scalar.primitive() {
                        Primitive::Float(rustc_abi::Float::F32) => {
                            fx.bcx.ins().f32const(f32::from_bits(raw as u32))
                        }
                        Primitive::Float(rustc_abi::Float::F64) => {
                            fx.bcx.ins().f64const(f64::from_bits(raw as u64))
                        }
                        Primitive::Float(_) => todo!("f16/f128 constants"),
                        _ => {
                            let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                            fx.bcx.ins().iconst(clif_ty, raw)
                        }
                    };
                    CValue::by_val(val, layout)
                }
                BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                    let ConstKind::Value(val) = konst.as_ref().kind() else {
                        panic!("non-value const in ScalarPair constant");
                    };
                    let bytes = &val.value.inner().memory;
                    let a_size = a_scalar.size(fx.dl).bytes() as usize;
                    let b_offset = a_scalar.size(fx.dl).align_to(b_scalar.align(fx.dl).abi).bytes() as usize;
                    let b_size = b_scalar.size(fx.dl).bytes() as usize;

                    let a_raw = {
                        let mut buf = [0u8; 8];
                        let len = a_size.min(8);
                        buf[..len].copy_from_slice(&bytes[..len]);
                        i64::from_le_bytes(buf)
                    };
                    let b_raw = {
                        let mut buf = [0u8; 8];
                        let len = b_size.min(8);
                        buf[..len].copy_from_slice(&bytes[b_offset..b_offset + len]);
                        i64::from_le_bytes(buf)
                    };

                    let a_clif = scalar_to_clif_type(fx.dl, &a_scalar);
                    let b_clif = scalar_to_clif_type(fx.dl, &b_scalar);
                    let a_val = fx.bcx.ins().iconst(a_clif, a_raw);
                    let b_val = fx.bcx.ins().iconst(b_clif, b_raw);
                    CValue::by_val_pair(a_val, b_val, layout)
                }
                _ => {
                    // Memory-repr constant: store raw bytes in a data section
                    let ConstKind::Value(val) = konst.as_ref().kind() else {
                        panic!("non-value const in memory-repr constant");
                    };
                    let bytes = &val.value.inner().memory;
                    let mut data_desc = DataDescription::new();
                    data_desc.define(bytes.to_vec().into_boxed_slice());
                    data_desc.set_align(layout.align.abi.bytes());
                    let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                    fx.module.define_data(data_id, &data_desc).unwrap();
                    let local_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                    let ptr = fx.bcx.ins().symbol_value(fx.pointer_type, local_id);
                    CValue::by_ref(pointer::Pointer::new(ptr), layout)
                }
            }
        }
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            codegen_place(fx, place).to_cvalue(fx)
        }
        OperandKind::Static(_) => todo!("static operand"),
    }
}

// ---------------------------------------------------------------------------
// Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_terminator(fx: &mut FunctionCx<'_, impl Module>, term: &TerminatorKind) {
    match term {
        TerminatorKind::Return => {
            let ret_place = fx.local_place(hir_ty::mir::return_slot()).clone();
            if ret_place.layout.is_zst() {
                fx.bcx.ins().return_(&[]);
            } else {
                match ret_place.layout.backend_repr {
                    BackendRepr::Scalar(_) => {
                        let cval = ret_place.to_cvalue(fx);
                        let val = cval.load_scalar(fx);
                        fx.bcx.ins().return_(&[val]);
                    }
                    BackendRepr::ScalarPair(_, _) => {
                        let cval = ret_place.to_cvalue(fx);
                        let (a, b) = cval.load_scalar_pair(fx);
                        fx.bcx.ins().return_(&[a, b]);
                    }
                    _ => {
                        // Memory-repr return: value already written to sret pointer
                        fx.bcx.ins().return_(&[]);
                    }
                }
            }
        }
        TerminatorKind::Goto { target } => {
            let block = fx.clif_block(*target);
            fx.bcx.ins().jump(block, &[]);
        }
        TerminatorKind::Unreachable => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
        }
        TerminatorKind::SwitchInt { discr, targets } => {
            let discr_cval = codegen_operand(fx, &discr.kind);
            let discr_val = discr_cval.load_scalar(fx);
            let otherwise = fx.clif_block(targets.otherwise());

            let mut switch = Switch::new();
            for (val, target) in targets.iter() {
                let block = fx.clif_block(target);
                switch.set_entry(val, block);
            }
            switch.emit(&mut fx.bcx, discr_val, otherwise);
        }
        TerminatorKind::Call { func, args, destination, target, .. } => {
            codegen_call(fx, func, args, destination, target);
        }
        TerminatorKind::Drop { place, target, .. } => {
            codegen_drop(fx, place, *target);
        }
        _ => todo!("terminator: {:?}", term),
    }
}

/// Codegen for `TerminatorKind::Drop`.
///
/// If the type has drop glue (direct Drop impl or fields that need dropping),
/// calls the generated `drop_in_place::<T>` function. Otherwise, this is a
/// no-op jump to the target block.
///
/// Reference: cg_clif/src/abi/mod.rs `codegen_drop`
fn codegen_drop(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &Place,
    target: BasicBlockId,
) {
    let target_block = fx.clif_block(target);
    let body = fx.ra_body();
    let ty = place_ty(fx.db(), body, place);

    let interner = DbInterner::new_with(fx.db(), fx.local_crate());
    if !hir_ty::drop::has_drop_glue_mono(interner, ty.as_ref()) {
        fx.bcx.ins().jump(target_block, &[]);
        return;
    }

    // Get a pointer to the place (drop takes *mut T / &mut self).
    // Spills register-stored places to the stack if necessary.
    let drop_place = codegen_place(fx, place);
    let ptr = drop_place.to_ptr_maybe_spill(fx).get_addr(&mut fx.bcx, fx.pointer_type);

    let mut drop_sig = Signature::new(fx.isa.default_call_conv());
    drop_sig.params.push(AbiParam::new(fx.pointer_type));

    // Optimization: if the type has a direct Drop impl and no fields need
    // recursive dropping, call Drop::drop directly (avoiding the
    // drop_in_place wrapper). Otherwise, call drop_in_place::<T>.
    let lang_items = hir_def::lang_item::lang_items(fx.db(), fx.local_crate());
    let direct_drop = lang_items.Drop.and_then(|drop_trait| {
        resolve_drop_impl(fx.db(), fx.local_crate(), drop_trait, &ty)
    });
    let needs_field_drops = type_has_droppable_fields(fx.db(), fx.local_crate(), &ty);

    let fn_name = if let (Some(drop_func_id), false) = (direct_drop, needs_field_drops) {
        // Simple case: just call Drop::drop directly
        let adt_subst = match ty.as_ref().kind() {
            TyKind::Adt(_, subst) => Some(subst.store()),
            _ => None,
        };
        let interner = DbInterner::new_no_crate(fx.db());
        let generic_args = adt_subst.unwrap_or_else(|| GenericArgs::empty(interner).store());
        symbol_mangling::mangle_function(
            fx.db(), drop_func_id, generic_args.as_ref(), fx.ext_crate_disambiguators(),
        )
    } else {
        // Needs recursive field drops — use drop_in_place glue
        symbol_mangling::mangle_drop_in_place(
            fx.db(), ty.as_ref(), fx.ext_crate_disambiguators(),
        )
    };

    let callee_id = fx.module
        .declare_function(&fn_name, Linkage::Import, &drop_sig)
        .expect("declare drop fn");
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
    fx.bcx.ins().call(callee_ref, &[ptr]);

    fx.bcx.ins().jump(target_block, &[]);
}

/// Check whether a type has fields that themselves need dropping.
/// Returns true if any field (struct fields, tuple elements, enum variant
/// fields, closure captures) has drop glue, requiring a `drop_in_place`
/// wrapper rather than a simple `Drop::drop` call.
fn type_has_droppable_fields(
    db: &dyn HirDatabase,
    krate: base_db::Crate,
    ty: &StoredTy,
) -> bool {
    let interner = DbInterner::new_with(db, krate);
    match ty.as_ref().kind() {
        TyKind::Adt(adt_def, subst) => {
            let adt_id = adt_def.inner().id;
            match adt_id {
                hir_def::AdtId::StructId(id) => {
                    use hir_def::signatures::StructFlags;
                    if db.struct_signature(id).flags.intersects(
                        StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA,
                    ) {
                        return false;
                    }
                    db.field_types(id.into()).iter().any(|(_, field_ty)| {
                        hir_ty::drop::has_drop_glue_mono(
                            interner,
                            field_ty.get().instantiate(interner, subst),
                        )
                    })
                }
                hir_def::AdtId::UnionId(_) => false,
                hir_def::AdtId::EnumId(id) => {
                    id.enum_variants(db).variants.iter().any(|&(variant, _, _)| {
                        db.field_types(variant.into()).iter().any(|(_, field_ty)| {
                            hir_ty::drop::has_drop_glue_mono(
                                interner,
                                field_ty.get().instantiate(interner, subst),
                            )
                        })
                    })
                }
            }
        }
        TyKind::Tuple(tys) => {
            tys.iter().any(|elem_ty| hir_ty::drop::has_drop_glue_mono(interner, elem_ty))
        }
        TyKind::Closure(closure_id, subst) => {
            let owner = db.lookup_intern_closure(closure_id.0).0;
            let infer = hir_ty::InferenceResult::for_body(db, owner);
            let (captures, _) = infer.closure_info(closure_id.0);
            captures.iter().any(|capture| {
                hir_ty::drop::has_drop_glue_mono(interner, capture.ty(db, subst))
            })
        }
        _ => false,
    }
}

/// Resolve the `Drop::drop` impl method for a given type, if any.
///
/// Returns `Some(FunctionId)` for the impl's `drop` method if the type
/// directly implements the `Drop` trait. Returns `None` if the type does
/// not have a `Drop` impl.
fn resolve_drop_impl(
    db: &dyn HirDatabase,
    krate: base_db::Crate,
    drop_trait: TraitId,
    ty: &StoredTy,
) -> Option<hir_def::FunctionId> {
    use hir_expand::name::Name;
    use intern::sym;

    // Only ADTs can have Drop impls
    let TyKind::Adt(adt_def, _) = ty.as_ref().kind() else {
        return None;
    };

    let interner = DbInterner::new_no_crate(db);

    // Check if the ADT has a Drop impl
    use rustc_type_ir::fast_reject::{TreatParams, simplify_type};
    let simplified = simplify_type(interner, ty.as_ref(), TreatParams::InstantiateWithInfer)?;

    // Search in the krate where the type is defined and our local crate
    let adt_id = adt_def.inner().id;
    let type_krate = match adt_id {
        hir_def::AdtId::StructId(id) => id.krate(db),
        hir_def::AdtId::EnumId(id) => id.krate(db),
        hir_def::AdtId::UnionId(id) => id.krate(db),
    };

    for search_krate in [krate, type_krate] {
        let trait_impls = TraitImpls::for_crate(db, search_krate);
        let (impl_ids, _) = trait_impls.for_trait_and_self_ty(drop_trait, &simplified);
        if let Some(&impl_id) = impl_ids.first() {
            // Found the Drop impl — look up the `drop` method
            let impl_items = impl_id.impl_items(db);
            let drop_method = impl_items.items.iter().find_map(|(name, item)| {
                if *name == Name::new_symbol_root(sym::drop) {
                    match item {
                        AssocItemId::FunctionId(fid) => Some(*fid),
                        _ => None,
                    }
                } else {
                    None
                }
            });
            if let Some(func_id) = drop_method {
                return Some(func_id);
            }
        }
    }

    None
}

fn codegen_call(
    fx: &mut FunctionCx<'_, impl Module>,
    func: &Operand,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Extract the function type from any operand (constant or non-constant)
    let body = fx.ra_body();
    let fn_ty_stored = operand_ty(fx.db(), body, &func.kind);

    match fn_ty_stored.as_ref().kind() {
        TyKind::FnDef(def, generic_args) => {
            let callable_def: CallableDefId = def.0;
            match callable_def {
                CallableDefId::FunctionId(callee_func_id) => {
                    codegen_direct_call(
                        fx,
                        callee_func_id,
                        generic_args,
                        args,
                        destination,
                        target,
                    );
                }
                CallableDefId::StructId(struct_id) => {
                    codegen_adt_constructor_call(
                        fx, VariantId::StructId(struct_id), generic_args,
                        args, destination, target,
                    );
                }
                CallableDefId::EnumVariantId(variant_id) => {
                    codegen_adt_constructor_call(
                        fx, VariantId::EnumVariantId(variant_id), generic_args,
                        args, destination, target,
                    );
                }
            }
        }
        TyKind::FnPtr(sig_tys, _header) => {
            codegen_fn_ptr_call(fx, func, &sig_tys, args, destination, target);
        }
        _ => todo!("non-FnDef/FnPtr call: {:?}", fn_ty_stored),
    }
}

/// Handle struct/enum variant constructor "calls" — these aren't real function
/// calls but rather aggregate construction lowered as Call terminators.
fn codegen_adt_constructor_call(
    fx: &mut FunctionCx<'_, impl Module>,
    variant_id: VariantId,
    _generic_args: GenericArgs<'_>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    use rustc_abi::Variants;
    let dest = codegen_place(fx, destination);
    if !dest.layout.is_zst() {
        let variant_idx = variant_id_to_idx(fx.db(), variant_id);
        let is_single_variant = matches!(&dest.layout.variants, Variants::Single { .. });

        // Fast path: Scalar ADT with single variant
        if is_single_variant {
            if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
                let non_zst: Vec<_> = args
                    .iter()
                    .map(|op| codegen_operand(fx, &op.kind))
                    .filter(|cv| !cv.layout.is_zst())
                    .collect();
                assert_eq!(non_zst.len(), 1);
                dest.write_cvalue(fx, non_zst.into_iter().next().unwrap());
                codegen_set_discriminant(fx, &dest, variant_idx);
                if let Some(target) = target {
                    let block = fx.clif_block(*target);
                    fx.bcx.ins().jump(block, &[]);
                }
                return;
            }
            if let BackendRepr::ScalarPair(_, _) = dest.layout.backend_repr {
                let non_zst: Vec<_> = args
                    .iter()
                    .map(|op| codegen_operand(fx, &op.kind))
                    .filter(|cv| !cv.layout.is_zst())
                    .collect();
                assert_eq!(non_zst.len(), 2);
                let val0 = non_zst[0].load_scalar(fx);
                let val1 = non_zst[1].load_scalar(fx);
                dest.write_cvalue(fx, CValue::by_val_pair(val0, val1, dest.layout.clone()));
                codegen_set_discriminant(fx, &dest, variant_idx);
                if let Some(target) = target {
                    let block = fx.clif_block(*target);
                    fx.bcx.ins().jump(block, &[]);
                }
                return;
            }
        }

        // For multi-variant enums on Var/VarPair places, use a temp stack
        // slot so field projections use correct memory offsets, then write
        // back to the original variable.
        let use_temp = matches!(&dest.layout.variants, Variants::Multiple { .. })
            && dest.is_register();
        let (work_dest, original_dest) = if use_temp {
            let tmp = CPlace::new_stack_slot(fx, dest.layout.clone());
            (tmp, Some(dest))
        } else {
            (dest, None)
        };

        let variant_ly = match &work_dest.layout.variants {
            Variants::Single { .. } => work_dest.layout.clone(),
            Variants::Multiple { variants, .. } => TArc::new(variants[variant_idx].clone()),
            Variants::Empty => panic!("constructor on empty variants"),
        };
        let variant_dest = work_dest.downcast_variant(variant_ly);

        for (i, arg) in args.iter().enumerate() {
            let field_cval = codegen_operand(fx, &arg.kind);
            if field_cval.layout.is_zst() {
                continue;
            }
            let field_layout = field_cval.layout.clone();
            let field_place = variant_dest.place_field(fx, i, field_layout);
            field_place.write_cvalue(fx, field_cval);
        }

        codegen_set_discriminant(fx, &work_dest, variant_idx);

        // Read back from stack slot into the original variable place
        if let Some(orig) = original_dest {
            let cval = work_dest.to_cvalue(fx);
            orig.write_cvalue(fx, cval);
        }
    }

    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    }
}

/// Indirect call through a fn pointer (`TyKind::FnPtr`).
/// Loads the fn pointer value, builds a signature from the FnPtr type,
/// and emits `call_indirect`.
fn codegen_fn_ptr_call(
    fx: &mut FunctionCx<'_, impl Module>,
    func_operand: &Operand,
    sig_tys: &rustc_type_ir::Binder<DbInterner, rustc_type_ir::FnSigTys<DbInterner>>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Load the fn pointer value
    let fn_ptr_cval = codegen_operand(fx, &func_operand.kind);
    let fn_ptr = fn_ptr_cval.load_scalar(fx);

    // Build signature from FnPtr type info
    let sig_tys_inner = sig_tys.clone().skip_binder();
    let mut sig = Signature::new(fx.isa.default_call_conv());

    // Return type
    let output = sig_tys_inner.output();
    let dest = codegen_place(fx, destination);
    let is_sret_return = if output.is_never() || dest.layout.is_zst() {
        false
    } else {
        match dest.layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &scalar)));
                false
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &a)));
                sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &b)));
                false
            }
            _ => {
                sig.params.push(AbiParam::special(fx.pointer_type, ArgumentPurpose::StructReturn));
                true
            }
        }
    };

    // Parameter types in signature
    for &param_ty in sig_tys_inner.inputs() {
        let param_layout = fx.db().layout_of_ty(param_ty.store(), fx.env().clone())
            .expect("fn ptr param layout");
        match param_layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &scalar)));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &a)));
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &b)));
            }
            _ if param_layout.is_zst() => {}
            _ => {
                sig.params.push(AbiParam::new(fx.pointer_type));
            }
        }
    }

    let sig_ref = fx.bcx.import_signature(sig);

    // Build argument values
    let mut call_args: Vec<Value> = Vec::new();

    let sret_slot = if is_sret_return {
        let slot = CPlace::new_stack_slot(fx, dest.layout.clone());
        let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
        call_args.push(ptr);
        Some(slot)
    } else {
        None
    };

    for arg in args {
        let cval = codegen_operand(fx, &arg.kind);
        if cval.layout.is_zst() {
            continue;
        }
        match cval.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = cval.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            BackendRepr::Scalar(_) => {
                call_args.push(cval.load_scalar(fx));
            }
            _ => {
                let ptr = cval.force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }

    // Emit indirect call
    let call = fx.bcx.ins().call_indirect(sig_ref, fn_ptr, &call_args);

    // Store return value
    if let Some(sret_slot) = sret_slot {
        let cval = sret_slot.to_cvalue(fx);
        dest.write_cvalue(fx, cval);
    } else {
        let results = fx.bcx.inst_results(call);
        if !dest.layout.is_zst() {
            match dest.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    let val = results.first().copied()
                        .expect("call_indirect returns no values but destination expects Scalar");
                    dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
                }
                BackendRepr::ScalarPair(_, _) => {
                    assert!(results.len() >= 2);
                    dest.write_cvalue(
                        fx,
                        CValue::by_val_pair(results[0], results[1], dest.layout.clone()),
                    );
                }
                _ => unreachable!("non-sret memory return from call_indirect"),
            }
        }
    }

    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    } else {
        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
    }
}

/// Call a closure body directly.
///
/// When we detect a call to `Fn::call` / `FnMut::call_mut` / `FnOnce::call_once`
/// where the self type is a concrete closure, we redirect to the closure's own
/// MIR body instead of going through trait dispatch.
fn codegen_closure_call(
    fx: &mut FunctionCx<'_, impl Module>,
    closure_id: hir_ty::db::InternedClosureId,
    closure_subst: StoredGenericArgs,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Get closure MIR body to build the signature
    let closure_body = fx.db()
        .monomorphized_mir_body_for_closure(closure_id, closure_subst, fx.env().clone())
        .expect("closure MIR");
    let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body, &[])
        .expect("closure sig");

    // Generate mangled name
    let closure_name = symbol_mangling::mangle_closure(
        fx.db(), closure_id, fx.ext_crate_disambiguators(),
    );

    // Declare in module (Import linkage — defined elsewhere in same module)
    let callee_id = fx.module
        .declare_function(&closure_name, Linkage::Import, &sig)
        .expect("declare closure");
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    // Destination
    let dest = codegen_place(fx, destination);
    let is_sret_return = !dest.layout.is_zst()
        && !matches!(
            dest.layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
        );

    // Build argument values
    let mut call_args: Vec<Value> = Vec::new();

    let sret_slot = if is_sret_return {
        let slot = CPlace::new_stack_slot(fx, dest.layout.clone());
        let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
        call_args.push(ptr);
        Some(slot)
    } else {
        None
    };

    for arg in args {
        let cval = codegen_operand(fx, &arg.kind);
        if cval.layout.is_zst() {
            continue;
        }
        match cval.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = cval.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            BackendRepr::Scalar(_) => {
                call_args.push(cval.load_scalar(fx));
            }
            _ => {
                let ptr = cval.force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);

    // Store return value
    if let Some(sret_slot) = sret_slot {
        let cval = sret_slot.to_cvalue(fx);
        dest.write_cvalue(fx, cval);
    } else {
        let results = fx.bcx.inst_results(call);
        if !dest.layout.is_zst() {
            match dest.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    dest.write_cvalue(fx, CValue::by_val(results[0], dest.layout.clone()));
                }
                BackendRepr::ScalarPair(_, _) => {
                    dest.write_cvalue(
                        fx,
                        CValue::by_val_pair(results[0], results[1], dest.layout.clone()),
                    );
                }
                _ => unreachable!("non-sret memory return from closure call"),
            }
        }
    }

    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    } else {
        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
    }
}

fn codegen_direct_call(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    if codegen_intrinsic_call(fx, callee_func_id, generic_args, args, destination, target) {
        return;
    }

    // Check for virtual dispatch: trait method called on dyn Trait
    if let ItemContainerId::TraitId(trait_id) = callee_func_id.loc(fx.db()).container {
        let interner = DbInterner::new_no_crate(fx.db());
        if hir_ty::method_resolution::is_dyn_method(
            interner,
            fx.env().param_env(),
            callee_func_id,
            generic_args,
        ).is_some() {
            codegen_virtual_call(fx, callee_func_id, trait_id, args, destination, target);
            return;
        }

        // Check for closure call: Fn::call / FnMut::call_mut / FnOnce::call_once
        // where self type is a closure — redirect to the closure's MIR body.
        if generic_args.len() > 0 {
            let self_ty = generic_args.type_at(0);
            if let TyKind::Closure(closure_id, closure_subst) = self_ty.kind() {
                codegen_closure_call(
                    fx, closure_id.0, closure_subst.store(),
                    args, destination, target,
                );
                return;
            }
        }
    }

    // Check if this is an extern function (no MIR available)
    let is_extern = matches!(
        callee_func_id.loc(fx.db()).container,
        ItemContainerId::ExternBlockId(_)
    );
    let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();

    let (callee_sig, callee_name) = if is_extern {
        // Extern functions: build signature from type info, use raw symbol name
        let sig = build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id)
            .expect("extern fn sig");
        let name = fx.db().function_signature(callee_func_id).name.as_str().to_owned();
        (sig, name)
    } else if is_cross_crate {
        // Cross-crate Rust functions: build signature from type info,
        // use v0 mangled name with real disambiguator from rlib
        let sig = build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id)
            .expect("cross-crate fn sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args,
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    } else {
        // Local functions: build signature from MIR, use v0 mangled name
        let callee_body = fx.db()
            .monomorphized_mir_body(callee_func_id.into(), generic_args.store(), fx.env().clone())
            .expect("failed to get callee MIR");
        let sig =
            build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body, &[]).expect("callee sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args,
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    };

    // Declare callee in module (Import linkage — it may be defined elsewhere or in same module)
    let callee_id = fx
        .module
        .declare_function(&callee_name, Linkage::Import, &callee_sig)
        .expect("declare callee");

    // Import into current function to get a FuncRef
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    // Determine destination layout to check for sret return
    let dest = codegen_place(fx, destination);
    let is_sret_return = !dest.layout.is_zst()
        && !matches!(
            dest.layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
        );

    // Build argument values (skip ZST args that have no Cranelift representation)
    let mut call_args: Vec<Value> = Vec::new();

    // If sret return: allocate stack slot for result, pass pointer as first arg
    let sret_slot = if is_sret_return {
        let slot = CPlace::new_stack_slot(fx, dest.layout.clone());
        let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
        call_args.push(ptr);
        Some(slot)
    } else {
        None
    };

    for arg in args {
        let cval = codegen_operand(fx, &arg.kind);
        if cval.layout.is_zst() {
            continue;
        }
        match cval.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = cval.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            BackendRepr::Scalar(_) => {
                call_args.push(cval.load_scalar(fx));
            }
            _ => {
                // Memory-repr arg: force to stack and pass pointer
                let ptr = cval.force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);

    // Store return value into destination place
    if let Some(sret_slot) = sret_slot {
        // sret: result is in the stack slot, copy to destination
        let cval = sret_slot.to_cvalue(fx);
        dest.write_cvalue(fx, cval);
    } else {
        let results = fx.bcx.inst_results(call);
        if !dest.layout.is_zst() {
            match dest.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    let val = results.first().copied()
                        .expect("call returns no values but destination expects Scalar");
                    dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
                }
                BackendRepr::ScalarPair(_, _) => {
                    assert!(results.len() >= 2, "call returns fewer than 2 values but destination expects ScalarPair");
                    dest.write_cvalue(
                        fx,
                        CValue::by_val_pair(results[0], results[1], dest.layout.clone()),
                    );
                }
                _ => unreachable!("non-sret memory return"),
            }
        }
    }

    // Jump to continuation block (or trap for diverging calls)
    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    } else {
        // Diverging call (returns `!`) — the callee never returns,
        // but Cranelift requires the block to be terminated.
        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
    }
}

/// Virtual dispatch: load fn ptr from vtable, call indirectly.
/// Reference: cg_clif/src/vtable.rs `get_ptr_and_method_ref` + cg_clif/src/abi/mod.rs:525-543
fn codegen_virtual_call(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_func_id: hir_def::FunctionId,
    trait_id: TraitId,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    let ptr_size = fx.dl.pointer_size().bytes();

    // Compute vtable index: 3 (header slots) + method position in trait
    let trait_items = trait_id.trait_items(fx.db());
    let callee_name = fx.db().function_signature(callee_func_id).name.clone();
    let mut method_idx = 0;
    let mut found = false;
    for (name, item) in trait_items.items.iter() {
        if let AssocItemId::FunctionId(fid) = item {
            if *fid == callee_func_id {
                found = true;
                break;
            }
            if *name == callee_name {
                found = true;
                break;
            }
            method_idx += 1;
        }
    }
    assert!(found, "method `{}` not found in trait", callee_name.as_str());

    let vtable_offset = (3 + method_idx) * ptr_size as usize;

    // Get self arg (&dyn Trait = ScalarPair(data_ptr, vtable_ptr))
    let self_cval = codegen_operand(fx, &args[0].kind);
    let (data_ptr, vtable_ptr) = self_cval.load_scalar_pair(fx);

    // Load fn ptr from vtable
    let fn_ptr = fx.bcx.ins().load(
        fx.pointer_type,
        vtable_memflags(),
        vtable_ptr,
        vtable_offset as i32,
    );

    // Build the indirect call signature:
    // - self param is a thin pointer (data_ptr)
    // - other params from the remaining args
    // - return type from destination layout
    let mut sig = Signature::new(fx.isa.default_call_conv());

    // Return type
    let dest = codegen_place(fx, destination);
    let is_sret_return = !dest.layout.is_zst()
        && !matches!(
            dest.layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
        );
    match dest.layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &scalar)));
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &a)));
            sig.returns.push(AbiParam::new(scalar_to_clif_type(fx.dl, &b)));
        }
        _ if dest.layout.is_zst() => {}
        _ => {
            // Memory-repr return: sret pointer as first param
            sig.params.push(AbiParam::special(fx.pointer_type, ArgumentPurpose::StructReturn));
        }
    }

    // Self param: thin pointer
    sig.params.push(AbiParam::new(fx.pointer_type));

    // Build call args
    let mut call_args: Vec<Value> = Vec::new();

    // If sret return: allocate stack slot, pass pointer as first arg
    let sret_slot = if is_sret_return {
        let slot = CPlace::new_stack_slot(fx, dest.layout.clone());
        let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
        call_args.push(ptr);
        Some(slot)
    } else {
        None
    };

    // data_ptr (thin self)
    call_args.push(data_ptr);

    // Remaining args (after self)
    for arg in &args[1..] {
        let cval = codegen_operand(fx, &arg.kind);
        if cval.layout.is_zst() {
            continue;
        }
        match cval.layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &scalar)));
                call_args.push(cval.load_scalar(fx));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &a)));
                sig.params.push(AbiParam::new(scalar_to_clif_type(fx.dl, &b)));
                let (va, vb) = cval.load_scalar_pair(fx);
                call_args.push(va);
                call_args.push(vb);
            }
            _ => {
                // Memory-repr arg: force to stack and pass pointer
                sig.params.push(AbiParam::new(fx.pointer_type));
                let ptr = cval.force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }

    // Emit indirect call
    let sig_ref = fx.bcx.import_signature(sig);
    let call = fx.bcx.ins().call_indirect(sig_ref, fn_ptr, &call_args);

    // Store return value
    if let Some(sret_slot) = sret_slot {
        let cval = sret_slot.to_cvalue(fx);
        dest.write_cvalue(fx, cval);
    } else {
        let results = fx.bcx.inst_results(call);
        if !dest.layout.is_zst() {
            match dest.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    let val = results[0];
                    dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
                }
                BackendRepr::ScalarPair(_, _) => {
                    dest.write_cvalue(
                        fx,
                        CValue::by_val_pair(results[0], results[1], dest.layout.clone()),
                    );
                }
                _ => {}
            }
        }
    }

    // Jump to continuation or trap for diverging
    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    } else {
        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
    }
}

fn codegen_intrinsic_call(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) -> bool {
    let body = fx.ra_body();
    if !FunctionSignature::is_intrinsic(fx.db(), callee_func_id) {
        return false;
    }

    let sig = fx.db().function_signature(callee_func_id);
    let name = sig.name.as_str();

    // Pre-compute the first generic type argument and its layout (e.g. size_of::<T>)
    // Done eagerly to avoid borrow conflicts with fx inside match arms.
    let (generic_ty, generic_ty_layout) = if generic_args.len() > 0 {
        let ty = generic_args.type_at(0);
        let layout = fx.db().layout_of_ty(ty.store(), fx.env().clone()).ok();
        (Some(ty.store()), layout)
    } else {
        (None, None)
    };

    let result = match name {
        "offset" | "arith_offset" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let ptr_cval = codegen_operand(fx, &args[0].kind);
            let offset_cval = codegen_operand(fx, &args[1].kind);
            let ptr = ptr_cval.load_scalar(fx);
            let offset = offset_cval.load_scalar(fx);

            let ptr_ty = operand_ty(fx.db(), body, &args[0].kind);
            let pointee_ty = ptr_ty
                .as_ref()
                .builtin_deref(true)
                .expect("offset intrinsic first argument must be a pointer");
            let pointee_layout = fx.db()
                .layout_of_ty(pointee_ty.store(), fx.env().clone())
                .expect("layout error for offset intrinsic pointee");

            let ptr_clif_ty = fx.bcx.func.dfg.value_type(ptr);
            let offset_clif_ty = fx.bcx.func.dfg.value_type(offset);
            let offset_signed = ty_is_signed_int(operand_ty(fx.db(), body, &args[1].kind));
            let offset = if offset_clif_ty == ptr_clif_ty {
                offset
            } else if ptr_clif_ty.wider_or_equal(offset_clif_ty) {
                if offset_signed {
                    fx.bcx.ins().sextend(ptr_clif_ty, offset)
                } else {
                    fx.bcx.ins().uextend(ptr_clif_ty, offset)
                }
            } else {
                fx.bcx.ins().ireduce(ptr_clif_ty, offset)
            };

            let byte_offset = fx.bcx.ins().imul_imm(offset, pointee_layout.size.bytes() as i64);
            Some(fx.bcx.ins().iadd(ptr, byte_offset))
        }
        "ptr_offset_from" | "ptr_offset_from_unsigned" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let ptr_cval = codegen_operand(fx, &args[0].kind);
            let base_cval = codegen_operand(fx, &args[1].kind);
            let ptr = ptr_cval.load_scalar(fx);
            let base = base_cval.load_scalar(fx);

            let ptr_ty = operand_ty(fx.db(), body, &args[0].kind);
            let pointee_ty = ptr_ty
                .as_ref()
                .builtin_deref(true)
                .expect("ptr_offset_from first argument must be a pointer");
            let pointee_layout = fx.db()
                .layout_of_ty(pointee_ty.store(), fx.env().clone())
                .expect("layout error for ptr_offset_from intrinsic pointee");
            let pointee_size = pointee_layout.size.bytes();
            assert!(pointee_size != 0, "ptr_offset_from on ZST pointee is unsupported");

            let diff_bytes = fx.bcx.ins().isub(ptr, base);
            if name == "ptr_offset_from_unsigned" {
                Some(fx.bcx.ins().udiv_imm(diff_bytes, pointee_size as i64))
            } else {
                Some(fx.bcx.ins().sdiv_imm(diff_bytes, pointee_size as i64))
            }
        }

        // --- size / alignment queries ---
        "size_of" => {
            let layout = generic_ty_layout.clone().expect("size_of: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64))
        }
        "min_align_of" | "pref_align_of" => {
            let layout = generic_ty_layout.clone().expect("align_of: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64))
        }
        "size_of_val" => {
            // For sized types, same as size_of
            let layout = generic_ty_layout.clone().expect("size_of_val: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64))
        }
        "min_align_of_val" => {
            let layout = generic_ty_layout.clone().expect("min_align_of_val: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64))
        }
        "needs_drop" => {
            let generic_ty = generic_ty.as_ref().expect("needs_drop requires a generic arg");
            let interner = DbInterner::new_with(fx.db(), fx.local_crate());
            let result = hir_ty::drop::has_drop_glue_mono(interner, generic_ty.as_ref());
            Some(fx.bcx.ins().iconst(types::I8, i64::from(result)))
        }
        "type_id" | "type_name" => {
            // These are complex; fall through to let them be unresolved
            // (they need const eval or string allocation)
            return false;
        }

        // --- memory operations ---
        "copy_nonoverlapping" => {
            assert_eq!(args.len(), 3, "copy_nonoverlapping expects 3 args");
            let src = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let dst = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            let layout = generic_ty_layout.clone().expect("copy_nonoverlapping: layout error");
            let elem_size = layout.size.bytes();
            let byte_amount = if elem_size != 1 {
                fx.bcx.ins().imul_imm(count, elem_size as i64)
            } else {
                count
            };
            let tc = fx.module.target_config();
            fx.bcx.call_memcpy(tc, dst, src, byte_amount);
            None
        }
        "copy" => {
            assert_eq!(args.len(), 3, "copy expects 3 args");
            let src = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let dst = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            let layout = generic_ty_layout.clone().expect("copy: layout error");
            let elem_size = layout.size.bytes();
            let byte_amount = if elem_size != 1 {
                fx.bcx.ins().imul_imm(count, elem_size as i64)
            } else {
                count
            };
            let tc = fx.module.target_config();
            fx.bcx.call_memmove(tc, dst, src, byte_amount);
            None
        }
        "write_bytes" => {
            assert_eq!(args.len(), 3, "write_bytes expects 3 args");
            let dst = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let val = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            let layout = generic_ty_layout.clone().expect("write_bytes: layout error");
            let elem_size = layout.size.bytes();
            let byte_amount = if elem_size != 1 {
                fx.bcx.ins().imul_imm(count, elem_size as i64)
            } else {
                count
            };
            let tc = fx.module.target_config();
            fx.bcx.call_memset(tc, dst, val, byte_amount);
            None
        }
        "volatile_load" | "unaligned_volatile_load" => {
            // Cranelift treats loads as volatile by default
            assert_eq!(args.len(), 1);
            let ptr = codegen_operand(fx, &args[0].kind);
            let inner_layout = generic_ty_layout.clone().expect("volatile_load: layout error");
            let val = CValue::by_ref(pointer::Pointer::new(ptr.load_scalar(fx)), inner_layout);
            let dest = codegen_place(fx, destination);
            dest.write_cvalue(fx, val);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "volatile_store" | "unaligned_volatile_store" | "nontemporal_store" => {
            // Cranelift treats stores as volatile by default
            assert_eq!(args.len(), 2);
            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let val = codegen_operand(fx, &args[1].kind);
            let dest = CPlace::for_ptr(pointer::Pointer::new(ptr), val.layout.clone());
            dest.write_cvalue(fx, val);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }

        // --- no-ops and hints ---
        "assume" | "assert_inhabited" | "assert_zero_valid" | "assert_mem_uninitialized_valid" => {
            None // no-op
        }
        "likely" | "unlikely" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            Some(val) // pass through
        }
        "black_box" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind);
            let dest = codegen_place(fx, destination);
            dest.write_cvalue(fx, val);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }

        // --- bit manipulation ---
        "ctlz" | "ctlz_nonzero" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let res = fx.bcx.ins().clz(val);
            // Result type is u32
            Some(codegen_intcast(fx, res, types::I32, false))
        }
        "cttz" | "cttz_nonzero" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let res = fx.bcx.ins().ctz(val);
            Some(codegen_intcast(fx, res, types::I32, false))
        }
        "ctpop" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let res = fx.bcx.ins().popcnt(val);
            Some(codegen_intcast(fx, res, types::I32, false))
        }
        "bswap" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            if fx.bcx.func.dfg.value_type(val) == types::I8 {
                Some(val)
            } else {
                Some(fx.bcx.ins().bswap(val))
            }
        }
        "bitreverse" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            Some(fx.bcx.ins().bitrev(val))
        }

        // --- rotate ---
        "rotate_left" => {
            assert_eq!(args.len(), 2);
            let x = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let y = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().rotl(x, y))
        }
        "rotate_right" => {
            assert_eq!(args.len(), 2);
            let x = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let y = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().rotr(x, y))
        }

        // --- exact_div ---
        "exact_div" => {
            assert_eq!(args.len(), 2);
            let x = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let y = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            if ty_is_signed_int(arg_ty) {
                Some(fx.bcx.ins().sdiv(x, y))
            } else {
                Some(fx.bcx.ins().udiv(x, y))
            }
        }

        // --- wrapping arithmetic ---
        "wrapping_add" | "unchecked_add" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().iadd(a, b))
        }
        "wrapping_sub" | "unchecked_sub" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().isub(a, b))
        }
        "wrapping_mul" | "unchecked_mul" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().imul(a, b))
        }
        "unchecked_shl" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().ishl(a, b))
        }
        "unchecked_shr" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            if ty_is_signed_int(arg_ty) {
                Some(fx.bcx.ins().sshr(a, b))
            } else {
                Some(fx.bcx.ins().ushr(a, b))
            }
        }
        "unchecked_div" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            if ty_is_signed_int(arg_ty) {
                Some(fx.bcx.ins().sdiv(a, b))
            } else {
                Some(fx.bcx.ins().udiv(a, b))
            }
        }
        "unchecked_rem" => {
            assert_eq!(args.len(), 2);
            let a = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let b = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            if ty_is_signed_int(arg_ty) {
                Some(fx.bcx.ins().srem(a, b))
            } else {
                Some(fx.bcx.ins().urem(a, b))
            }
        }

        // --- abort ---
        "abort" => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(2).unwrap());
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }

        // --- pointer masking ---
        "ptr_mask" => {
            assert_eq!(args.len(), 2);
            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let mask = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            Some(fx.bcx.ins().band(ptr, mask))
        }

        // --- transmute ---
        "transmute" => {
            assert_eq!(args.len(), 1);
            let src = codegen_operand(fx, &args[0].kind);
            let dest = codegen_place(fx, destination);
            // Force src to stack, then read back as dest type
            let ptr = src.force_stack(fx);
            let dest_val = CValue::by_ref(ptr, dest.layout.clone());
            dest.write_cvalue(fx, dest_val);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }

        // --- atomic fence (no-op for single-threaded JIT) ---
        "atomic_fence_seqcst" | "atomic_fence_acquire" | "atomic_fence_release"
        | "atomic_fence_acqrel" | "atomic_singlethreadfence_seqcst"
        | "atomic_singlethreadfence_acquire" | "atomic_singlethreadfence_release"
        | "atomic_singlethreadfence_acqrel" => {
            fx.bcx.ins().fence();
            None
        }

        _ => return false,
    };

    // Write result to destination
    let dest = codegen_place(fx, destination);
    if let Some(val) = result {
        if !dest.layout.is_zst() {
            dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
        }
    }

    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    }
    true
}

// ---------------------------------------------------------------------------
// Top-level: compile a MIR body to an object file
// ---------------------------------------------------------------------------

/// Build a Cranelift ISA for the host machine.
pub fn build_host_isa(is_pic: bool) -> Arc<dyn TargetIsa> {
    let mut flags_builder = settings::builder();
    flags_builder.set("is_pic", if is_pic { "true" } else { "false" }).unwrap();
    flags_builder.set("opt_level", "none").unwrap();
    let flags = settings::Flags::new(flags_builder);

    let isa_builder = cranelift_native::builder().expect("host ISA not supported");
    isa_builder.finish(flags).expect("failed to build ISA")
}

/// Build a Cranelift signature from a MIR body's locals (return type + params).
///
/// `mirdata_layouts` is a pre-computed layout table from `.mirdata` files.
/// Currently unused (r-a MIR locals don't carry layout indices), but threaded
/// through for future mirdata body compilation.
pub fn build_fn_sig(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    _mirdata_layouts: &[LayoutArc],
) -> Result<Signature, String> {
    let mut sig = Signature::new(isa.default_call_conv());

    let pointer_ty = pointer_ty(dl);

    // Return type
    let ret_local = &body.locals[hir_ty::mir::return_slot()];
    let ret_layout = db
        .layout_of_ty(ret_local.ty.clone(), env.clone())
        .map_err(|e| format!("return type layout error: {:?}", e))?;
    match ret_layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
        }
        _ if ret_layout.is_zst() => {}
        _ => {
            // Memory-repr return: pass as sret pointer (first param)
            sig.params.push(AbiParam::special(pointer_ty, ArgumentPurpose::StructReturn));
        }
    }

    // Parameter types
    for &param_local in &body.param_locals {
        let param = &body.locals[param_local];
        let param_layout = db
            .layout_of_ty(param.ty.clone(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        match param_layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
            }
            _ if param_layout.is_zst() => {}
            _ => {
                // Memory-repr param: pass by pointer
                sig.params.push(AbiParam::new(pointer_ty));
            }
        }
    }

    Ok(sig)
}

/// Build a Cranelift signature from a function's type information (via `callable_item_signature`)
/// instead of from a MIR body. Needed for extern functions where we don't have MIR.
fn build_fn_sig_from_ty(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    func_id: hir_def::FunctionId,
) -> Result<Signature, String> {
    let mut sig = Signature::new(isa.default_call_conv());

    let pointer_ty = pointer_ty(dl);

    let fn_sig = db
        .callable_item_signature(func_id.into())
        .skip_binder()
        .skip_binder();

    // Return type — skip if `!` (never) or ZST
    let output = *fn_sig.inputs_and_output.as_slice().split_last().unwrap().0;
    if !output.is_never() {
        let ret_layout = db
            .layout_of_ty(output.store(), env.clone())
            .map_err(|e| format!("return type layout error: {:?}", e))?;
        match ret_layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
                sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
            }
            _ if ret_layout.is_zst() => {}
            _ => {
                // Memory-repr return: pass as sret pointer (first param)
                sig.params.push(AbiParam::special(pointer_ty, ArgumentPurpose::StructReturn));
            }
        }
    }

    // Parameter types
    for &param_ty in fn_sig.inputs_and_output.inputs() {
        let param_layout = db
            .layout_of_ty(param_ty.store(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        match param_layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
            }
            BackendRepr::ScalarPair(a, b) => {
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
                sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
            }
            _ if param_layout.is_zst() => {}
            _ => {
                // Memory-repr param: pass by pointer
                sig.params.push(AbiParam::new(pointer_ty));
            }
        }
    }

    Ok(sig)
}

/// Scan a MIR body and return the set of locals whose address is taken
/// (via `Rvalue::Ref` or `Rvalue::AddressOf`). These locals must be
/// stack-allocated so that writes through pointers to them are observable.
fn address_taken_locals(body: &MirBody) -> std::collections::HashSet<LocalId> {
    let mut result = std::collections::HashSet::new();
    for (_, bb) in body.basic_blocks.iter() {
        for stmt in &bb.statements {
            match &stmt.kind {
                StatementKind::Assign(_, Rvalue::Ref(_, place))
                | StatementKind::Assign(_, Rvalue::AddressOf(_, place)) => {
                    result.insert(place.local);
                }
                _ => {}
            }
        }
    }
    result
}

/// Compile a single MIR body to a named function in a Module (ObjectModule or JITModule).
///
/// `mirdata_layouts` is a pre-computed layout table from `.mirdata` files.
/// Pass `&[]` when compiling r-a MIR bodies (all layouts come from `db.layout_of_ty()`).
pub fn compile_fn(
    module: &mut impl Module,
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    fn_name: &str,
    linkage: Linkage,
    local_crate: base_db::Crate,
    ext_crate_disambiguators: &HashMap<String, u64>,
    mirdata_layouts: &[LayoutArc],
) -> Result<FuncId, String> {
    let pointer_type = pointer_ty(dl);
    let sig = build_fn_sig(isa, db, dl, env, body, mirdata_layouts)?;

    // Declare function in module
    let func_id = module
        .declare_function(fn_name, linkage, &sig)
        .map_err(|e| format!("declare_function: {e}"))?;

    // Build the function body
    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
        sig.clone(),
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create Cranelift blocks for MIR basic blocks
        let mut block_map = Vec::with_capacity(body.basic_blocks.len());
        for (_bb_id, _bb) in body.basic_blocks.iter() {
            let block = bcx.create_block();
            block_map.push(block);
        }

        // Set up entry block
        let entry_block = block_map[body.start_block.into_raw().into_u32() as usize];
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);

        // Compute which locals have their address taken (via Ref/AddressOf).
        // These must be stack-allocated even if they're Scalar, because code
        // may write through the pointer.
        let addr_taken = address_taken_locals(body);

        // Build local_map: create CPlace for each local using direct bcx access
        // (We can't use FunctionCx yet because we're still building it)
        let mut local_map = Vec::with_capacity(body.locals.len());
        for (local_id, local) in body.locals.iter() {
            let local_layout = db
                .layout_of_ty(local.ty.clone(), env.clone())
                .map_err(|e| format!("local layout error: {:?}", e))?;

            // Force stack allocation for locals whose address is taken,
            // since writing through the pointer must update the actual local.
            let force_stack = addr_taken.contains(&local_id);

            let place = match local_layout.backend_repr {
                BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _) if force_stack => {
                    // Address-taken scalar: allocate as stack slot
                    let size = u32::try_from(local_layout.size.bytes())
                        .expect("stack slot too large");
                    let align_shift = {
                        let a = local_layout.align.abi.bytes();
                        assert!(a.is_power_of_two());
                        a.trailing_zeros() as u8
                    };
                    let slot = bcx.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            size,
                            align_shift,
                        ),
                    );
                    CPlace::for_ptr(pointer::Pointer::stack_slot(slot), local_layout)
                }
                BackendRepr::Scalar(scalar) => {
                    let clif_ty = scalar_to_clif_type(dl, &scalar);
                    let var = bcx.declare_var(clif_ty);
                    CPlace::new_var_raw(var, local_layout)
                }
                BackendRepr::ScalarPair(a, b) => {
                    let a_clif = scalar_to_clif_type(dl, &a);
                    let b_clif = scalar_to_clif_type(dl, &b);
                    let var1 = bcx.declare_var(a_clif);
                    let var2 = bcx.declare_var(b_clif);
                    CPlace::new_var_pair_raw(var1, var2, local_layout)
                }
                _ if local_layout.is_zst() => {
                    CPlace::for_ptr(
                        pointer::Pointer::dangling(local_layout.align.abi),
                        local_layout,
                    )
                }
                _ => {
                    // Non-scalar, non-ZST: allocate stack slot
                    let size = u32::try_from(local_layout.size.bytes())
                        .expect("stack slot too large");
                    let align_shift = {
                        let a = local_layout.align.abi.bytes();
                        assert!(a.is_power_of_two());
                        a.trailing_zeros() as u8
                    };
                    let slot = bcx.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            size,
                            align_shift,
                        ),
                    );
                    CPlace::for_ptr(pointer::Pointer::stack_slot(slot), local_layout)
                }
            };
            local_map.push(place);
        }

        // Detect sret (indirect return): Memory-repr return type
        let ret_local = &body.locals[hir_ty::mir::return_slot()];
        let ret_layout = db
            .layout_of_ty(ret_local.ty.clone(), env.clone())
            .map_err(|e| format!("return type layout error: {:?}", e))?;
        let is_sret = !ret_layout.is_zst()
            && !matches!(
                ret_layout.backend_repr,
                BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
            );

        // Wire function parameters to their locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;

        // If sret, block param 0 is the sret pointer — override return slot
        if is_sret {
            let sret_ptr = block_params[param_idx];
            param_idx += 1;
            let ret_idx = hir_ty::mir::return_slot().into_raw().into_u32() as usize;
            local_map[ret_idx] =
                CPlace::for_ptr(pointer::Pointer::new(sret_ptr), ret_layout);
        }

        for &param_local in &body.param_locals {
            let param_idx_local = param_local.into_raw().into_u32() as usize;
            let place = &local_map[param_idx_local];
            if place.layout.is_zst() {
                continue;
            }
            match place.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    if place.is_register() {
                        place.def_var(0, block_params[param_idx], &mut bcx);
                    } else {
                        // Address-taken scalar: store param into stack slot
                        let mut flags = MemFlags::new();
                        flags.set_notrap();
                        place.to_ptr().store(&mut bcx, block_params[param_idx], flags);
                    }
                    param_idx += 1;
                }
                BackendRepr::ScalarPair(_, _) => {
                    if place.is_register() {
                        place.def_var(0, block_params[param_idx], &mut bcx);
                        place.def_var(1, block_params[param_idx + 1], &mut bcx);
                    } else {
                        // Address-taken scalar pair: store both parts into stack slot
                        let mut flags = MemFlags::new();
                        flags.set_notrap();
                        let ptr = place.to_ptr();
                        ptr.store(&mut bcx, block_params[param_idx], flags);
                        let BackendRepr::ScalarPair(ref a, ref b) = place.layout.backend_repr
                        else { unreachable!() };
                        let b_off = value_and_place::scalar_pair_b_offset(dl, *a, *b);
                        ptr.offset_i64(&mut bcx, pointer_type, b_off)
                            .store(&mut bcx, block_params[param_idx + 1], flags);
                    }
                    param_idx += 2;
                }
                _ => {
                    // Memory-repr param: block param is a pointer to the data
                    let ptr_val = block_params[param_idx];
                    param_idx += 1;
                    let layout = place.layout.clone();
                    local_map[param_idx_local] =
                        CPlace::for_ptr(
                            pointer::Pointer::new(ptr_val),
                            layout,
                        );
                }
            }
        }

        let mut fx = FunctionCx {
            bcx,
            module,
            isa,
            pointer_type,
            dl,
            mir: MirSource::Ra {
                db,
                env: env.clone(),
                body,
                local_crate,
                ext_crate_disambiguators,
            },
            block_map,
            local_map,
        };

        // Codegen each basic block
        for (bb_id, bb) in body.basic_blocks.iter() {
            let clif_block = fx.clif_block(bb_id);
            if bb_id != body.start_block {
                fx.bcx.switch_to_block(clif_block);
            }

            for stmt in &bb.statements {
                codegen_statement(&mut fx, &stmt.kind);
            }

            if let Some(term) = &bb.terminator {
                codegen_terminator(&mut fx, &term.kind);
            }
        }

        fx.bcx.seal_all_blocks();
        fx.bcx.finalize();
    }

    // Compile and define the function
    let mut ctx = Context::for_function(func);
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define_function: {e}"))?;

    Ok(func_id)
}

/// Compile a MIR body and return the object file bytes.
pub fn compile_to_object(
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    fn_name: &str,
    local_crate: base_db::Crate,
) -> Result<Vec<u8>, String> {
    let isa = build_host_isa(true);
    let empty_map = HashMap::new();

    let builder =
        ObjectBuilder::new(isa.clone(), "rac_output", cranelift_module::default_libcall_names())
            .map_err(|e| format!("ObjectBuilder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    compile_fn(
        &mut module, &*isa, db, dl, env, body, fn_name, Linkage::Export,
        local_crate, &empty_map, &[],
    )?;

    let product = module.finish();
    let bytes = product.emit().map_err(|e| format!("emit: {e}"))?;
    Ok(bytes)
}

/// Emit a C-ABI `main(argc, argv) -> isize` entry point that calls the user's
/// Rust `main()` and returns 0. Skips `lang_start` for now.
pub fn emit_entry_point(
    module: &mut ObjectModule,
    isa: &dyn TargetIsa,
    user_main_func_id: FuncId,
) -> Result<(), String> {
    let ptr_ty = module.target_config().pointer_type();

    // C main signature: (argc: isize, argv: *const *const u8) -> isize
    let mut cmain_sig = Signature::new(isa.default_call_conv());
    cmain_sig.params.push(AbiParam::new(ptr_ty));
    cmain_sig.params.push(AbiParam::new(ptr_ty));
    cmain_sig.returns.push(AbiParam::new(ptr_ty));

    let cmain_func_id = module
        .declare_function("main", Linkage::Export, &cmain_sig)
        .map_err(|e| format!("declare main: {e}"))?;

    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, cmain_func_id.as_u32()),
        cmain_sig,
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);
        let block = bcx.create_block();
        bcx.switch_to_block(block);
        bcx.append_block_params_for_function_params(block);

        // Call user's main()
        let user_main_ref = module.declare_func_in_func(user_main_func_id, bcx.func);
        bcx.ins().call(user_main_ref, &[]);

        // Return 0
        let zero = bcx.ins().iconst(ptr_ty, 0);
        bcx.ins().return_(&[zero]);

        bcx.seal_all_blocks();
        bcx.finalize();
    }

    let mut ctx = Context::for_function(func);
    module
        .define_function(cmain_func_id, &mut ctx)
        .map_err(|e| format!("define main: {e}"))?;

    Ok(())
}

/// When an unsizing coercion `&T → &dyn Trait` is found, discover the impl
/// methods that will be placed in the vtable and add them to the work queue.
fn collect_vtable_methods(
    db: &dyn HirDatabase,
    _env: &StoredParamEnvAndCrate,
    body: &MirBody,
    operand: &Operand,
    target_ty: &StoredTy,
    local_crate: base_db::Crate,
    empty_args: &StoredGenericArgs,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
) {
    // Extract the trait from the target type
    let Some(pointee_ty) = target_ty.as_ref().builtin_deref(true) else { return };
    let Some(trait_id) = pointee_ty.dyn_trait() else { return };

    // Extract the concrete source type
    let from_ty = operand_ty(db, body, &operand.kind);
    let Some(source_pointee) = from_ty.as_ref().builtin_deref(true) else { return };

    // Find the impl for this concrete type
    let interner = DbInterner::new_no_crate(db);
    let trait_impls = TraitImpls::for_crate(db, local_crate);
    use rustc_type_ir::fast_reject::{TreatParams, simplify_type};
    let Some(simplified) = simplify_type(interner, source_pointee, TreatParams::InstantiateWithInfer) else { return };
    let (impl_ids, _) = trait_impls.for_trait_and_self_ty(trait_id, &simplified);
    let Some(&impl_id) = impl_ids.first() else { return };

    // Add all trait method implementations to the queue
    let trait_items = trait_id.trait_items(db);
    let impl_items = impl_id.impl_items(db);
    for (trait_method_name, trait_item) in trait_items.items.iter() {
        let AssocItemId::FunctionId(_) = trait_item else { continue };
        // Find the corresponding impl method
        for (impl_name, impl_item) in impl_items.items.iter() {
            if impl_name == trait_method_name {
                if let AssocItemId::FunctionId(impl_func_id) = impl_item {
                    if impl_func_id.krate(db) == local_crate {
                        queue.push_back((*impl_func_id, empty_args.clone()));
                    }
                }
            }
        }
    }
}

/// Generate a `drop_in_place::<T>` glue function for the given type.
///
/// The generated function takes `*mut T` and:
/// 1. Calls `Drop::drop(&mut *ptr)` if T has a Drop impl
/// 2. Recursively calls `drop_in_place` for each field that needs dropping
///
/// Returns the Cranelift `FuncId` for the generated function.
fn compile_drop_in_place(
    module: &mut impl Module,
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    ty: &StoredTy,
    local_crate: base_db::Crate,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> Result<FuncId, String> {
    let pointer_type = pointer_ty(dl);
    let interner = DbInterner::new_with(db, local_crate);

    // Signature: fn(*mut T) -> void
    let mut sig = Signature::new(isa.default_call_conv());
    sig.params.push(AbiParam::new(pointer_type));

    let fn_name = symbol_mangling::mangle_drop_in_place(
        db, ty.as_ref(), ext_crate_disambiguators,
    );
    let func_id = module
        .declare_function(&fn_name, Linkage::Local, &sig)
        .map_err(|e| format!("declare drop_in_place: {e}"))?;

    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, func_id.as_u32()),
        sig.clone(),
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);
        let entry_block = bcx.create_block();
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);
        let self_ptr = bcx.block_params(entry_block)[0];

        // 1. If T has a direct Drop impl, call Drop::drop(&mut *ptr)
        let lang_items = hir_def::lang_item::lang_items(db, local_crate);
        if let Some(drop_trait) = lang_items.Drop {
            if let Some(drop_func_id) = resolve_drop_impl(db, local_crate, drop_trait, ty) {
                let mut drop_sig = Signature::new(isa.default_call_conv());
                drop_sig.params.push(AbiParam::new(pointer_type));

                let adt_subst = match ty.as_ref().kind() {
                    TyKind::Adt(_, subst) => Some(subst.store()),
                    _ => None,
                };
                let generic_args =
                    adt_subst.unwrap_or_else(|| GenericArgs::empty(interner).store());
                let drop_fn_name = symbol_mangling::mangle_function(
                    db, drop_func_id, generic_args.as_ref(), ext_crate_disambiguators,
                );

                let callee_id = module
                    .declare_function(&drop_fn_name, Linkage::Import, &drop_sig)
                    .expect("declare Drop::drop");
                let callee_ref = module.declare_func_in_func(callee_id, bcx.func);
                bcx.ins().call(callee_ref, &[self_ptr]);
            }
        }

        // 2. Drop fields that need dropping
        // The drop_in_place::<FieldType> functions have the same signature: fn(*mut T)
        let field_drop_sig = sig.clone();

        match ty.as_ref().kind() {
            TyKind::Adt(adt_def, subst) => {
                let adt_id = adt_def.inner().id;
                match adt_id {
                    hir_def::AdtId::StructId(id) => {
                        use hir_def::signatures::StructFlags;
                        if !db.struct_signature(id).flags.intersects(
                            StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA,
                        ) {
                            let layout = db
                                .layout_of_ty(ty.clone(), env.clone())
                                .map_err(|e| format!("layout error: {:?}", e))?;
                            let field_types = db.field_types(id.into());
                            for (field_idx, (_, field_ty_binder)) in
                                field_types.iter().enumerate()
                            {
                                let field_ty =
                                    field_ty_binder.get().instantiate(interner, subst);
                                if hir_ty::drop::has_drop_glue_mono(interner, field_ty) {
                                    let offset =
                                        layout.fields.offset(field_idx).bytes() as i64;
                                    let field_ptr =
                                        bcx.ins().iadd_imm(self_ptr, offset);
                                    let field_ty_stored = field_ty.store();
                                    let name = symbol_mangling::mangle_drop_in_place(
                                        db,
                                        field_ty_stored.as_ref(),
                                        ext_crate_disambiguators,
                                    );
                                    let callee = module
                                        .declare_function(
                                            &name,
                                            Linkage::Import,
                                            &field_drop_sig,
                                        )
                                        .expect("declare field drop_in_place");
                                    let callee_ref =
                                        module.declare_func_in_func(callee, bcx.func);
                                    bcx.ins().call(callee_ref, &[field_ptr]);
                                }
                            }
                        }
                    }
                    hir_def::AdtId::UnionId(_) => {} // union fields not dropped
                    hir_def::AdtId::EnumId(id) => {
                        // For enums, we need to switch on discriminant and drop
                        // the appropriate variant's fields.
                        // TODO: implement enum variant field drops
                        let _ = id;
                    }
                }
            }
            TyKind::Tuple(tys) => {
                let layout = db
                    .layout_of_ty(ty.clone(), env.clone())
                    .map_err(|e| format!("layout error: {:?}", e))?;
                for (idx, elem_ty) in tys.iter().enumerate() {
                    if hir_ty::drop::has_drop_glue_mono(interner, elem_ty) {
                        let offset = layout.fields.offset(idx).bytes() as i64;
                        let field_ptr = bcx.ins().iadd_imm(self_ptr, offset);
                        let elem_stored = elem_ty.store();
                        let name = symbol_mangling::mangle_drop_in_place(
                            db,
                            elem_stored.as_ref(),
                            ext_crate_disambiguators,
                        );
                        let callee = module
                            .declare_function(&name, Linkage::Import, &field_drop_sig)
                            .expect("declare tuple field drop_in_place");
                        let callee_ref = module.declare_func_in_func(callee, bcx.func);
                        bcx.ins().call(callee_ref, &[field_ptr]);
                    }
                }
            }
            TyKind::Closure(closure_id, subst) => {
                let owner = db.lookup_intern_closure(closure_id.0).0;
                let infer = hir_ty::InferenceResult::for_body(db, owner);
                let (captures, _) = infer.closure_info(closure_id.0);
                let layout = db
                    .layout_of_ty(ty.clone(), env.clone())
                    .map_err(|e| format!("layout error: {:?}", e))?;
                for (idx, capture) in captures.iter().enumerate() {
                    let cap_ty = capture.ty(db, subst);
                    if hir_ty::drop::has_drop_glue_mono(interner, cap_ty) {
                        let offset = layout.fields.offset(idx).bytes() as i64;
                        let field_ptr = bcx.ins().iadd_imm(self_ptr, offset);
                        let cap_stored = cap_ty.store();
                        let name = symbol_mangling::mangle_drop_in_place(
                            db,
                            cap_stored.as_ref(),
                            ext_crate_disambiguators,
                        );
                        let callee = module
                            .declare_function(&name, Linkage::Import, &field_drop_sig)
                            .expect("declare capture drop_in_place");
                        let callee_ref = module.declare_func_in_func(callee, bcx.func);
                        bcx.ins().call(callee_ref, &[field_ptr]);
                    }
                }
            }
            _ => {} // primitive types, references, fn ptrs — nothing to drop
        }

        bcx.ins().return_(&[]);
        bcx.seal_all_blocks();
        bcx.finalize();
    }

    let mut ctx = Context::new();
    ctx.func = func;
    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("define drop_in_place: {e}"))?;

    Ok(func_id)
}

/// Collect all non-extern, non-intrinsic, local monomorphized function instances
/// reachable from `root` by walking MIR call terminators. Returns them in BFS
/// order with `root` first. Each entry is a `(FunctionId, StoredGenericArgs)` pair
/// representing a specific monomorphization. Cross-crate functions are skipped.
///
/// Also returns any closures discovered via `AggregateKind::Closure` in statements,
/// and types that need `drop_in_place` glue functions.
fn collect_reachable_fns(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    root: hir_def::FunctionId,
    local_crate: base_db::Crate,
) -> (
    Vec<(hir_def::FunctionId, StoredGenericArgs)>,
    Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    Vec<StoredTy>,
) {
    use std::collections::{HashSet, VecDeque};

    let interner = hir_ty::next_solver::DbInterner::new_no_crate(db);
    let empty_args = GenericArgs::empty(interner).store();
    let empty_args_stored = empty_args.clone();

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    let mut closure_visited = HashSet::new();
    let mut closure_result = Vec::new();

    let mut drop_types = HashSet::new();

    queue.push_back((root, empty_args));

    while let Some((func_id, generic_args)) = queue.pop_front() {
        if !visited.insert((func_id, generic_args.clone())) {
            continue;
        }

        // Skip extern functions (no MIR to compile)
        if matches!(func_id.loc(db).container, ItemContainerId::ExternBlockId(_)) {
            continue;
        }

        // Skip intrinsics (handled inline during codegen)
        if FunctionSignature::is_intrinsic(db, func_id) {
            continue;
        }

        // Skip cross-crate functions (already compiled in rlibs)
        if func_id.krate(db) != local_crate {
            continue;
        }

        // Skip abstract trait method definitions (they have no MIR body;
        // only their impl methods are compiled)
        if matches!(func_id.loc(db).container, ItemContainerId::TraitId(_)) {
            continue;
        }

        result.push((func_id, generic_args.clone()));

        // Get monomorphized MIR and scan for direct callees
        let Ok(body) = db.monomorphized_mir_body(
            func_id.into(),
            generic_args,
            env.clone(),
        ) else {
            continue;
        };

        scan_body_for_callees(
            db, env, &body, interner, local_crate,
            &empty_args_stored, &mut queue,
            &mut closure_visited, &mut closure_result,
            &mut drop_types,
        );
    }

    // Also scan closure bodies for callees and nested closures
    let mut i = 0;
    while i < closure_result.len() {
        let (closure_id, closure_subst) = closure_result[i].clone();
        i += 1;
        let Ok(closure_body) = db.monomorphized_mir_body_for_closure(
            closure_id, closure_subst, env.clone(),
        ) else {
            continue;
        };
        scan_body_for_callees(
            db, env, &closure_body, interner, local_crate,
            &empty_args_stored, &mut queue,
            &mut closure_visited, &mut closure_result,
            &mut drop_types,
        );
        // Process any newly discovered functions from closure bodies
        while let Some((func_id, generic_args)) = queue.pop_front() {
            if !visited.insert((func_id, generic_args.clone())) {
                continue;
            }
            if matches!(func_id.loc(db).container, ItemContainerId::ExternBlockId(_)) {
                continue;
            }
            if FunctionSignature::is_intrinsic(db, func_id) {
                continue;
            }
            if func_id.krate(db) != local_crate {
                continue;
            }
            if matches!(func_id.loc(db).container, ItemContainerId::TraitId(_)) {
                continue;
            }
            result.push((func_id, generic_args.clone()));
            let Ok(body) = db.monomorphized_mir_body(
                func_id.into(), generic_args, env.clone(),
            ) else {
                continue;
            };
            scan_body_for_callees(
                db, env, &body, interner, local_crate,
                &empty_args_stored, &mut queue,
                &mut closure_visited, &mut closure_result,
                &mut drop_types,
            );
        }
    }

    (result, closure_result, drop_types.into_iter().collect())
}

/// Transitively collect all types needing `drop_in_place` glue and their
/// Drop impl methods. Starting from a root type, recurses into fields of
/// structs, tuples, enums, and closures, collecting every type that has
/// drop glue. Also pushes any Drop impl methods found into `fn_queue` so
/// they get compiled.
fn collect_drop_info(
    db: &dyn HirDatabase,
    local_crate: base_db::Crate,
    ty: &StoredTy,
    fn_queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    drop_types: &mut std::collections::HashSet<StoredTy>,
) {
    let interner = DbInterner::new_with(db, local_crate);

    if !hir_ty::drop::has_drop_glue_mono(interner, ty.as_ref()) {
        return;
    }
    if !drop_types.insert(ty.clone()) {
        return;
    }

    // If this type has a direct Drop impl, add its method to the fn queue
    let lang_items = hir_def::lang_item::lang_items(db, local_crate);
    if let Some(drop_trait) = lang_items.Drop {
        if let Some(drop_func_id) = resolve_drop_impl(db, local_crate, drop_trait, ty) {
            let drop_args = match ty.as_ref().kind() {
                TyKind::Adt(_, subst) => subst.store(),
                _ => GenericArgs::empty(interner).store(),
            };
            fn_queue.push_back((drop_func_id, drop_args));
        }
    }

    // Recurse into fields
    match ty.as_ref().kind() {
        TyKind::Adt(adt_def, subst) => {
            let adt_id = adt_def.inner().id;
            match adt_id {
                hir_def::AdtId::StructId(id) => {
                    use hir_def::signatures::StructFlags;
                    if db.struct_signature(id).flags.intersects(
                        StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA,
                    ) {
                        return;
                    }
                    let field_types = db.field_types(id.into());
                    for (_, field_ty) in field_types.iter() {
                        let ft = field_ty.get().instantiate(interner, subst).store();
                        collect_drop_info(db, local_crate, &ft, fn_queue, drop_types);
                    }
                }
                hir_def::AdtId::UnionId(_) => {} // union fields not dropped
                hir_def::AdtId::EnumId(id) => {
                    for &(variant, _, _) in &id.enum_variants(db).variants {
                        let field_types = db.field_types(variant.into());
                        for (_, field_ty) in field_types.iter() {
                            let ft = field_ty.get().instantiate(interner, subst).store();
                            collect_drop_info(db, local_crate, &ft, fn_queue, drop_types);
                        }
                    }
                }
            }
        }
        TyKind::Tuple(tys) => {
            for elem_ty in tys.iter() {
                let ft = elem_ty.store();
                collect_drop_info(db, local_crate, &ft, fn_queue, drop_types);
            }
        }
        TyKind::Closure(closure_id, subst) => {
            let owner = db.lookup_intern_closure(closure_id.0).0;
            let infer = hir_ty::InferenceResult::for_body(db, owner);
            let (captures, _) = infer.closure_info(closure_id.0);
            for capture in captures.iter() {
                let cap_ty = capture.ty(db, subst).store();
                collect_drop_info(db, local_crate, &cap_ty, fn_queue, drop_types);
            }
        }
        _ => {}
    }
}

/// Scan a MIR body for callees (functions and closures) and push them onto
/// the respective work queues.
fn scan_body_for_callees(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    interner: DbInterner,
    local_crate: base_db::Crate,
    empty_args_stored: &StoredGenericArgs,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    drop_types: &mut std::collections::HashSet<StoredTy>,
) {
    use hir_ty::mir::AggregateKind;

    for (_, bb) in body.basic_blocks.iter() {
        // Scan statements for coercions and closure constructions
        for stmt in &bb.statements {
            match &stmt.kind {
                StatementKind::Assign(_, Rvalue::Cast(cast_kind, operand, target_ty)) => {
                    match cast_kind {
                        // Unsizing coercions → discover vtable impl methods
                        CastKind::PointerCoercion(PointerCast::Unsize) => {
                            collect_vtable_methods(
                                db, env, body, operand, target_ty, local_crate,
                                empty_args_stored, queue,
                            );
                        }
                        // ReifyFnPointer: fn item → fn pointer. The target fn needs compilation.
                        CastKind::PointerCoercion(PointerCast::ReifyFnPointer) => {
                            let from_ty = operand_ty(db, body, &operand.kind);
                            if let TyKind::FnDef(def, callee_args) = from_ty.as_ref().kind() {
                                if let CallableDefId::FunctionId(callee_id) = def.0 {
                                    queue.push_back((callee_id, callee_args.store()));
                                }
                            }
                        }
                        _ => {}
                    }
                }
                // Closure constructions: discover closure bodies
                StatementKind::Assign(_, Rvalue::Aggregate(AggregateKind::Closure(ty), _)) => {
                    if let TyKind::Closure(closure_id, closure_subst) = ty.as_ref().kind() {
                        let key = (closure_id.0, closure_subst.store());
                        if closure_visited.insert(key.clone()) {
                            closure_result.push(key);
                        }
                    }
                }
                _ => {}
            }
        }

        let Some(term) = &bb.terminator else { continue };
        match &term.kind {
            TerminatorKind::Call { func, .. } => {
                let OperandKind::Constant { ty, .. } = &func.kind else { continue };
                let TyKind::FnDef(def, callee_args) = ty.as_ref().kind() else { continue };
                if let CallableDefId::FunctionId(callee_id) = def.0 {
                    // Skip virtual calls — trait methods on dyn types are dispatched
                    // through vtables, not compiled as standalone functions.
                    if let ItemContainerId::TraitId(_) = callee_id.loc(db).container {
                        if hir_ty::method_resolution::is_dyn_method(
                            interner,
                            env.param_env(),
                            callee_id,
                            callee_args,
                        ).is_some() {
                            continue;
                        }
                    }
                    queue.push_back((callee_id, callee_args.store()));
                }
            }
            TerminatorKind::Drop { place, .. } => {
                // Transitively discover drop types and their Drop impl methods
                let ty = place_ty(db, body, place);
                collect_drop_info(db, local_crate, &ty, queue, drop_types);
            }
            _ => {}
        }
    }
}

/// Compile a crate to an executable: discover reachable functions from main,
/// compile them all, emit entry point, link.
pub fn compile_executable(
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    main_func_id: hir_def::FunctionId,
    output_path: &Path,
    ext_crate_disambiguators: &HashMap<String, u64>,
) -> Result<(), String> {
    let isa = build_host_isa(true);
    let local_crate = main_func_id.krate(db);

    let builder =
        ObjectBuilder::new(isa.clone(), "rac_output", cranelift_module::default_libcall_names())
            .map_err(|e| format!("ObjectBuilder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    // Discover and compile all reachable local monomorphized function instances
    let (reachable_fns, reachable_closures, drop_types) =
        collect_reachable_fns(db, env, main_func_id, local_crate);
    let mut user_main_id = None;

    for (func_id, generic_args) in &reachable_fns {
        let body = db
            .monomorphized_mir_body((*func_id).into(), generic_args.clone(), env.clone())
            .map_err(|e| format!("MIR error for reachable fn: {:?}", e))?;
        let fn_name = symbol_mangling::mangle_function(
            db, *func_id, generic_args.as_ref(), ext_crate_disambiguators,
        );
        let func_clif_id = compile_fn(
            &mut module, &*isa, db, dl, env, &body, &fn_name, Linkage::Export,
            local_crate, ext_crate_disambiguators, &[],
        )?;

        if *func_id == main_func_id {
            user_main_id = Some(func_clif_id);
        }
    }

    // Compile reachable closure bodies
    for (closure_id, closure_subst) in &reachable_closures {
        let body = db
            .monomorphized_mir_body_for_closure(*closure_id, closure_subst.clone(), env.clone())
            .map_err(|e| format!("MIR error for closure: {:?}", e))?;
        let closure_name = symbol_mangling::mangle_closure(db, *closure_id, ext_crate_disambiguators);
        compile_fn(
            &mut module, &*isa, db, dl, env, &body, &closure_name, Linkage::Export,
            local_crate, ext_crate_disambiguators, &[],
        )?;
    }

    // Compile drop_in_place glue functions
    for ty in &drop_types {
        compile_drop_in_place(
            &mut module, &*isa, db, dl, env, ty,
            local_crate, ext_crate_disambiguators,
        )?;
    }

    let user_main_id = user_main_id.expect("main not in reachable functions");
    emit_entry_point(&mut module, &*isa, user_main_id)?;

    // Emit object file to a temp path
    let product = module.finish();
    let obj_bytes = product.emit().map_err(|e| format!("emit: {e}"))?;

    let obj_path = output_path.with_extension("o");
    std::fs::write(&obj_path, &obj_bytes)
        .map_err(|e| format!("write {}: {e}", obj_path.display()))?;

    // Link
    let result = link::link_executable(&obj_path, output_path);

    // Clean up .o file
    let _ = std::fs::remove_file(&obj_path);

    result
}

#[cfg(test)]
mod tests;

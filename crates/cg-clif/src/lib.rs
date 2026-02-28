//! MIR → Cranelift IR codegen for rust-analyzer.
//!
//! Translates r-a's MIR representation to Cranelift IR and emits object files
//! via cranelift-object. Based on patterns from cg_clif (rustc's Cranelift backend).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use cranelift_codegen::Context;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    AbiParam, ArgumentPurpose, AtomicRmwOp, Block, BlockArg, InstBuilder, MemFlags, Signature,
    Type, Value, types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_module::{DataDescription, FuncId, Linkage, Module, ModuleError};
use cranelift_object::{ObjectBuilder, ObjectModule};
use either::Either;
use hir_def::attrs::AttrFlags;
use hir_def::builtin_derive::BuiltinDeriveImplMethod;
use hir_def::signatures::{FunctionSignature, StaticFlags};
use hir_def::{
    AssocItemId, BuiltinDeriveImplId, CallableDefId, GeneralConstId, HasModule, ItemContainerId,
    Lookup, StaticId, TraitId, VariantId,
};
use hir_ty::PointerCast;
use hir_ty::consteval::{try_const_usize, usize_const};
use hir_ty::db::HirDatabase;
use hir_ty::method_resolution::TraitImpls;
use hir_ty::mir::{
    BasicBlockId, BinOp, CastKind, LocalId, MirBody, Operand, OperandKind, Place, ProjectionElem,
    Rvalue, StatementKind, TerminatorKind, UnOp,
};
use hir_ty::next_solver::{
    Const, ConstKind, DbInterner, GenericArgs, IntoKind, StoredGenericArgs, StoredTy, TyKind,
    ValueConst,
};
use hir_ty::traits::StoredParamEnvAndCrate;
use intern::sym;
use rac_abi::VariantIdx;
use rustc_abi::{BackendRepr, Primitive, Scalar, Size, TargetDataLayout};
use rustc_type_ir::elaborate::supertrait_def_ids;
use rustc_type_ir::inherent::{GenericArgs as _, Region as _, Ty as _};
use triomphe::Arc as TArc;

mod abi;
pub mod link;
mod pointer;
pub mod symbol_mangling;
mod value_and_place;

use hir_ty::layout::Layout;
use rac_abi::callconv::PassMode;
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

fn iconst_from_bits(bcx: &mut FunctionBuilder<'_>, ty: Type, bits: u128) -> Value {
    if ty == types::I128 {
        let lsb = bcx.ins().iconst(types::I64, bits as u64 as i64);
        let msb = bcx.ins().iconst(types::I64, (bits >> 64) as u64 as i64);
        bcx.ins().iconcat(lsb, msb)
    } else {
        bcx.ins().iconst(ty, bits as u64 as i64)
    }
}

fn int_min_max_values(bcx: &mut FunctionBuilder<'_>, ty: Type, signed: bool) -> (Value, Value) {
    assert!(ty.is_int(), "expected integer type, got {ty:?}");
    let bits = ty.bits();
    if signed {
        let sign_bit = 1u128 << (bits - 1);
        (
            iconst_from_bits(bcx, ty, sign_bit),
            iconst_from_bits(bcx, ty, sign_bit - 1),
        )
    } else {
        let max = if bits == 128 { u128::MAX } else { (1u128 << bits) - 1 };
        (iconst_from_bits(bcx, ty, 0), iconst_from_bits(bcx, ty, max))
    }
}

fn stable_hash64_with_seed(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Emit a return instruction from the given return place.
/// ABI-driven via `fx.fn_abi.ret`.
pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, impl Module>, ret_place: &CPlace) {
    let ret_abi = fx.fn_abi.ret.clone();
    abi::returning::codegen_return(fx, ret_place, &ret_abi);
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
}

// ---------------------------------------------------------------------------
// FunctionCx: per-function codegen state
// ---------------------------------------------------------------------------

pub(crate) struct FunctionCx<'a, M: Module> {
    pub(crate) bcx: FunctionBuilder<'a>,
    pub(crate) module: &'a mut M,
    pub(crate) isa: &'a dyn TargetIsa,
    pub(crate) fn_abi: abi::FnAbi,
    pub(crate) pointer_type: Type,
    pub(crate) dl: &'a TargetDataLayout,
    pub(crate) mir: MirSource<'a>,
    /// MIR basic block → Cranelift block (indexed by raw block id)
    pub(crate) block_map: Vec<Block>,
    /// MIR local → CPlace (indexed by raw local id)
    pub(crate) local_map: Vec<CPlace>,
    /// Drop flags: local raw index → boolean Variable (i8: 0=dead, 1=live).
    /// Only present for locals that have drop glue. Checked in codegen_drop
    /// to skip drops on moved-out locals (r-a's MIR lacks drop elaboration).
    pub(crate) drop_flags: HashMap<u32, Variable>,
}

impl<'a, M: Module> FunctionCx<'a, M> {
    fn db(&self) -> &'a dyn HirDatabase {
        match &self.mir {
            MirSource::Ra { db, .. } => *db,
        }
    }

    fn env(&self) -> &StoredParamEnvAndCrate {
        match &self.mir {
            MirSource::Ra { env, .. } => env,
        }
    }

    fn ra_body(&self) -> &'a MirBody {
        match &self.mir {
            MirSource::Ra { body, .. } => body,
        }
    }

    fn local_crate(&self) -> base_db::Crate {
        match &self.mir {
            MirSource::Ra { local_crate, .. } => *local_crate,
        }
    }

    fn ext_crate_disambiguators(&self) -> &'a HashMap<String, u64> {
        match &self.mir {
            MirSource::Ra { ext_crate_disambiguators, .. } => ext_crate_disambiguators,
        }
    }

    fn clif_block(&self, bb: BasicBlockId) -> Block {
        self.block_map[bb.into_raw().into_u32() as usize]
    }

    fn local_place(&self, local: LocalId) -> &CPlace {
        &self.local_map[local.into_raw().into_u32() as usize]
    }

    /// Set the drop flag for a local to "live" (1). Called on assignment.
    fn set_drop_flag(&mut self, local: LocalId) {
        let idx = local.into_raw().into_u32();
        if let Some(&var) = self.drop_flags.get(&idx) {
            let one = self.bcx.ins().iconst(types::I8, 1);
            self.bcx.def_var(var, one);
        }
    }

    /// Clear the drop flag for a local to "dead" (0). Called on move.
    fn clear_drop_flag(&mut self, local: LocalId) {
        let idx = local.into_raw().into_u32();
        if let Some(&var) = self.drop_flags.get(&idx) {
            let zero = self.bcx.ins().iconst(types::I8, 0);
            self.bcx.def_var(var, zero);
        }
    }
}

// ---------------------------------------------------------------------------
// Constant extraction
// ---------------------------------------------------------------------------

fn resolve_const_value<'db>(db: &'db dyn HirDatabase, mut konst: Const<'db>) -> ValueConst<'db> {
    loop {
        match konst.kind() {
            ConstKind::Value(val) => return val,
            ConstKind::Unevaluated(uv) => {
                let evaluated = match uv.def.0 {
                    GeneralConstId::ConstId(id) => db.const_eval(id, uv.args, None),
                    GeneralConstId::StaticId(id) => db.const_eval_static(id),
                }
                .unwrap_or_else(|e| panic!("failed to evaluate const {:?}: {e:?}", konst));

                if evaluated == konst {
                    panic!("const eval made no progress for unevaluated const: {:?}", konst);
                }

                konst = evaluated;
            }
            _ => panic!("resolve_const_value: unsupported const kind: {:?}", konst),
        }
    }
}

fn const_to_u128<'db>(db: &'db dyn HirDatabase, konst: Const<'db>, size: Size) -> u128 {
    let val = resolve_const_value(db, konst);
    let bytes = &val.value.inner().memory;
    let mut buf = [0u8; 16];
    let len = (size.bytes() as usize).min(16);
    buf[..len].copy_from_slice(&bytes[..len]);
    u128::from_le_bytes(buf)
}

/// Extract a `u64` from a constant. Used for array lengths and repeat counts.
fn const_to_u64<'db>(db: &'db dyn HirDatabase, konst: Const<'db>) -> u64 {
    let val = resolve_const_value(db, konst);
    let bytes = &val.value.inner().memory;
    let mut buf = [0u8; 8];
    let len = bytes.len().min(8);
    buf[..len].copy_from_slice(&bytes[..len]);
    u64::from_le_bytes(buf)
}

// ---------------------------------------------------------------------------
// Type helpers
// ---------------------------------------------------------------------------

/// Extract the stored type from an operand.
fn operand_ty(db: &dyn HirDatabase, body: &MirBody, kind: &OperandKind) -> StoredTy {
    match kind {
        OperandKind::Constant { ty, .. } => ty.clone(),
        OperandKind::Copy(place) | OperandKind::Move(place) => place_ty(db, body, place),
        OperandKind::Static(static_id) => static_operand_ty(db, *static_id),
    }
}

fn peel_ref_layers<'db>(mut ty: hir_ty::next_solver::Ty<'db>) -> hir_ty::next_solver::Ty<'db> {
    while let TyKind::Ref(_, inner, _) = ty.kind() {
        ty = inner;
    }
    ty
}

fn is_fn_trait_method(db: &dyn HirDatabase, trait_id: TraitId, method_name: &str) -> bool {
    let trait_krate = trait_id.module(db).krate(db);
    let lang_items = hir_def::lang_item::lang_items(db, trait_krate);
    let is_fn_trait = Some(trait_id) == lang_items.Fn
        || Some(trait_id) == lang_items.FnMut
        || Some(trait_id) == lang_items.FnOnce;
    is_fn_trait && matches!(method_name, "call" | "call_mut" | "call_once")
}

fn extern_fn_symbol_name(db: &dyn HirDatabase, func_id: hir_def::FunctionId) -> String {
    if let Some(link_name) = hir_def::attrs::AttrFlags::link_name(db, func_id) {
        link_name.as_str().to_owned()
    } else {
        db.function_signature(func_id).name.as_str().to_owned()
    }
}

fn static_pointee_ty(db: &dyn HirDatabase, static_id: StaticId) -> hir_ty::next_solver::Ty<'_> {
    hir_ty::InferenceResult::for_body(db, static_id.into())
        .expr_ty(db.body(static_id.into()).body_expr)
}

fn static_operand_ty(db: &dyn HirDatabase, static_id: StaticId) -> StoredTy {
    let interner = DbInterner::new_no_crate(db);
    hir_ty::next_solver::Ty::new_ref(
        interner,
        hir_ty::next_solver::Region::new_static(interner),
        static_pointee_ty(db, static_id),
        hir_ty::next_solver::Mutability::Not,
    )
    .store()
}

fn place_ty(db: &dyn HirDatabase, body: &MirBody, place: &Place) -> StoredTy {
    let mut ty = body.locals[place.local].ty.clone();
    let local_crate = body.owner.krate(db);
    let projections = place.projection.lookup(&body.projection_store);
    for proj in projections {
        ty = match proj {
            ProjectionElem::Field(field) => field_type(db, &ty, field),
            ProjectionElem::Deref => {
                ty.as_ref().builtin_deref(true).expect("deref on non-pointer").store()
            }
            ProjectionElem::Downcast(_) => ty, // Downcast doesn't change the Rust type
            ProjectionElem::ClosureField(idx) => closure_field_type(db, &ty, *idx),
            ProjectionElem::Index(_) | ProjectionElem::ConstantIndex { .. } => {
                match ty.as_ref().kind() {
                    TyKind::Array(elem, _) | TyKind::Slice(elem) => elem.store(),
                    _ => panic!("Index on non-array/slice type"),
                }
            }
            ProjectionElem::Subslice { from, to } => match ty.as_ref().kind() {
                TyKind::Array(elem, len) => {
                    let new_len =
                        try_const_usize(db, len).and_then(|n| n.checked_sub(u128::from(*from + *to)));
                    let new_len_const = usize_const(db, new_len, local_crate);
                    let interner = DbInterner::new_no_crate(db);
                    hir_ty::next_solver::Ty::new_array_with_const_len(interner, elem, new_len_const)
                        .store()
                }
                TyKind::Slice(_) | TyKind::Str => ty,
                _ => panic!("Subslice projection on non-array/slice/str type"),
            },
            ProjectionElem::OpaqueCast(cast_ty) => cast_ty.clone(),
        };
    }
    ty
}

/// Walk to the structural unsized tail of a type.
///
/// For wrappers like `struct S<T: ?Sized>(u8, T)`, this returns `T`.
fn unsized_tail_ty(db: &dyn HirDatabase, mut ty: StoredTy) -> StoredTy {
    loop {
        match ty.as_ref().kind() {
            TyKind::Adt(adt_id, args) => {
                let hir_def::AdtId::StructId(struct_id) = adt_id.inner().id else {
                    return ty;
                };
                let field_types = db.field_types(struct_id.into());
                let Some((_, field_ty)) = field_types.iter().last() else {
                    return ty;
                };
                let interner = DbInterner::new_no_crate(db);
                let next_ty = field_ty.get().instantiate(interner, args).store();
                if next_ty == ty {
                    return ty;
                }
                ty = next_ty;
            }
            TyKind::Tuple(tys) => {
                let Some(last) = tys.as_slice().last() else {
                    return ty;
                };
                let next_ty = last.store();
                if next_ty == ty {
                    return ty;
                }
                ty = next_ty;
            }
            _ => return ty,
        }
    }
}

/// Walk source/target types together to the structural unsized tail.
///
/// This mirrors rustc's lockstep-tail behavior for unsizing:
/// `S<[T; N]> -> S<[T]>` yields `[T; N] -> [T]`, while `T -> dyn Trait`
/// remains unchanged.
fn lockstep_unsized_tails(
    db: &dyn HirDatabase,
    mut source_ty: StoredTy,
    mut target_ty: StoredTy,
) -> (StoredTy, StoredTy) {
    loop {
        let interner = DbInterner::new_no_crate(db);
        match (source_ty.as_ref().kind(), target_ty.as_ref().kind()) {
            (TyKind::Adt(source_adt, source_args), TyKind::Adt(target_adt, target_args))
                if source_adt.inner().id == target_adt.inner().id =>
            {
                let hir_def::AdtId::StructId(source_struct_id) = source_adt.inner().id else {
                    return (source_ty, target_ty);
                };
                let hir_def::AdtId::StructId(target_struct_id) = target_adt.inner().id else {
                    return (source_ty, target_ty);
                };
                let source_fields = db.field_types(source_struct_id.into());
                let target_fields = db.field_types(target_struct_id.into());
                let (Some((_, source_field_ty)), Some((_, target_field_ty))) =
                    (source_fields.iter().last(), target_fields.iter().last())
                else {
                    return (source_ty, target_ty);
                };
                let next_source = source_field_ty.get().instantiate(interner, source_args).store();
                let next_target = target_field_ty.get().instantiate(interner, target_args).store();
                if next_source == source_ty && next_target == target_ty {
                    return (source_ty, target_ty);
                }
                source_ty = next_source;
                target_ty = next_target;
            }
            (TyKind::Tuple(source_tys), TyKind::Tuple(target_tys))
                if source_tys.len() == target_tys.len() =>
            {
                let (Some(source_last), Some(target_last)) =
                    (source_tys.as_slice().last(), target_tys.as_slice().last())
                else {
                    return (source_ty, target_ty);
                };
                let next_source = source_last.store();
                let next_target = target_last.store();
                if next_source == source_ty && next_target == target_ty {
                    return (source_ty, target_ty);
                }
                source_ty = next_source;
                target_ty = next_target;
            }
            _ => return (source_ty, target_ty),
        }
    }
}

/// Compute runtime size/alignment for a potentially-DST pointee type.
///
/// Mirrors upstream cg_clif's `unsize::size_and_align_of` behavior for
/// `size_of_val`/`align_of_val` intrinsics.
fn size_and_align_of_pointee(
    fx: &mut FunctionCx<'_, impl Module>,
    pointee_ty: StoredTy,
    pointee_layout: &LayoutArc,
    metadata: Option<Value>,
) -> (Value, Value) {
    let static_size = fx.bcx.ins().iconst(fx.pointer_type, pointee_layout.size.bytes() as i64);
    let static_align =
        fx.bcx.ins().iconst(fx.pointer_type, pointee_layout.align.abi.bytes() as i64);

    let tail_ty = unsized_tail_ty(fx.db(), pointee_ty);
    let (tail_size, tail_align) = match tail_ty.as_ref().kind() {
        TyKind::Dynamic(..) => {
            let vtable = metadata.expect("size/align_of_val(dyn) requires metadata");
            let ptr_size = fx.dl.pointer_size().bytes() as i32;
            let size = fx.bcx.ins().load(fx.pointer_type, vtable_memflags(), vtable, ptr_size);
            let align = fx.bcx.ins().load(fx.pointer_type, vtable_memflags(), vtable, ptr_size * 2);
            (Some(size), Some(align))
        }
        TyKind::Slice(elem_ty) => {
            let len = metadata.expect("size/align_of_val(slice) requires metadata");
            let elem_layout = fx
                .db()
                .layout_of_ty(elem_ty.store(), fx.env().clone())
                .expect("size/align_of_val(slice): elem layout error");
            let elem_size = elem_layout.size.bytes() as i64;
            let size = if elem_size == 1 { len } else { fx.bcx.ins().imul_imm(len, elem_size) };
            let align = fx.bcx.ins().iconst(fx.pointer_type, elem_layout.align.abi.bytes() as i64);
            (Some(size), Some(align))
        }
        TyKind::Str => {
            let len = metadata.expect("size/align_of_val(str) requires metadata");
            let align = fx.bcx.ins().iconst(fx.pointer_type, 1);
            (Some(len), Some(align))
        }
        _ => (None, None),
    };

    match (tail_size, tail_align) {
        (Some(tail_size), Some(tail_align)) => {
            let full_align = fx.bcx.ins().umax(static_align, tail_align);
            let full_size = fx.bcx.ins().iadd(static_size, tail_size);
            // Align size up to full alignment: (size + align - 1) & -align.
            let align_minus_one = fx.bcx.ins().iadd_imm(full_align, -1);
            let add = fx.bcx.ins().iadd(full_size, align_minus_one);
            let neg_align = fx.bcx.ins().ineg(full_align);
            let aligned_size = fx.bcx.ins().band(add, neg_align);
            (aligned_size, full_align)
        }
        _ => (static_size, static_align),
    }
}

pub(crate) fn bin_op_to_intcc(op: &BinOp, signed: bool) -> IntCC {
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

pub(crate) fn bin_op_to_floatcc(op: &BinOp) -> FloatCC {
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

fn ty_is_atomic_int_or_ptr(ty: &StoredTy) -> bool {
    matches!(ty.as_ref().kind(), TyKind::Int(_) | TyKind::Uint(_) | TyKind::RawPtr(_, _))
}

fn ty_is_atomic_int(ty: &StoredTy) -> bool {
    matches!(ty.as_ref().kind(), TyKind::Int(_) | TyKind::Uint(_))
}

fn ty_is_atomic_unsigned_int(ty: &StoredTy) -> bool {
    matches!(ty.as_ref().kind(), TyKind::Uint(_))
}

fn atomic_scalar_clif_ty(dl: &TargetDataLayout, layout: &Layout) -> Type {
    let BackendRepr::Scalar(scalar) = layout.backend_repr else {
        panic!("atomic intrinsic expects scalar layout, got {:?}", layout.backend_repr);
    };
    scalar_to_clif_type(dl, &scalar)
}

pub(crate) fn codegen_intcast(
    fx: &mut FunctionCx<'_, impl Module>,
    val: Value,
    to_ty: Type,
    signed: bool,
) -> Value {
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

pub(crate) fn codegen_libcall1(
    fx: &mut FunctionCx<'_, impl Module>,
    name: &str,
    params: &[Type],
    ret: Type,
    args: &[Value],
) -> Value {
    let mut sig = Signature::new(fx.isa.default_call_conv());
    sig.params.extend(params.iter().copied().map(AbiParam::new));
    sig.returns.push(AbiParam::new(ret));

    let func_id = fx.module.declare_function(name, Linkage::Import, &sig).expect("declare libcall");
    let func_ref = fx.module.declare_func_in_func(func_id, fx.bcx.func);
    let call = fx.bcx.ins().call(func_ref, args);
    fx.bcx.inst_results(call)[0]
}

/// Allocate heap storage for a boxed pointee.
///
/// Uses `__rust_alloc(size, align)` for non-ZST pointees and traps on null.
/// For ZST pointees returns an aligned dangling pointer, matching Box semantics.
fn codegen_box_alloc(fx: &mut FunctionCx<'_, impl Module>, pointee_layout: &Layout) -> Value {
    if pointee_layout.is_zst() {
        return fx.bcx.ins().iconst(fx.pointer_type, pointee_layout.align.abi.bytes() as i64);
    }

    let size = fx.bcx.ins().iconst(fx.pointer_type, pointee_layout.size.bytes() as i64);
    let align = fx.bcx.ins().iconst(fx.pointer_type, pointee_layout.align.abi.bytes() as i64);
    let ptr = codegen_libcall1(
        fx,
        "__rust_alloc",
        &[fx.pointer_type, fx.pointer_type],
        fx.pointer_type,
        &[size, align],
    );

    // `Box` must not contain a null pointer; trap hard on allocation failure.
    fx.bcx.ins().trapz(ptr, cranelift_codegen::ir::TrapCode::user(3).unwrap());
    ptr
}

/// Type-system-agnostic scalar cast kinds.
pub(crate) enum ScalarCastKind {
    IntToInt,
    FloatToInt,
    IntToFloat,
    FloatToFloat,
    /// Thin pointer casts (PtrToPtr, FnPtrToPtr, expose/with provenance, MutToConst, etc.)
    PtrLike,
}

/// Shared scalar-to-scalar cast codegen.
///
/// Given a loaded scalar `from_val`, its signedness, the target scalar, and the
/// cast kind, emits the appropriate cranelift conversion instruction.
pub(crate) fn codegen_scalar_cast(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: ScalarCastKind,
    from_val: Value,
    from_signed: bool,
    target_scalar: &Scalar,
    dest_layout: &LayoutArc,
) -> CValue {
    let from_clif_ty = fx.bcx.func.dfg.value_type(from_val);
    let target_clif_ty = scalar_to_clif_type(fx.dl, target_scalar);

    let val = match kind {
        ScalarCastKind::IntToInt => codegen_intcast(fx, from_val, target_clif_ty, from_signed),
        ScalarCastKind::FloatToInt => {
            let target_signed = matches!(target_scalar.primitive(), Primitive::Int(_, true));
            if target_signed {
                fx.bcx.ins().fcvt_to_sint_sat(target_clif_ty, from_val)
            } else {
                fx.bcx.ins().fcvt_to_uint_sat(target_clif_ty, from_val)
            }
        }
        ScalarCastKind::IntToFloat => {
            if from_signed {
                fx.bcx.ins().fcvt_from_sint(target_clif_ty, from_val)
            } else {
                fx.bcx.ins().fcvt_from_uint(target_clif_ty, from_val)
            }
        }
        ScalarCastKind::FloatToFloat => match (from_clif_ty, target_clif_ty) {
            (from, to) if from == to => from_val,
            (from, to) if to.wider_or_equal(from) => fx.bcx.ins().fpromote(to, from_val),
            (to, from) if to.wider_or_equal(from) => fx.bcx.ins().fdemote(from, from_val),
            _ => unreachable!("invalid float cast from {from_clif_ty:?} to {target_clif_ty:?}"),
        },
        ScalarCastKind::PtrLike => codegen_intcast(fx, from_val, target_clif_ty, false),
    };
    CValue::by_val(val, dest_layout.clone())
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
        let CallableDefId::FunctionId(trait_or_callee_func_id) = def.0 else {
            panic!("ReifyFnPointer on non-function: {:?}", def);
        };
        let env = fx.env().clone();
        let mut callee_func_id = trait_or_callee_func_id;
        let mut generic_args = generic_args;
        let mut builtin_derive_shim_id = None;

        if let ItemContainerId::TraitId(_) = trait_or_callee_func_id.loc(fx.db()).container {
            let interner = DbInterner::new_no_crate(fx.db());
            if hir_ty::method_resolution::is_dyn_method(
                interner,
                env.param_env(),
                trait_or_callee_func_id,
                generic_args,
            )
            .is_some()
            {
                // Keep trait item identity for dyn methods; they don't have a
                // concrete impl target at reify time.
            } else {
                let lookup_result =
                    fx.db().lookup_impl_method(env.as_ref(), trait_or_callee_func_id, generic_args);
                match lookup_result {
                    (Either::Left(resolved_id), resolved_args) => {
                        callee_func_id = resolved_id;
                        generic_args = resolved_args;
                    }
                    (Either::Right((derive_impl_id, derive_method)), resolved_args) => {
                        builtin_derive_shim_id = Some(declare_builtin_derive_method_shim(
                            fx,
                            trait_or_callee_func_id,
                            derive_impl_id,
                            derive_method,
                            resolved_args,
                        ));
                    }
                }
            }
        }

        if let Some(shim_id) = builtin_derive_shim_id {
            let callee_ref = fx.module.declare_func_in_func(shim_id, fx.bcx.func);
            let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, callee_ref);
            return CValue::by_val(func_addr, result_layout.clone());
        }

        // Declare the function in the module and get its address
        let is_extern =
            matches!(callee_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
        let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();
        let interner = DbInterner::new_no_crate(fx.db());
        let empty_args = GenericArgs::empty(interner);
        let (callee_sig, callee_name) = if is_extern {
            let sig =
                build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id, empty_args)
                    .expect("extern fn sig");
            let name = extern_fn_symbol_name(fx.db(), callee_func_id);
            (sig, name)
        } else if is_cross_crate {
            let sig = build_fn_sig_from_ty(
                fx.isa,
                fx.db(),
                fx.dl,
                fx.env(),
                callee_func_id,
                generic_args,
            )
            .expect("cross-crate fn sig");
            let name = symbol_mangling::mangle_function(
                fx.db(),
                callee_func_id,
                generic_args,
                fx.ext_crate_disambiguators(),
            );
            (sig, name)
        } else {
            let callee_body = fx
                .db()
                .monomorphized_mir_body(
                    callee_func_id.into(),
                    generic_args.store(),
                    fx.env().clone(),
                )
                .expect("failed to get callee MIR for ReifyFnPointer");
            let sig =
                build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body).expect("callee sig");
            let name = symbol_mangling::mangle_function(
                fx.db(),
                callee_func_id,
                generic_args,
                fx.ext_crate_disambiguators(),
            );
            (sig, name)
        };
        let callee_id = fx
            .module
            .declare_function(&callee_name, Linkage::Import, &callee_sig)
            .expect("declare callee for ReifyFnPointer");
        let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
        let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, callee_ref);
        return CValue::by_val(func_addr, result_layout.clone());
    }

    // Handle ClosureFnPointer: non-capturing closure -> fn pointer.
    // Must be before the generic scalar path because closures are memory-repr.
    if let CastKind::PointerCoercion(PointerCast::ClosureFnPointer(_)) = kind {
        let from_ty = operand_ty(fx.db(), body, &operand.kind);
        let TyKind::Closure(closure_id, closure_subst) = from_ty.as_ref().kind() else {
            panic!("ClosureFnPointer on non-closure type: {:?}", from_ty);
        };

        let closure_layout = fx
            .db()
            .layout_of_ty(from_ty.clone(), fx.env().clone())
            .expect("ClosureFnPointer source layout");
        if !closure_layout.is_zst() {
            panic!(
                "ClosureFnPointer requires a non-capturing closure; got layout {:?}",
                closure_layout.backend_repr
            );
        }

        let closure_body = fx
            .db()
            .monomorphized_mir_body_for_closure(
                closure_id.0,
                closure_subst.store(),
                fx.env().clone(),
            )
            .expect("failed to get closure MIR for ClosureFnPointer");
        let closure_sig =
            build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body).expect("closure sig");
        let closure_name = symbol_mangling::mangle_closure(
            fx.db(),
            closure_id.0,
            closure_subst,
            fx.ext_crate_disambiguators(),
        );
        let closure_func_id = fx
            .module
            .declare_function(&closure_name, Linkage::Import, &closure_sig)
            .expect("declare closure for ClosureFnPointer");
        let closure_ref = fx.module.declare_func_in_func(closure_func_id, fx.bcx.func);
        let func_addr = fx.bcx.ins().func_addr(fx.pointer_type, closure_ref);
        return CValue::by_val(func_addr, result_layout.clone());
    }

    let from_ty = operand_ty(fx.db(), body, &operand.kind);
    let from_cval = codegen_operand(fx, &operand.kind);

    // Handle Transmute: reinterpret the bits with a different layout.
    if let CastKind::Transmute = kind {
        return codegen_transmute(fx, from_cval, result_layout);
    }

    let is_ptr_like_cast = matches!(
        kind,
        CastKind::PtrToPtr
            | CastKind::FnPtrToPtr
            | CastKind::PointerExposeProvenance
            | CastKind::PointerWithExposedProvenance
            | CastKind::PointerCoercion(_)
    );

    // Handle wide-pointer casts explicitly (mirrors upstream behavior):
    // - wide -> wide: preserve data + metadata
    // - wide -> thin: drop metadata
    if is_ptr_like_cast {
        match (&from_cval.layout.backend_repr, &result_layout.backend_repr) {
            (BackendRepr::ScalarPair(_, _), BackendRepr::ScalarPair(_, _)) => {
                let (data, meta) = from_cval.load_scalar_pair(fx);
                return CValue::by_val_pair(data, meta, result_layout.clone());
            }
            (BackendRepr::ScalarPair(_, _), BackendRepr::Scalar(_)) => {
                let (data, _) = from_cval.load_scalar_pair(fx);
                return CValue::by_val(data, result_layout.clone());
            }
            (BackendRepr::Scalar(_), BackendRepr::ScalarPair(_, _)) => {
                panic!("unsupported thin->wide ptr-like cast without unsize coercion")
            }
            _ => {}
        }
    }

    let from_val = from_cval.load_scalar(fx);

    let BackendRepr::Scalar(target_scalar) = result_layout.backend_repr else {
        todo!("cast target must be scalar")
    };

    let from_signed = ty_is_signed_int(from_ty);
    let scalar_kind = match kind {
        CastKind::IntToInt => ScalarCastKind::IntToInt,
        CastKind::FloatToInt => ScalarCastKind::FloatToInt,
        CastKind::IntToFloat => ScalarCastKind::IntToFloat,
        CastKind::FloatToFloat => ScalarCastKind::FloatToFloat,
        CastKind::PtrToPtr
        | CastKind::FnPtrToPtr
        | CastKind::PointerExposeProvenance
        | CastKind::PointerWithExposedProvenance
        | CastKind::PointerCoercion(_) => ScalarCastKind::PtrLike,
        CastKind::Transmute => unreachable!("handled above"),
        CastKind::DynStar => todo!("dyn* cast"),
    };
    codegen_scalar_cast(fx, scalar_kind, from_val, from_signed, &target_scalar, result_layout)
}

/// Handle `CastKind::Transmute`: reinterpret bits with a different layout.
/// Reference: cg_clif/src/value_and_place.rs `write_cvalue_transmute`
pub(crate) fn codegen_transmute(
    fx: &mut FunctionCx<'_, impl Module>,
    from: CValue,
    target_layout: &LayoutArc,
) -> CValue {
    assert_eq!(
        from.layout.size, target_layout.size,
        "transmute between differently-sized types: {:?} vs {:?}",
        from.layout.size, target_layout.size,
    );

    match (&from.layout.backend_repr, &target_layout.backend_repr) {
        (BackendRepr::ScalarPair(_, _), BackendRepr::ScalarPair(b_a, b_b)) => {
            // ScalarPair→ScalarPair: load both halves, bitcast each if types differ.
            let (val_a, val_b) = from.load_scalar_pair(fx);
            let dst_a = scalar_to_clif_type(fx.dl, b_a);
            let dst_b = scalar_to_clif_type(fx.dl, b_b);
            let a = clif_bitcast(fx, val_a, dst_a);
            let b = clif_bitcast(fx, val_b, dst_b);
            CValue::by_val_pair(a, b, target_layout.clone())
        }
        (BackendRepr::Scalar(_), BackendRepr::Scalar(dst_scalar)) => {
            let val = from.load_scalar(fx);
            let dst_ty = scalar_to_clif_type(fx.dl, dst_scalar);
            let val = clif_bitcast(fx, val, dst_ty);
            CValue::by_val(val, target_layout.clone())
        }
        _ => {
            // Different representations: spill to stack, reload with new layout.
            let ptr = from.force_stack(fx);
            CValue::by_ref(ptr, target_layout.clone())
        }
    }
}

/// Bitcast a Cranelift value to a different type of the same size.
/// Returns unchanged if types already match.
pub(crate) fn clif_bitcast(
    fx: &mut FunctionCx<'_, impl Module>,
    val: Value,
    dst_ty: Type,
) -> Value {
    let src_ty = fx.bcx.func.dfg.value_type(val);
    if src_ty == dst_ty { val } else { fx.bcx.ins().bitcast(dst_ty, MemFlags::new(), val) }
}

fn unsize_metadata_for_pointees(
    fx: &mut FunctionCx<'_, impl Module>,
    source_pointee: StoredTy,
    target_pointee: StoredTy,
    from_meta: Option<Value>,
) -> Value {
    // Follow source/target in lockstep so we only peel matching wrappers.
    let (source_tail, target_tail) =
        lockstep_unsized_tails(fx.db(), source_pointee.clone(), target_pointee.clone());

    match target_tail.as_ref().kind() {
        TyKind::Dynamic(..) => match source_tail.as_ref().kind() {
            TyKind::Dynamic(..) => {
                from_meta.expect("dyn unsize from wide source requires metadata")
            }
            _ => {
                let trait_id = target_tail
                    .as_ref()
                    .dyn_trait()
                    .expect("dyn unsize target pointee must have a principal trait");
                get_or_create_vtable(fx, source_tail, trait_id)
            }
        },
        TyKind::Slice(_) | TyKind::Str => match source_tail.as_ref().kind() {
            TyKind::Array(_, len) => {
                let len = try_const_usize(fx.db(), len)
                    .expect("array->slice unsize requires monomorphic array length")
                    as i64;
                fx.bcx.ins().iconst(fx.pointer_type, len)
            }
            TyKind::Slice(_) | TyKind::Str => {
                from_meta.expect("slice/str unsize from wide source requires metadata")
            }
            _ => panic!(
                "unsupported unsize to slice/str: source tail kind {:?} (source pointee {:?}, target pointee {:?})",
                source_tail.as_ref().kind(),
                source_pointee.as_ref().kind(),
                target_pointee.as_ref().kind(),
            ),
        },
        _ => panic!(
            "unsupported unsize target tail kind {:?} (source tail {:?}, source pointee {:?}, target pointee {:?})",
            target_tail.as_ref().kind(),
            source_tail.as_ref().kind(),
            source_pointee.as_ref().kind(),
            target_pointee.as_ref().kind(),
        ),
    }
}

/// Handle `PointerCoercion(Unsize)`.
///
/// Produces a fat pointer `(data_ptr, metadata)` for:
/// - `&T -> &dyn Trait` / `*const T -> *const dyn Trait` (vtable metadata)
/// - `&[T; N] -> &[T]` / `*const [T; N] -> *const [T]` (length metadata)
/// Reference: cg_clif/src/unsize.rs `coerce_unsized_into`
fn codegen_unsize_coercion(
    fx: &mut FunctionCx<'_, impl Module>,
    operand: &Operand,
    target_ty: &StoredTy,
    result_layout: &LayoutArc,
) -> CValue {
    let body = fx.ra_body();
    let from_cval = codegen_operand(fx, &operand.kind);
    let (data_ptr, from_meta) = match from_cval.layout.backend_repr {
        BackendRepr::ScalarPair(_, _) => {
            let (ptr, meta) = from_cval.load_scalar_pair(fx);
            (ptr, Some(meta))
        }
        _ => (from_cval.load_scalar(fx), None),
    };

    // Extract source/target pointee types.
    let target_pointee = target_ty
        .as_ref()
        .builtin_deref(true)
        .expect("Unsize target must be a pointer/reference type");
    let from_ty = operand_ty(fx.db(), body, &operand.kind);
    let source_pointee = from_ty
        .as_ref()
        .builtin_deref(true)
        .expect("Unsize source must be a pointer/reference type");
    let metadata = unsize_metadata_for_pointees(
        fx,
        source_pointee.store(),
        target_pointee.store(),
        from_meta,
    );

    match result_layout.backend_repr {
        BackendRepr::ScalarPair(_, _) => {
            CValue::by_val_pair(data_ptr, metadata, result_layout.clone())
        }
        BackendRepr::Scalar(_) => CValue::by_val(data_ptr, result_layout.clone()),
        _ => panic!("unsupported unsize result layout: {:?}", result_layout.backend_repr),
    }
}

/// Build or retrieve a vtable for `concrete_ty` implementing `trait_id`.
///
/// Vtable layout (matches rustc):
/// - Slot 0: drop_in_place fn ptr (null when the concrete type has no drop glue)
/// - Slot 1: size of concrete type (usize)
/// - Slot 2: alignment of concrete type (usize)
/// - Slot 3+: trait methods in declaration order
///
/// Reference: cg_clif/src/vtable.rs `get_vtable` + cg_clif/src/constant.rs `data_id_for_vtable`
fn trait_method_substs_for_receiver<'db>(
    db: &'db dyn HirDatabase,
    local_crate: base_db::Crate,
    trait_method_func_id: hir_def::FunctionId,
    concrete_self_ty: &StoredTy,
) -> GenericArgs<'db> {
    let interner = DbInterner::new_with(db, local_crate);
    let identity = GenericArgs::identity_for_item(interner, trait_method_func_id.into());
    assert!(!identity.is_empty(), "trait method has no generic args: {:?}", trait_method_func_id);
    GenericArgs::new_from_iter(
        interner,
        std::iter::once(concrete_self_ty.as_ref().into()).chain(identity.iter().skip(1)),
    )
}

fn declare_builtin_derive_method_shim(
    fx: &mut FunctionCx<'_, impl Module>,
    trait_method_func_id: hir_def::FunctionId,
    derive_impl_id: BuiltinDeriveImplId,
    derive_method: BuiltinDeriveImplMethod,
    resolved_args: GenericArgs<'_>,
) -> FuncId {
    assert!(
        matches!(derive_method, BuiltinDeriveImplMethod::fmt),
        "unsupported builtin-derive vtable method: {:?}::{:?}",
        derive_impl_id,
        derive_method
    );

    let sig =
        build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), trait_method_func_id, resolved_args)
            .expect("builtin-derive method shim sig");
    assert!(
        !sig.params.iter().any(|param| param.purpose == ArgumentPurpose::StructReturn),
        "builtin-derive method shim with sret return is unsupported: {:?}::{:?}",
        derive_impl_id,
        derive_method
    );

    let trait_method_name = symbol_mangling::mangle_function(
        fx.db(),
        trait_method_func_id,
        resolved_args,
        fx.ext_crate_disambiguators(),
    );
    let shim_name = format!("__builtin_derive_method_shim_{trait_method_name}");

    let shim_id = fx
        .module
        .declare_function(&shim_name, Linkage::Local, &sig)
        .expect("declare builtin-derive method shim");

    let mut func = cranelift_codegen::ir::Function::with_name_signature(
        cranelift_codegen::ir::UserFuncName::user(0, shim_id.as_u32()),
        sig.clone(),
    );
    let mut func_ctx = FunctionBuilderContext::new();
    {
        let mut bcx = FunctionBuilder::new(&mut func, &mut func_ctx);
        let entry_block = bcx.create_block();
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);

        // Builtin derive `Debug::fmt` is modeled as always succeeding here;
        // `fmt::Result` uses discriminant 0 for `Ok(())`.
        let mut ret_values = Vec::with_capacity(sig.returns.len());
        for ret in &sig.returns {
            let ty = ret.value_type;
            let zero = if ty.is_int() {
                bcx.ins().iconst(ty, 0)
            } else if ty == types::F32 {
                bcx.ins().f32const(0.0)
            } else if ty == types::F64 {
                bcx.ins().f64const(0.0)
            } else {
                panic!(
                    "builtin-derive method shim return type is unsupported: {:?}::{:?} ret_ty={:?}",
                    derive_impl_id, derive_method, ty
                );
            };
            ret_values.push(zero);
        }
        bcx.ins().return_(&ret_values);

        bcx.seal_all_blocks();
        bcx.finalize();
    }

    let mut ctx = Context::for_function(func);
    match fx.module.define_function(shim_id, &mut ctx) {
        Ok(()) | Err(ModuleError::DuplicateDefinition(_)) => {}
        Err(e) => panic!("define builtin-derive method shim: {e}"),
    }

    shim_id
}

fn get_or_create_vtable(
    fx: &mut FunctionCx<'_, impl Module>,
    concrete_ty: StoredTy,
    trait_id: TraitId,
) -> Value {
    let ptr_size = fx.dl.pointer_size().bytes() as usize;
    let drop_interner = DbInterner::new_with(fx.db(), fx.local_crate());
    let concrete_has_drop_glue =
        hir_ty::drop::has_drop_glue_mono(drop_interner, concrete_ty.as_ref());

    // Get concrete type layout for size/align
    let concrete_layout = fx
        .db()
        .layout_of_ty(concrete_ty.clone(), fx.env().clone())
        .expect("layout error for vtable concrete type");
    let concrete_size = concrete_layout.size.bytes();
    let concrete_align = concrete_layout.align.abi.bytes();

    // Find the impl for this concrete type.
    let krate = fx.local_crate();
    let interner = DbInterner::new_no_crate(fx.db());

    // Simplify the concrete type for lookup (same approach as hir-ty's method_resolution)
    use rustc_type_ir::fast_reject::{TreatParams, simplify_type};
    let simplified =
        simplify_type(interner, concrete_ty.as_ref(), TreatParams::InstantiateWithInfer)
            .expect("cannot simplify concrete type for vtable lookup");

    // Closure traits (`Fn` / `FnMut` / `FnOnce`) are built-in and often don't
    // show up as explicit impl items in TraitImpls. In that case, wire the
    // vtable method slot directly to the closure body symbol.
    let closure_fallback = match concrete_ty.as_ref().kind() {
        TyKind::Closure(closure_id, closure_subst) => Some((closure_id.0, closure_subst.store())),
        _ => None,
    };

    // Collect vtable methods in declaration order.
    // For closure fallback (`Fn`/`FnMut`/`FnOnce`), include supertrait methods
    // to mirror rustc's vtable shape (e.g. dyn FnMut has call_once + call_mut).
    let method_func_ids: Vec<hir_def::FunctionId> = if closure_fallback.is_some() {
        let mut trait_hierarchy: Vec<TraitId> =
            supertrait_def_ids(interner, trait_id.into()).map(|it| it.0).collect();
        trait_hierarchy.reverse();
        let mut seen_traits = std::collections::HashSet::new();
        trait_hierarchy.retain(|it| seen_traits.insert(*it));

        trait_hierarchy
            .into_iter()
            .flat_map(|tid| {
                tid.trait_items(fx.db()).items.iter().filter_map(|(_name, item)| match item {
                    AssocItemId::FunctionId(fid) => Some(*fid),
                    _ => None,
                })
            })
            .collect()
    } else {
        trait_id
            .trait_items(fx.db())
            .items
            .iter()
            .filter_map(|(_name, item)| match item {
                AssocItemId::FunctionId(fid) => Some(*fid),
                _ => None,
            })
            .collect()
    };
    let num_methods = method_func_ids.len();

    // Build unique vtable name
    let vtable_name = format!(
        "__vtable_{}_for_{:?}",
        trait_id
            .trait_items(fx.db())
            .items
            .first()
            .map(|(n, _)| n.as_str().to_string())
            .unwrap_or_default(),
        simplified,
    );

    // Declare the vtable data object
    let data_id = fx
        .module
        .declare_data(&vtable_name, Linkage::Local, false, false)
        .expect("declare vtable data");

    // Build vtable data
    let total_size = ptr_size * (3 + num_methods);
    let mut data = DataDescription::new();
    let mut vtable_bytes = vec![0u8; total_size];

    // Slot 0: drop_in_place — left null for trivially dropless types.

    // Slot 1: size
    vtable_bytes[ptr_size..ptr_size * 2]
        .copy_from_slice(&(concrete_size as u64).to_le_bytes()[..ptr_size]);

    // Slot 2: alignment
    vtable_bytes[ptr_size * 2..ptr_size * 3]
        .copy_from_slice(&(concrete_align as u64).to_le_bytes()[..ptr_size]);

    data.define(vtable_bytes.into_boxed_slice());

    if concrete_has_drop_glue {
        let mut drop_sig = Signature::new(fx.isa.default_call_conv());
        drop_sig.params.push(AbiParam::new(fx.pointer_type));
        let drop_fn_name = symbol_mangling::mangle_drop_in_place(
            fx.db(),
            concrete_ty.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        let drop_func_id = fx
            .module
            .declare_function(&drop_fn_name, Linkage::Import, &drop_sig)
            .expect("declare vtable drop_in_place");
        let drop_func_ref = fx.module.declare_func_in_data(drop_func_id, &mut data);
        data.write_function_addr(0, drop_func_ref);
    }

    // Slot 3+: trait method fn ptrs — emit as relocations
    let closure_func_id = closure_fallback.as_ref().map(|(closure_id, closure_subst)| {
        let closure_body = fx
            .db()
            .monomorphized_mir_body_for_closure(
                *closure_id,
                closure_subst.clone(),
                fx.env().clone(),
            )
            .expect("failed to get closure MIR for vtable");
        let closure_sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body)
            .expect("closure sig for vtable");
        let closure_name = symbol_mangling::mangle_closure(
            fx.db(),
            *closure_id,
            closure_subst.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        fx.module
            .declare_function(&closure_name, Linkage::Import, &closure_sig)
            .expect("declare closure vtable method")
    });

    for (method_idx, trait_method_func_id) in method_func_ids.iter().enumerate() {
        let trait_method_name = fx.db().function_signature(*trait_method_func_id).name.clone();
        let func_id = if closure_fallback.is_some() {
            let closure_func_id = closure_func_id.expect("closure vtable method func id");
            assert!(
                matches!(trait_method_name.as_str(), "call" | "call_mut" | "call_once"),
                "closure fallback cannot build vtable entry for trait method `{}`",
                trait_method_name.as_str()
            );
            closure_func_id
        } else {
            let trait_method_subst = trait_method_substs_for_receiver(
                fx.db(),
                krate,
                *trait_method_func_id,
                &concrete_ty,
            );
            let resolved_method = match fx.db().lookup_impl_method(
                fx.env().as_ref(),
                *trait_method_func_id,
                trait_method_subst,
            ) {
                (Either::Left(resolved_id), resolved_args) => {
                    Either::Left((resolved_id, resolved_args.store()))
                }
                (Either::Right((derive_impl_id, derive_method)), resolved_args) => {
                    Either::Right((derive_impl_id, derive_method, resolved_args.store()))
                }
            };
            match resolved_method {
                Either::Left((impl_func_id, impl_generic_args)) => {
                    let impl_generic_args = impl_generic_args.as_ref();
                    // Declare/import the impl function.
                    let impl_body = fx
                        .db()
                        .monomorphized_mir_body(
                            impl_func_id.into(),
                            impl_generic_args.clone().store(),
                            fx.env().clone(),
                        )
                        .expect("failed to get impl method MIR for vtable");
                    let impl_sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &impl_body)
                        .expect("impl method sig for vtable");
                    let impl_fn_name = symbol_mangling::mangle_function(
                        fx.db(),
                        impl_func_id,
                        impl_generic_args,
                        fx.ext_crate_disambiguators(),
                    );

                    fx.module
                        .declare_function(&impl_fn_name, Linkage::Import, &impl_sig)
                        .expect("declare vtable method")
                }
                Either::Right((derive_impl_id, derive_method, resolved_args)) => {
                    declare_builtin_derive_method_shim(
                        fx,
                        *trait_method_func_id,
                        derive_impl_id,
                        derive_method,
                        resolved_args.as_ref(),
                    )
                }
            }
        };

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
                    Either::Left(field_id) => field_id.local_id.into_raw().into_u32() as usize,
                    Either::Right(tuple_field_id) => tuple_field_id.index as usize,
                };

                // Determine the field type from the current type
                let field_ty = field_type(fx.db(), &cur_ty, field);
                let field_layout = fx
                    .db()
                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                    .expect("field layout error");

                cplace = cplace.place_field(fx, field_idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Deref => {
                let inner_ty =
                    cur_ty.as_ref().builtin_deref(true).expect("deref on non-pointer type");
                let inner_layout = fx
                    .db()
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
                let variant_layout = variant_layout(&cplace.layout, *variant_idx);
                cplace = cplace.downcast_variant(variant_layout);
                // cur_ty stays the same (Downcast is just a type assertion)
            }
            ProjectionElem::ClosureField(idx) => {
                // Closure captures are stored as fields of the closure struct
                let field_ty = closure_field_type(fx.db(), &cur_ty, *idx);
                let field_layout = fx
                    .db()
                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                    .expect("closure field layout error");
                cplace = cplace.place_field(fx, *idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Index(index_local) => {
                let index_place = fx.local_place(*index_local).clone();
                let index_val = index_place.to_cvalue(fx).load_scalar(fx);

                // r-a MIR can emit `Index` directly on `&[T]` / `&[T; N]` without an
                // explicit preceding `Deref` projection.
                if !matches!(cur_ty.as_ref().kind(), TyKind::Array(_, _) | TyKind::Slice(_)) {
                    if let Some(inner_ty) = cur_ty.as_ref().builtin_deref(true)
                        && matches!(inner_ty.kind(), TyKind::Array(_, _) | TyKind::Slice(_))
                    {
                        let inner_layout = fx
                            .db()
                            .layout_of_ty(inner_ty.store(), fx.env().clone())
                            .expect("index autoderef layout error");

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
                }
                let is_slice = matches!(cur_ty.as_ref().kind(), TyKind::Slice(_));
                let elem_ty = match cur_ty.as_ref().kind() {
                    TyKind::Array(elem, _) | TyKind::Slice(elem) => elem.store(),
                    _ => panic!("Index on non-array/slice type"),
                };
                let elem_layout =
                    fx.db().layout_of_ty(elem_ty.clone(), fx.env().clone()).expect("elem layout");
                let offset = fx.bcx.ins().imul_imm(index_val, elem_layout.size.bytes() as i64);
                // Arrays in registers → spill to memory
                if cplace.is_register() {
                    let cval = cplace.to_cvalue(fx);
                    let ptr = cval.force_stack(fx);
                    cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                }
                let base_ptr = if is_slice { cplace.to_ptr_unsized().0 } else { cplace.to_ptr() };
                cplace = CPlace::for_ptr(
                    base_ptr.offset_value(&mut fx.bcx, fx.pointer_type, offset),
                    elem_layout,
                );
                cur_ty = elem_ty;
            }
            ProjectionElem::ConstantIndex { offset, from_end } => {
                // r-a MIR can emit `ConstantIndex` directly on `&[T]` / `&[T; N]`
                // without an explicit `Deref` projection.
                if !matches!(cur_ty.as_ref().kind(), TyKind::Array(_, _) | TyKind::Slice(_)) {
                    if let Some(inner_ty) = cur_ty.as_ref().builtin_deref(true)
                        && matches!(inner_ty.kind(), TyKind::Array(_, _) | TyKind::Slice(_))
                    {
                        let inner_layout = fx
                            .db()
                            .layout_of_ty(inner_ty.store(), fx.env().clone())
                            .expect("constant index autoderef layout error");

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
                }
                let is_slice = matches!(cur_ty.as_ref().kind(), TyKind::Slice(_));
                let elem_ty = match cur_ty.as_ref().kind() {
                    TyKind::Array(elem, _) | TyKind::Slice(elem) => elem.store(),
                    _ => panic!("ConstantIndex on non-array/slice type"),
                };
                let elem_layout =
                    fx.db().layout_of_ty(elem_ty.clone(), fx.env().clone()).expect("elem layout");
                let index = if !*from_end {
                    fx.bcx.ins().iconst(fx.pointer_type, *offset as i64)
                } else {
                    // from_end: use array length or slice metadata length.
                    let len = if is_slice {
                        cplace.to_ptr_unsized().1
                    } else {
                        match &cplace.layout.fields {
                            rustc_abi::FieldsShape::Array { count, .. } => {
                                fx.bcx.ins().iconst(fx.pointer_type, *count as i64)
                            }
                            _ => panic!("ConstantIndex from_end on non-array layout"),
                        }
                    };
                    fx.bcx.ins().iadd_imm(len, -(*offset as i64))
                };
                if cplace.is_register() {
                    let cval = cplace.to_cvalue(fx);
                    let ptr = cval.force_stack(fx);
                    cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                }
                let byte_offset = fx.bcx.ins().imul_imm(index, elem_layout.size.bytes() as i64);
                let base_ptr = if is_slice { cplace.to_ptr_unsized().0 } else { cplace.to_ptr() };
                cplace = CPlace::for_ptr(
                    base_ptr.offset_value(&mut fx.bcx, fx.pointer_type, byte_offset),
                    elem_layout,
                );
                cur_ty = elem_ty;
            }
            ProjectionElem::Subslice { from, to } => match cur_ty.as_ref().kind() {
                TyKind::Array(elem_ty, len) => {
                    let elem_layout = fx
                        .db()
                        .layout_of_ty(elem_ty.store(), fx.env().clone())
                        .expect("array subslice elem layout");

                    if cplace.is_register() {
                        let cval = cplace.to_cvalue(fx);
                        let ptr = cval.force_stack(fx);
                        cplace = CPlace::for_ptr(ptr, cplace.layout.clone());
                    }

                    let from_bytes = i64::try_from(*from)
                        .unwrap()
                        .checked_mul(i64::try_from(elem_layout.size.bytes()).unwrap())
                        .expect("array subslice offset overflow");
                    let base_ptr =
                        cplace.to_ptr().offset_i64(&mut fx.bcx, fx.pointer_type, from_bytes);

                    let new_len = try_const_usize(fx.db(), len)
                        .and_then(|n| n.checked_sub(u128::from(*from + *to)));
                    let new_len_const = usize_const(fx.db(), new_len, fx.local_crate());
                    let interner = DbInterner::new_no_crate(fx.db());
                    let subslice_ty = hir_ty::next_solver::Ty::new_array_with_const_len(
                        interner,
                        elem_ty,
                        new_len_const,
                    );
                    let subslice_layout = fx
                        .db()
                        .layout_of_ty(subslice_ty.store(), fx.env().clone())
                        .expect("array subslice layout");

                    cplace = CPlace::for_ptr(base_ptr, subslice_layout);
                    cur_ty = subslice_ty.store();
                }
                TyKind::Slice(elem_ty) => {
                    let elem_layout = fx
                        .db()
                        .layout_of_ty(elem_ty.store(), fx.env().clone())
                        .expect("slice subslice elem layout");
                    let (ptr, len) = cplace.to_ptr_unsized();
                    let from_bytes = i64::try_from(*from)
                        .unwrap()
                        .checked_mul(i64::try_from(elem_layout.size.bytes()).unwrap())
                        .expect("slice subslice offset overflow");
                    let new_ptr = ptr.offset_i64(&mut fx.bcx, fx.pointer_type, from_bytes);
                    let total_trim = i64::try_from(*from + *to).unwrap();
                    let new_len = fx.bcx.ins().iadd_imm(len, -total_trim);
                    cplace = CPlace::for_ptr_with_extra(new_ptr, new_len, cplace.layout.clone());
                }
                TyKind::Str => {
                    let (ptr, len) = cplace.to_ptr_unsized();
                    let from_bytes = i64::try_from(*from).unwrap();
                    let new_ptr = ptr.offset_i64(&mut fx.bcx, fx.pointer_type, from_bytes);
                    let total_trim = i64::try_from(*from + *to).unwrap();
                    let new_len = fx.bcx.ins().iadd_imm(len, -total_trim);
                    cplace = CPlace::for_ptr_with_extra(new_ptr, new_len, cplace.layout.clone());
                }
                _ => panic!("Subslice projection on non-array/slice/str type"),
            },
            ProjectionElem::OpaqueCast(_) => todo!("OpaqueCast projection"),
        }
    }
    cplace
}

fn type_is_copy_for_codegen(fx: &FunctionCx<'_, impl Module>, ty: &StoredTy) -> bool {
    let Some(copy_trait) = hir_def::lang_item::lang_items(fx.db(), fx.local_crate()).Copy else {
        return false;
    };
    hir_ty::traits::implements_trait_unique(ty.as_ref(), fx.db(), fx.env().as_ref(), copy_trait)
}

fn copy_operand_consumes_ownership(fx: &FunctionCx<'_, impl Module>, place: &Place) -> bool {
    let ty = place_ty(fx.db(), fx.ra_body(), place);
    !type_is_copy_for_codegen(fx, &ty)
}

fn operand_consumes_ownership(fx: &FunctionCx<'_, impl Module>, kind: &OperandKind) -> bool {
    match kind {
        OperandKind::Move(_) => true,
        OperandKind::Copy(place) => copy_operand_consumes_ownership(fx, place),
        OperandKind::Constant { .. } | OperandKind::Static(_) => false,
    }
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
                Either::Left(field_id) => field_id.local_id.into_raw().into_u32() as usize,
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
fn closure_field_type(db: &dyn HirDatabase, closure_ty: &StoredTy, idx: usize) -> StoredTy {
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
pub(crate) fn variant_layout(
    parent_layout: &TArc<Layout>,
    variant_idx: VariantIdx,
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
pub(crate) fn codegen_icmp_imm(
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

fn enum_id_from_ty(ty: &StoredTy) -> Option<hir_def::EnumId> {
    let TyKind::Adt(adt_def, _) = ty.as_ref().kind() else {
        return None;
    };
    match adt_def.inner().id {
        hir_def::AdtId::EnumId(enum_id) => Some(enum_id),
        _ => None,
    }
}

fn enum_variant_discriminant_bits(
    db: &dyn HirDatabase,
    enum_ty: &StoredTy,
    variant_index: VariantIdx,
) -> u128 {
    let enum_id = enum_id_from_ty(enum_ty)
        .unwrap_or_else(|| panic!("expected enum type for discriminant write, got: {:?}", enum_ty));
    let variants = &enum_id.enum_variants(db).variants;
    let &(variant_id, _, _) = variants.get(variant_index.as_usize()).unwrap_or_else(|| {
        panic!("variant index {} out of range for enum {:?}", variant_index.as_u32(), enum_id,)
    });
    let discr = db.const_eval_discriminant(variant_id).unwrap_or_else(|e| {
        panic!(
            "failed to evaluate discriminant for enum {:?} variant {}: {e:?}",
            enum_id,
            variant_index.as_u32(),
        )
    });
    discr as u128
}

fn truncate_bits_to_clif_int_ty(bits: u128, ty: Type) -> u128 {
    debug_assert!(ty.is_int(), "expected integer type for switch index, got {ty:?}");
    let bit_width = ty.bits();
    if bit_width >= 128 {
        bits
    } else {
        let mask = (1u128 << bit_width) - 1;
        bits & mask
    }
}

fn remap_variant_index_to_discriminant(
    fx: &mut FunctionCx<'_, impl Module>,
    enum_ty: &StoredTy,
    variant_index_value: Value,
) -> Value {
    let Some(enum_id) = enum_id_from_ty(enum_ty) else {
        return variant_index_value;
    };

    let variant_ids: Vec<_> = {
        let db = fx.db();
        enum_id.enum_variants(db).variants.iter().map(|&(variant_id, _, _)| variant_id).collect()
    };

    let discr_ty = fx.bcx.func.dfg.value_type(variant_index_value);
    let mut mapped = variant_index_value;
    for (idx, variant_id) in variant_ids.into_iter().enumerate() {
        let discr_bits = {
            let db = fx.db();
            db.const_eval_discriminant(variant_id).unwrap_or_else(|e| {
                panic!(
                    "failed to evaluate discriminant for enum {:?} variant {}: {e:?}",
                    enum_id, idx,
                )
            }) as u128
        };

        if discr_bits == idx as u128 {
            continue;
        }

        let is_variant = codegen_icmp_imm(fx, IntCC::Equal, variant_index_value, idx as i128);
        let discr_value = iconst_from_bits(&mut fx.bcx, discr_ty, discr_bits);
        mapped = fx.bcx.ins().select(is_variant, discr_value, mapped);
    }
    mapped
}

/// Pick which scalar lane of a `ScalarPair` holds the enum tag.
///
/// For niche-encoded enums like `Option<(usize, &T)>`, `tag_field` can point
/// at a composite payload field while the actual niche tag lives in one of that
/// field's scalar ABI lanes.
fn scalar_pair_tag_lane(a: Scalar, b: Scalar, tag: &Scalar, tag_field: rac_abi::FieldIdx) -> usize {
    if a == *tag && b != *tag {
        0
    } else if b == *tag && a != *tag {
        1
    } else {
        match tag_field.as_usize() {
            0 => 0,
            1 => 1,
            idx => panic!("invalid scalar-pair tag field index: {idx}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Statement codegen
// ---------------------------------------------------------------------------

fn codegen_statement(fx: &mut FunctionCx<'_, impl Module>, stmt: &StatementKind) {
    match stmt {
        StatementKind::Assign(place, rvalue) => {
            codegen_assign(fx, place, rvalue);
            // Mark the destination local as live for drop-flag tracking.
            if place.projection.lookup(&fx.ra_body().projection_store).is_empty() {
                fx.set_drop_flag(place.local);
            }
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
        StatementKind::Nop | StatementKind::FakeRead(_) => {}
        StatementKind::Deinit(_) => {}
        StatementKind::SetDiscriminant { place, variant_index } => {
            let enum_ty = {
                let body = fx.ra_body();
                place_ty(fx.db(), body, place)
            };
            let dest = codegen_place(fx, place);
            codegen_set_discriminant(fx, &dest, &enum_ty, *variant_index);
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
            let dest_ty = {
                let body = fx.ra_body();
                place_ty(fx.db(), body, place)
            };
            codegen_aggregate(fx, kind, operands, &dest_ty, dest);
            return;
        }
        Rvalue::Ref(_, ref_place) | Rvalue::AddressOf(_, ref_place) => {
            let place = codegen_place(fx, ref_place);
            let ref_val = place.place_ref(fx, dest.layout.clone());
            dest.write_cvalue(fx, ref_val);
            return;
        }
        Rvalue::Discriminant(disc_place) => {
            let disc_ty = {
                let body = fx.ra_body();
                place_ty(fx.db(), body, disc_place)
            };
            let disc_cplace = codegen_place(fx, disc_place);
            let disc_val = codegen_get_discriminant(fx, &disc_cplace, &disc_ty, &dest.layout);
            dest.write_cvalue(fx, CValue::by_val(disc_val, dest.layout.clone()));
            return;
        }
        Rvalue::Repeat(operand, count) => {
            let elem_val = codegen_operand(fx, &operand.kind);
            let count_val = const_to_u64(fx.db(), count.as_ref());
            let elem_layout = elem_val.layout.clone();
            for i in 0..count_val {
                let field_place = dest.place_field(fx, i as usize, elem_layout.clone());
                field_place.write_cvalue(fx, elem_val.clone());
            }
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
                    let len_val = const_to_u64(fx.db(), len) as i64;
                    CValue::by_val(
                        fx.bcx.ins().iconst(fx.pointer_type, len_val),
                        result_layout.clone(),
                    )
                }
                TyKind::Slice(_) => {
                    // Slice: length is the metadata of the fat pointer.
                    // The place is behind a fat pointer; get the metadata
                    // that was stored during deref projection.
                    let cplace = codegen_place(fx, place);
                    let meta = cplace.get_extra().expect("slice Len requires fat pointer metadata");
                    CValue::by_val(meta, result_layout.clone())
                }
                _ => todo!("Len on non-array/slice type: {:?}", place_ty),
            }
        }
        Rvalue::CopyForDeref(place) => {
            // CopyForDeref is semantically identical to Copy at the codegen level.
            codegen_place(fx, place).to_cvalue(fx)
        }
        Rvalue::Repeat(..) => {
            unreachable!("Repeat should be handled in codegen_assign")
        }
        Rvalue::ShallowInitBoxWithAlloc(ty) => {
            let pointee_layout = fx
                .db()
                .layout_of_ty(ty.clone(), fx.env().clone())
                .expect("ShallowInitBoxWithAlloc: pointee layout error");
            let ptr = codegen_box_alloc(fx, &pointee_layout);
            CValue::by_val(ptr, result_layout.clone())
        }
        Rvalue::ShallowInitBox(operand, _ty) => {
            // Transmute *mut u8 → Box<T>. At codegen level, just reinterpret.
            let ptr = codegen_operand(fx, &operand.kind);
            CValue::by_val(ptr.load_scalar(fx), result_layout.clone())
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

/// Write ADT fields into `dest` with proper variant handling.
///
/// Handles single-variant ADT scalar fast path and multi-variant enums
/// (spilling to a temp stack slot when needed).
/// Used by both `AggregateKind::Adt` and `codegen_adt_constructor_call`.
fn codegen_adt_fields(
    fx: &mut FunctionCx<'_, impl Module>,
    variant_idx: VariantIdx,
    field_vals: &[CValue],
    dest_ty: &StoredTy,
    dest: CPlace,
) {
    use rustc_abi::Variants;
    let is_single_variant = matches!(&dest.layout.variants, Variants::Single { .. });

    // Fast path: Scalar ADT with single variant (wrapper struct)
    if is_single_variant {
        if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
            let lanes = collect_scalar_abi_lanes(fx, field_vals);
            assert_eq!(lanes.len(), 1, "Scalar ADT expects 1 ABI scalar lane");
            dest.write_cvalue(fx, CValue::by_val(lanes[0], dest.layout.clone()));
            codegen_set_discriminant(fx, &dest, dest_ty, variant_idx);
            return;
        }
    }

    // General case: for multi-variant enums on register places,
    // spill to memory so field projections use correct offsets.
    let use_temp = matches!(&dest.layout.variants, Variants::Multiple { .. }) && dest.is_register();
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

    for (i, field_cval) in field_vals.iter().enumerate() {
        if field_cval.layout.is_zst() {
            continue;
        }
        let field_layout = field_cval.layout.clone();
        let field_place = variant_dest.place_field(fx, i, field_layout);
        field_place.write_cvalue(fx, field_cval.clone());
    }

    codegen_set_discriminant(fx, &work_dest, dest_ty, variant_idx);

    if let Some(orig) = original_dest {
        let cval = work_dest.to_cvalue(fx);
        orig.write_cvalue(fx, cval);
    }
}

/// Collect ABI scalar lanes from aggregate fields.
///
/// For ABI scalar/scalar-pair destinations, source fields may be nested wrappers,
/// including single non-ZST fields whose own representation is `ScalarPair`.
/// This helper expands each non-ZST field into its scalar lanes so callers can
/// validate lane count against destination ABI, instead of assuming one lane per field.
fn collect_scalar_abi_lanes(fx: &mut FunctionCx<'_, impl Module>, fields: &[CValue]) -> Vec<Value> {
    let mut lanes = Vec::new();

    for field in fields {
        if field.layout.is_zst() {
            continue;
        }
        match field.layout.backend_repr {
            BackendRepr::Scalar(_) => lanes.push(field.load_scalar(fx)),
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = field.load_scalar_pair(fx);
                lanes.push(a);
                lanes.push(b);
            }
            _ => {
                panic!(
                    "non-scalar field representation in scalar ABI aggregate: {:?}",
                    field.layout.backend_repr
                );
            }
        }
    }

    lanes
}

fn codegen_aggregate(
    fx: &mut FunctionCx<'_, impl Module>,
    kind: &hir_ty::mir::AggregateKind,
    operands: &[Operand],
    dest_ty: &StoredTy,
    dest: CPlace,
) {
    use hir_ty::mir::AggregateKind;
    match kind {
        AggregateKind::Tuple(_)
        | AggregateKind::Array(_)
        | AggregateKind::Closure(_)
        | AggregateKind::Coroutine(_)
        | AggregateKind::CoroutineClosure(_) => {
            let field_vals: Vec<_> =
                operands.iter().map(|op| codegen_operand(fx, &op.kind)).collect();

            // For single scalar, construct directly
            if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
                let lanes = collect_scalar_abi_lanes(fx, &field_vals);
                assert_eq!(lanes.len(), 1, "Scalar aggregate expects 1 ABI scalar lane");
                dest.write_cvalue(fx, CValue::by_val(lanes[0], dest.layout.clone()));
                return;
            }

            // General case: write each field to the destination place
            for (i, field_cval) in field_vals.iter().enumerate() {
                let field_layout = field_cval.layout.clone();
                let field_place = dest.place_field(fx, i, field_layout);
                field_place.write_cvalue(fx, field_cval.clone());
            }
        }
        AggregateKind::Adt(variant_id, _subst) => {
            let variant_idx = variant_id_to_idx(fx.db(), *variant_id);
            let field_vals: Vec<_> =
                operands.iter().map(|op| codegen_operand(fx, &op.kind)).collect();
            codegen_adt_fields(fx, variant_idx, &field_vals, dest_ty, dest);
        }
        AggregateKind::Union(_, _) => {
            // Union aggregate: single active field written at offset 0.
            assert_eq!(operands.len(), 1, "union aggregate should have exactly 1 operand");
            let field_val = codegen_operand(fx, &operands[0].kind);
            let field_layout = field_val.layout.clone();
            let field_place = dest.place_field(fx, 0, field_layout);
            field_place.write_cvalue(fx, field_val);
        }
        AggregateKind::RawPtr(_, _) => {
            // RawPtr aggregate: (data_ptr, metadata) → thin or fat pointer
            assert_eq!(operands.len(), 2, "RawPtr aggregate must have 2 operands");
            let data = codegen_operand(fx, &operands[0].kind);
            let meta = codegen_operand(fx, &operands[1].kind);
            let result = if meta.layout.is_zst() {
                CValue::by_val(data.load_scalar(fx), dest.layout.clone())
            } else {
                CValue::by_val_pair(data.load_scalar(fx), meta.load_scalar(fx), dest.layout.clone())
            };
            dest.write_cvalue(fx, result);
        }
    }
}

/// Read the discriminant of an enum place.
/// Follows upstream cg_clif/src/discriminant.rs `codegen_get_discriminant`.
pub(crate) fn codegen_get_discriminant(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &CPlace,
    place_ty: &StoredTy,
    dest_layout: &LayoutArc,
) -> Value {
    use rustc_abi::Variants;
    let BackendRepr::Scalar(dest_scalar) = dest_layout.backend_repr else {
        panic!("discriminant destination must be scalar");
    };
    let dest_clif_ty = scalar_to_clif_type(fx.dl, &dest_scalar);

    match &place.layout.variants {
        Variants::Single { index } => {
            let discr_val = enum_variant_discriminant_bits(fx.db(), place_ty, *index);
            iconst_from_bits(&mut fx.bcx, dest_clif_ty, discr_val)
        }
        Variants::Multiple { tag, tag_field, tag_encoding, .. } => {
            use rustc_abi::TagEncoding;
            let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);

            // Read the tag value — handle register and memory places
            let tag_val = match place.layout.backend_repr {
                BackendRepr::Scalar(_) => place.to_cvalue(fx).load_scalar(fx),
                BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                    let (a, b) = place.to_cvalue(fx).load_scalar_pair(fx);
                    match scalar_pair_tag_lane(a_scalar, b_scalar, tag, *tag_field) {
                        0 => a,
                        1 => b,
                        _ => unreachable!(),
                    }
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
                        let is_niche =
                            codegen_icmp_imm(fx, IntCC::Equal, tag_val, *niche_start as i128);
                        let tagged_discr = iconst_from_bits(
                            &mut fx.bcx,
                            dest_clif_ty,
                            niche_variants.start().as_u32().into(),
                        );
                        (is_niche, tagged_discr, 0)
                    } else {
                        // General case: compute relative_tag, check range
                        let niche_start_val =
                            iconst_from_bits(&mut fx.bcx, tag_clif_ty, *niche_start);
                        let relative_discr = fx.bcx.ins().isub(tag_val, niche_start_val);
                        let cast_tag = codegen_intcast(fx, relative_discr, dest_clif_ty, false);
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
                        let delta_val = iconst_from_bits(&mut fx.bcx, dest_clif_ty, delta);
                        fx.bcx.ins().iadd(tagged_discr, delta_val)
                    };

                    let untagged_variant_val = iconst_from_bits(
                        &mut fx.bcx,
                        dest_clif_ty,
                        untagged_variant.as_u32().into(),
                    );
                    let variant_index =
                        fx.bcx.ins().select(is_niche, tagged_discr, untagged_variant_val);
                    remap_variant_index_to_discriminant(fx, place_ty, variant_index)
                }
            }
        }
        Variants::Empty => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
            iconst_from_bits(&mut fx.bcx, dest_clif_ty, 0)
        }
    }
}

/// Set the discriminant for an enum place.
/// Follows upstream cg_clif/src/discriminant.rs `codegen_set_discriminant`.
pub(crate) fn codegen_set_discriminant(
    fx: &mut FunctionCx<'_, impl Module>,
    place: &CPlace,
    enum_ty: &StoredTy,
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
                    let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);
                    let discr_bits =
                        enum_variant_discriminant_bits(fx.db(), enum_ty, variant_index);
                    let discr_bits = tag.size(fx.dl).truncate(discr_bits);
                    let to = iconst_from_bits(&mut fx.bcx, tag_clif_ty, discr_bits);

                    if let BackendRepr::ScalarPair(a_scalar, b_scalar) = place.layout.backend_repr {
                        let lane = scalar_pair_tag_lane(a_scalar, b_scalar, tag, *tag_field);
                        let tag_place = place.place_scalar_pair_lane(fx, lane, tag_layout);
                        tag_place.write_cvalue(fx, CValue::by_val(to, tag_place.layout.clone()));
                    } else {
                        let ptr = place.place_field(fx, tag_field.as_usize(), tag_layout);
                        ptr.write_cvalue(fx, CValue::by_val(to, ptr.layout.clone()));
                    }
                }
                TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                    if variant_index != *untagged_variant {
                        let niche_type = scalar_to_clif_type(fx.dl, tag);
                        let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                        let niche_value = (niche_value as u128).wrapping_add(*niche_start);
                        let niche_value = iconst_from_bits(&mut fx.bcx, niche_type, niche_value);

                        if let BackendRepr::ScalarPair(a_scalar, b_scalar) =
                            place.layout.backend_repr
                        {
                            let lane = scalar_pair_tag_lane(a_scalar, b_scalar, tag, *tag_field);
                            let niche = place.place_scalar_pair_lane(fx, lane, tag_layout);
                            niche.write_cvalue(
                                fx,
                                CValue::by_val(niche_value, niche.layout.clone()),
                            );
                        } else {
                            let niche = place.place_field(fx, tag_field.as_usize(), tag_layout);
                            niche.write_cvalue(
                                fx,
                                CValue::by_val(niche_value, niche.layout.clone()),
                            );
                        }
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

    match lhs_cval.layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            let lhs_val = lhs_cval.load_scalar(fx);
            let rhs_val = rhs_cval.load_scalar(fx);

            // Overflow binops return (T, bool) as a ScalarPair
            if matches!(
                op,
                BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
            ) {
                let Primitive::Int(_, signed) = scalar.primitive() else {
                    panic!("overflow binop on non-integer type");
                };
                let (res, has_overflow) =
                    codegen_checked_int_binop(fx, op, lhs_val, rhs_val, signed);
                let BackendRepr::ScalarPair(_, overflow_scalar) = result_layout.backend_repr else {
                    panic!("overflow binop must return ScalarPair");
                };
                let overflow_ty = scalar_to_clif_type(fx.dl, &overflow_scalar);
                let has_overflow = if fx.bcx.func.dfg.value_type(has_overflow) == overflow_ty {
                    has_overflow
                } else {
                    let one = fx.bcx.ins().iconst(overflow_ty, 1);
                    let zero = fx.bcx.ins().iconst(overflow_ty, 0);
                    fx.bcx.ins().select(has_overflow, one, zero)
                };
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
                        let pointee_layout = fx
                            .db()
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
                        let byte_offset =
                            fx.bcx.ins().imul_imm(rhs, pointee_layout.size.bytes() as i64);
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
        BackendRepr::ScalarPair(_, _) => {
            let (lhs_a, lhs_b) = lhs_cval.load_scalar_pair(fx);
            let (rhs_a, rhs_b) = rhs_cval.load_scalar_pair(fx);
            let val = match op {
                BinOp::Eq => {
                    let a_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_a, rhs_a);
                    let b_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_b, rhs_b);
                    fx.bcx.ins().band(a_eq, b_eq)
                }
                BinOp::Ne => {
                    let a_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_a, rhs_a);
                    let b_ne = fx.bcx.ins().icmp(IntCC::NotEqual, lhs_b, rhs_b);
                    fx.bcx.ins().bor(a_ne, b_ne)
                }
                BinOp::Lt | BinOp::Le | BinOp::Ge | BinOp::Gt => {
                    let cc = bin_op_to_intcc(op, false);
                    let a_eq = fx.bcx.ins().icmp(IntCC::Equal, lhs_a, rhs_a);
                    let a_cmp = fx.bcx.ins().icmp(cc, lhs_a, rhs_a);
                    let b_cmp = fx.bcx.ins().icmp(cc, lhs_b, rhs_b);
                    fx.bcx.ins().select(a_eq, b_cmp, a_cmp)
                }
                _ => panic!("unsupported ScalarPair binop: {:?}", op),
            };
            CValue::by_val(val, result_layout.clone())
        }
        _ => panic!("expected scalar or scalarpair type for binop lhs"),
    }
}

pub(crate) fn codegen_int_binop(
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

pub(crate) fn codegen_checked_int_binop(
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
                    let has_underflow =
                        b.ins().icmp_imm(IntCC::SignedLessThan, val_w, -(1i64 << (ty.bits() - 1)));
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

pub(crate) fn codegen_saturating_int_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    op: &BinOp,
    lhs: Value,
    rhs: Value,
    signed: bool,
) -> Value {
    let ty = fx.bcx.func.dfg.value_type(lhs);
    let (min, max) = int_min_max_values(&mut fx.bcx, ty, signed);

    let checked_op = match op {
        BinOp::Add => BinOp::AddWithOverflow,
        BinOp::Sub => BinOp::SubWithOverflow,
        _ => unreachable!("not a saturating int binop: {op:?}"),
    };
    let (val, has_overflow) = codegen_checked_int_binop(fx, &checked_op, lhs, rhs, signed);

    match (op, signed) {
        (BinOp::Add, false) => fx.bcx.ins().select(has_overflow, max, val),
        (BinOp::Sub, false) => fx.bcx.ins().select(has_overflow, min, val),
        (BinOp::Add, true) => {
            let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
            let sat_val = fx.bcx.ins().select(rhs_ge_zero, max, min);
            fx.bcx.ins().select(has_overflow, sat_val, val)
        }
        (BinOp::Sub, true) => {
            let rhs_ge_zero = fx.bcx.ins().icmp_imm(IntCC::SignedGreaterThanOrEqual, rhs, 0);
            let sat_val = fx.bcx.ins().select(rhs_ge_zero, min, max);
            fx.bcx.ins().select(has_overflow, sat_val, val)
        }
        _ => unreachable!(),
    }
}

pub(crate) fn codegen_float_binop(
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

    // PtrMetadata operates on ScalarPair, handle before load_scalar
    if matches!(op, UnOp::PtrMetadata) {
        return match cval.layout.backend_repr {
            BackendRepr::Scalar(_) => CValue::zst(result_layout.clone()),
            BackendRepr::ScalarPair(_, _) => {
                CValue::by_val(cval.load_scalar_pair(fx).1, result_layout.clone())
            }
            _ => panic!("PtrMetadata on unexpected repr"),
        };
    }

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
        UnOp::PtrMetadata => unreachable!("handled above"),
    };
    CValue::by_val(result, result_layout.clone())
}

fn codegen_static_operand(fx: &mut FunctionCx<'_, impl Module>, static_id: StaticId) -> CValue {
    let db = fx.db();
    let static_sig = db.static_signature(static_id);
    let is_mutable = static_sig.flags.contains(StaticFlags::MUTABLE);
    let is_extern = static_sig.flags.contains(StaticFlags::EXTERN);
    let is_local_static = static_id.krate(db) == fx.local_crate();
    let symbol_name = if static_sig.flags.contains(StaticFlags::EXTERN) {
        static_sig.name.as_str().to_owned()
    } else {
        symbol_mangling::mangle_static(db, static_id, fx.ext_crate_disambiguators())
    };

    let static_ref_ty = static_operand_ty(db, static_id);
    let layout =
        db.layout_of_ty(static_ref_ty, fx.env().clone()).expect("layout error for static operand");

    let ptr = if is_extern {
        let data_id = fx
            .module
            .declare_data(&symbol_name, Linkage::Import, is_mutable, false)
            .unwrap_or_else(|e| panic!("declare imported static `{symbol_name}` failed: {e}"));
        let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
        fx.bcx.ins().symbol_value(fx.pointer_type, local_data_id)
    } else {
        let const_eval_result = db.const_eval_static(static_id);
        if !is_local_static && const_eval_result.is_err() {
            let data_id = fx
                .module
                .declare_data(&symbol_name, Linkage::Import, is_mutable, false)
                .unwrap_or_else(|e| {
                    panic!("declare imported static `{symbol_name}` failed: {e}")
                });
            let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
            fx.bcx.ins().symbol_value(fx.pointer_type, local_data_id)
        } else {
            let data_id = fx
                .module
                .declare_data(&symbol_name, Linkage::Local, is_mutable, false)
                .unwrap_or_else(|e| panic!("declare static `{symbol_name}` failed: {e}"));

            match const_eval_result {
                Ok(konst) => {
                    let const_value = resolve_const_value(db, konst);
                    let mut data_desc = DataDescription::new();
                    data_desc.define(const_value.value.inner().memory.clone());

                    let pointee_layout = db
                        .layout_of_ty(static_pointee_ty(db, static_id).store(), fx.env().clone())
                        .expect("layout error for static pointee");
                    data_desc.set_align(pointee_layout.align.abi.bytes());

                    match fx.module.define_data(data_id, &data_desc) {
                        Ok(()) | Err(ModuleError::DuplicateDefinition(_)) => {}
                        Err(e) => panic!("define static `{symbol_name}` failed: {e}"),
                    }
                }
                Err(err) => {
                    panic!("const_eval_static failed for local static `{symbol_name}`: {err:?}",);
                }
            }

            let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
            fx.bcx.ins().symbol_value(fx.pointer_type, local_data_id)
        }
    };

    CValue::by_val(ptr, layout)
}

fn codegen_operand(fx: &mut FunctionCx<'_, impl Module>, kind: &OperandKind) -> CValue {
    match kind {
        OperandKind::Constant { konst, ty } => {
            let layout = fx
                .db()
                .layout_of_ty(ty.clone(), fx.env().clone())
                .expect("layout error for constant type");
            if layout.is_zst() {
                return CValue::zst(layout);
            }
            match layout.backend_repr {
                BackendRepr::Scalar(scalar) => {
                    let raw_bits = const_to_u128(fx.db(), konst.as_ref(), scalar.size(fx.dl));
                    let const_memory_map = {
                        let val = resolve_const_value(fx.db(), konst.as_ref());
                        val.value.inner().memory_map.clone()
                    };
                    let const_allocs = create_const_data_sections(fx, &const_memory_map);
                    let val = match scalar.primitive() {
                        Primitive::Pointer(_) => codegen_scalar_const(
                            fx,
                            &scalar,
                            raw_bits,
                            &const_allocs,
                            Some(&const_memory_map),
                            Some(ty),
                        ),
                        Primitive::Float(rustc_abi::Float::F32) => {
                            fx.bcx.ins().f32const(f32::from_bits(raw_bits as u32))
                        }
                        Primitive::Float(rustc_abi::Float::F64) => {
                            fx.bcx.ins().f64const(f64::from_bits(raw_bits as u64))
                        }
                        Primitive::Float(_) => todo!("f16/f128 constants"),
                        _ => {
                            let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                            // Integer constants can carry const-eval virtual addresses
                            // (e.g. pointer-to-int casts in const fn). If this looks like
                            // one, lower it as relocation + offset instead of raw bits.
                            if clif_ty == fx.pointer_type
                                && looks_like_const_eval_virtual_addr(raw_bits as usize)
                            {
                                if let Some(ptr) =
                                    const_ptr_from_data_alloc(fx, raw_bits as usize, &const_allocs)
                                {
                                    ptr
                                } else {
                                    iconst_from_bits(&mut fx.bcx, clif_ty, raw_bits)
                                }
                            } else {
                                iconst_from_bits(&mut fx.bcx, clif_ty, raw_bits)
                            }
                        }
                    };
                    CValue::by_val(val, layout)
                }
                BackendRepr::ScalarPair(a_scalar, b_scalar) => {
                    let (const_memory, const_memory_map) = {
                        let val = resolve_const_value(fx.db(), konst.as_ref());
                        let const_bytes = val.value.inner();
                        (const_bytes.memory.clone(), const_bytes.memory_map.clone())
                    };
                    let bytes = &const_memory;
                    let a_size = a_scalar.size(fx.dl).bytes() as usize;
                    let b_offset =
                        a_scalar.size(fx.dl).align_to(b_scalar.align(fx.dl).abi).bytes() as usize;
                    let b_size = b_scalar.size(fx.dl).bytes() as usize;

                    let a_raw = {
                        let mut buf = [0u8; 16];
                        let len = a_size.min(16);
                        buf[..len].copy_from_slice(&bytes[..len]);
                        u128::from_le_bytes(buf)
                    };
                    let b_raw = {
                        let mut buf = [0u8; 16];
                        let len = b_size.min(16);
                        buf[..len].copy_from_slice(&bytes[b_offset..b_offset + len]);
                        u128::from_le_bytes(buf)
                    };

                    // Create data sections for allocations in the memory_map,
                    // building a mapping of old addresses → GlobalValues.
                    let const_allocs = create_const_data_sections(fx, &const_memory_map);

                    let a_val = codegen_scalar_const(
                        fx,
                        &a_scalar,
                        a_raw,
                        &const_allocs,
                        Some(&const_memory_map),
                        Some(ty),
                    );
                    let b_val = codegen_scalar_const(
                        fx,
                        &b_scalar,
                        b_raw,
                        &const_allocs,
                        Some(&const_memory_map),
                        Some(ty),
                    );
                    CValue::by_val_pair(a_val, b_val, layout)
                }
                _ => {
                    // Memory-repr constant.
                    let (const_memory, const_memory_map) = {
                        let val = resolve_const_value(fx.db(), konst.as_ref());
                        let const_bytes = val.value.inner();
                        (const_bytes.memory.clone(), const_bytes.memory_map.clone())
                    };

                    // Fast path: no pointer-bearing side allocations in const-eval map.
                    if matches!(const_memory_map, hir_ty::MemoryMap::Empty) {
                        let mut data_desc = DataDescription::new();
                        data_desc.define(const_memory.clone());
                        data_desc.set_align(layout.align.abi.bytes());
                        let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                        fx.module.define_data(data_id, &data_desc).unwrap();
                        let local_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                        let ptr = fx.bcx.ins().symbol_value(fx.pointer_type, local_id);
                        return CValue::by_ref(pointer::Pointer::new(ptr), layout);
                    }

                    // Slow path: materialize bytes, then patch pointer-sized words that carry
                    // const-eval virtual addresses to relocation-based runtime pointers.
                    let mut data_desc = DataDescription::new();
                    data_desc.define(const_memory.clone());
                    data_desc.set_align(layout.align.abi.bytes());
                    let blob_data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                    fx.module.define_data(blob_data_id, &data_desc).unwrap();
                    let blob_local_id = fx.module.declare_data_in_func(blob_data_id, fx.bcx.func);
                    let blob_ptr = fx.bcx.ins().symbol_value(fx.pointer_type, blob_local_id);

                    let slot = CPlace::new_stack_slot(fx, layout.clone());
                    let dst = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
                    let byte_count =
                        fx.bcx.ins().iconst(fx.pointer_type, const_memory.len() as i64);
                    fx.bcx.call_memcpy(fx.module.target_config(), dst, blob_ptr, byte_count);

                    let const_allocs = create_const_data_sections(fx, &const_memory_map);
                    let ptr_size = fx.dl.pointer_size().bytes_usize();
                    let mut flags = MemFlags::new();
                    flags.set_notrap();
                    if const_memory.len() >= ptr_size {
                        for offset in (0..=const_memory.len() - ptr_size).step_by(ptr_size) {
                            let raw_addr = match ptr_size {
                                8 => {
                                    let mut buf = [0u8; 8];
                                    buf.copy_from_slice(&const_memory[offset..offset + 8]);
                                    u64::from_le_bytes(buf) as usize
                                }
                                4 => {
                                    let mut buf = [0u8; 4];
                                    buf.copy_from_slice(&const_memory[offset..offset + 4]);
                                    u32::from_le_bytes(buf) as usize
                                }
                                _ => unreachable!("unsupported pointer size"),
                            };

                            if !looks_like_const_eval_virtual_addr(raw_addr) {
                                continue;
                            }

                            if let Some(patched_ptr) =
                                const_ptr_from_data_alloc(fx, raw_addr, &const_allocs)
                            {
                                let field_ptr = slot.to_ptr().offset_i64(
                                    &mut fx.bcx,
                                    fx.pointer_type,
                                    offset as i64,
                                );
                                field_ptr.store(&mut fx.bcx, patched_ptr, flags);
                            }
                        }
                    }

                    CValue::by_ref(slot.to_ptr(), layout)
                }
            }
        }
        OperandKind::Copy(place) => {
            let val = codegen_place(fx, place).to_cvalue(fx);
            // r-a MIR can occasionally encode a by-value use of a non-`Copy`
            // place as `OperandKind::Copy`. Treat this as ownership transfer,
            // matching `Move` semantics for our per-local drop flags.
            if copy_operand_consumes_ownership(fx, place) {
                fx.clear_drop_flag(place.local);
            }
            val
        }
        OperandKind::Move(place) => {
            let val = codegen_place(fx, place).to_cvalue(fx);
            // Clear drop flag on any move from this local. Our drop flags are
            // tracked per-local (not per-field), so projected moves must also
            // make the local non-droppable to avoid dropping moved-out fields.
            fx.clear_drop_flag(place.local);
            val
        }
        OperandKind::Static(static_id) => codegen_static_operand(fx, *static_id),
    }
}

/// Create anonymous data sections for each allocation in a `MemoryMap`,
/// returning the allocation base/size + Cranelift GlobalValue.
#[derive(Clone, Copy)]
struct ConstDataAlloc {
    base: usize,
    len: usize,
    gv: cranelift_codegen::ir::GlobalValue,
}

fn create_const_data_sections(
    fx: &mut FunctionCx<'_, impl Module>,
    memory_map: &hir_ty::MemoryMap<'_>,
) -> Vec<ConstDataAlloc> {
    use hir_ty::MemoryMap;
    let mut allocs = Vec::new();
    match memory_map {
        MemoryMap::Empty => {}
        MemoryMap::Simple(data) => {
            let mut data_desc = DataDescription::new();
            data_desc.define(data.to_vec().into_boxed_slice());
            let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
            fx.module.define_data(data_id, &data_desc).unwrap();
            let gv = fx.module.declare_data_in_func(data_id, fx.bcx.func);
            allocs.push(ConstDataAlloc { base: 0, len: data.len(), gv });
        }
        MemoryMap::Complex(cm) => {
            for (addr, data) in cm.memory_iter() {
                let mut data_desc = DataDescription::new();
                data_desc.define(data.to_vec().into_boxed_slice());
                let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                fx.module.define_data(data_id, &data_desc).unwrap();
                let gv = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                allocs.push(ConstDataAlloc { base: *addr, len: data.len(), gv });
            }
        }
    }
    allocs
}

fn const_ptr_from_data_alloc(
    fx: &mut FunctionCx<'_, impl Module>,
    addr: usize,
    allocs: &[ConstDataAlloc],
) -> Option<Value> {
    for alloc in allocs {
        let Some(end) = alloc.base.checked_add(alloc.len) else {
            continue;
        };
        if addr < alloc.base || addr > end {
            continue;
        }

        let base_ptr = fx.bcx.ins().symbol_value(fx.pointer_type, alloc.gv);
        let offset = addr - alloc.base;
        if offset == 0 {
            return Some(base_ptr);
        }

        let offset_val = iconst_from_bits(&mut fx.bcx, fx.pointer_type, offset as u128);
        return Some(fx.bcx.ins().iadd(base_ptr, offset_val));
    }
    None
}

#[cfg(target_pointer_width = "64")]
fn looks_like_const_eval_virtual_addr(addr: usize) -> bool {
    const STACK_OFFSET: usize = 1 << 60;
    const HEAP_OFFSET: usize = 1 << 59;
    addr >= HEAP_OFFSET || addr >= STACK_OFFSET
}

#[cfg(target_pointer_width = "32")]
fn looks_like_const_eval_virtual_addr(addr: usize) -> bool {
    const STACK_OFFSET: usize = 1 << 30;
    const HEAP_OFFSET: usize = 1 << 29;
    addr >= HEAP_OFFSET || addr >= STACK_OFFSET
}

/// Lower a const-eval vtable-map id to a code pointer when the mapped type
/// is callable (`FnDef` or closure).
fn codegen_const_callable_from_vtable_id(
    fx: &mut FunctionCx<'_, impl Module>,
    vtable_id: usize,
    memory_map: &hir_ty::MemoryMap<'_>,
) -> Option<Value> {
    let mapped_ty = memory_map.vtable_ty(vtable_id).ok()?;
    match mapped_ty.kind() {
        TyKind::FnDef(def, generic_args) => {
            let CallableDefId::FunctionId(mut callee_func_id) = def.0 else {
                return None;
            };
            let mut callee_args = generic_args;

            if let ItemContainerId::TraitId(_) = callee_func_id.loc(fx.db()).container {
                let interner = DbInterner::new_no_crate(fx.db());
                if hir_ty::method_resolution::is_dyn_method(
                    interner,
                    fx.env().param_env(),
                    callee_func_id,
                    callee_args,
                )
                .is_none()
                {
                    match fx.db().lookup_impl_method(fx.env().as_ref(), callee_func_id, callee_args)
                    {
                        (Either::Left(resolved_id), resolved_args) => {
                            callee_func_id = resolved_id;
                            callee_args = resolved_args;
                        }
                        (Either::Right(_), _) => return None,
                    }
                }
            }

            let is_extern =
                matches!(callee_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
            let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();
            let interner = DbInterner::new_no_crate(fx.db());
            let empty_args = GenericArgs::empty(interner);

            let (callee_sig, callee_name) = if is_extern {
                let sig = build_fn_sig_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    callee_func_id,
                    empty_args,
                )
                .expect("extern fn sig for const fn ptr");
                let name = extern_fn_symbol_name(fx.db(), callee_func_id);
                (sig, name)
            } else if is_cross_crate {
                let sig = build_fn_sig_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    callee_func_id,
                    callee_args,
                )
                .expect("cross-crate fn sig for const fn ptr");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    callee_func_id,
                    callee_args,
                    fx.ext_crate_disambiguators(),
                );
                (sig, name)
            } else {
                let callee_body = fx
                    .db()
                    .monomorphized_mir_body(
                        callee_func_id.into(),
                        callee_args.store(),
                        fx.env().clone(),
                    )
                    .expect("failed to get local callee MIR for const fn ptr");
                let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body)
                    .expect("const fn ptr sig");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    callee_func_id,
                    callee_args,
                    fx.ext_crate_disambiguators(),
                );
                (sig, name)
            };

            let callee_id = fx
                .module
                .declare_function(&callee_name, Linkage::Import, &callee_sig)
                .expect("declare callee for const fn ptr");
            let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
            Some(fx.bcx.ins().func_addr(fx.pointer_type, callee_ref))
        }
        TyKind::Closure(closure_id, closure_subst) => {
            let (closure_sig, closure_name) = match fx.db().monomorphized_mir_body_for_closure(
                closure_id.0,
                closure_subst.store(),
                fx.env().clone(),
            ) {
                Ok(closure_body) => {
                    let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body)
                        .expect("closure sig for const fn ptr");
                    let name = symbol_mangling::mangle_closure(
                        fx.db(),
                        closure_id.0,
                        closure_subst,
                        fx.ext_crate_disambiguators(),
                    );
                    (sig, name)
                }
                Err(hir_ty::mir::MirLowerError::UnresolvedName(name)) => {
                    let fn_ptr_ty = closure_subst.split_closure_args().closure_sig_as_fn_ptr_ty;
                    let sig =
                        build_fn_ptr_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), fn_ptr_ty)?;
                    (sig, name)
                }
                Err(_) => return None,
            };
            let closure_func_id = fx
                .module
                .declare_function(&closure_name, Linkage::Import, &closure_sig)
                .expect("declare closure for const fn ptr");
            let closure_ref = fx.module.declare_func_in_func(closure_func_id, fx.bcx.func);
            Some(fx.bcx.ins().func_addr(fx.pointer_type, closure_ref))
        }
        _ => None,
    }
}

fn build_fn_ptr_sig_from_ty(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    fn_ptr_ty: hir_ty::next_solver::Ty<'_>,
) -> Option<Signature> {
    let TyKind::FnPtr(sig_tys, header) = fn_ptr_ty.kind() else {
        return None;
    };

    let fn_abi = abi::fn_abi_for_fn_ptr(
        isa,
        db,
        dl,
        env,
        &sig_tys,
        matches!(header.abi, hir_ty::FnAbi::RustCall),
        header.c_variadic,
    )
    .ok()?;

    Some(fn_abi.sig)
}

/// Emit a Cranelift Value for a single scalar constant.
/// If the scalar is a pointer and the address falls within a const-data
/// allocation, emits `symbol_value + offset`. Otherwise, emits an `iconst`
/// (or float const).
fn codegen_scalar_const(
    fx: &mut FunctionCx<'_, impl Module>,
    scalar: &Scalar,
    raw: u128,
    const_allocs: &[ConstDataAlloc],
    memory_map: Option<&hir_ty::MemoryMap<'_>>,
    _const_ty: Option<&StoredTy>,
) -> Value {
    match scalar.primitive() {
        Primitive::Pointer(_) => {
            let addr = raw as usize;
            if let Some(ptr) = const_ptr_from_data_alloc(fx, addr, const_allocs) {
                ptr
            } else if addr == 0 {
                iconst_from_bits(&mut fx.bcx, fx.pointer_type, 0)
            } else if let Some(ptr) =
                memory_map.and_then(|mm| codegen_const_callable_from_vtable_id(fx, addr, mm))
            {
                ptr
            } else {
                iconst_from_bits(&mut fx.bcx, fx.pointer_type, raw)
            }
        }
        Primitive::Float(rustc_abi::Float::F32) => {
            fx.bcx.ins().f32const(f32::from_bits(raw as u32))
        }
        Primitive::Float(rustc_abi::Float::F64) => {
            fx.bcx.ins().f64const(f64::from_bits(raw as u64))
        }
        _ => {
            let clif_ty = scalar_to_clif_type(fx.dl, scalar);
            iconst_from_bits(&mut fx.bcx, clif_ty, raw)
        }
    }
}

fn call_result_unsize_metadata(
    fx: &mut FunctionCx<'_, impl Module>,
    source_return_ty: &StoredTy,
    destination_place: &Place,
) -> Value {
    let Some(src_pointee) = source_return_ty.as_ref().builtin_deref(true) else {
        panic!("cannot unsize non-pointer call return: {:?}", source_return_ty);
    };
    let dest_ty = place_ty(fx.db(), fx.ra_body(), destination_place);
    let Some(dest_pointee) = dest_ty.as_ref().builtin_deref(true) else {
        panic!("call destination for unsize return is not a pointer: {:?}", dest_ty);
    };

    unsize_metadata_for_pointees(fx, src_pointee.store(), dest_pointee.store(), None)
}

/// Prepare a call result `CValue` for the destination place.
///
/// This happens at call lowering (producer side), not in `CPlace::write_cvalue`.
fn prepare_call_result_cvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    call: cranelift_codegen::ir::Inst,
    ret_abi: &abi::ArgAbi,
    dest: &CPlace,
    source_return_ty: Option<&StoredTy>,
    destination_place: &Place,
) -> Option<CValue> {
    let results = fx.bcx.inst_results(call).to_vec();
    match ret_abi.mode {
        PassMode::Ignore | PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => None,
        PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
            unreachable!("unsized return ABI is unsupported")
        }
        PassMode::Direct(_) => match results.as_slice() {
            [value] => {
                if matches!(dest.layout.backend_repr, BackendRepr::ScalarPair(_, _)) {
                    let Some(source_return_ty) = source_return_ty else {
                        panic!(
                            "scalar-pair call destination expects metadata source for direct return"
                        );
                    };
                    let metadata =
                        call_result_unsize_metadata(fx, source_return_ty, destination_place);
                    Some(CValue::by_val_pair(*value, metadata, dest.layout.clone()))
                } else {
                    Some(CValue::by_val(*value, dest.layout.clone()))
                }
            }
            _ => {
                panic!("direct return ABI expects 1 return value, got {}", results.len())
            }
        },
        PassMode::Pair(_, _) => match results.as_slice() {
            [a, b] => Some(CValue::by_val_pair(*a, *b, dest.layout.clone())),
            _ => panic!("pair return ABI expects 2 return values, got {}", results.len()),
        },
        PassMode::Cast { ref cast, .. } => {
            let ret_layout =
                ret_abi.layout.as_ref().expect("Cast return ABI must carry a layout").clone();
            Some(abi::pass_mode::from_casted_value(fx, &results, ret_layout, cast))
        }
    }
}

/// Store a prepared call return value into `dest` and jump to the continuation block
/// (or trap for diverging calls). Shared by all call codegen paths.
fn store_call_result_and_jump(
    fx: &mut FunctionCx<'_, impl Module>,
    sret_slot: Option<CPlace>,
    dest: CPlace,
    call_result: Option<CValue>,
    destination_place: &Place,
    target: &Option<BasicBlockId>,
) {
    if let Some(sret_slot) = sret_slot {
        let cval = sret_slot.to_cvalue(fx);
        dest.write_cvalue(fx, cval);
    } else if let Some(cval) = call_result {
        dest.write_cvalue(fx, cval);
    } else {
        match dest.layout.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _) if !dest.layout.is_zst() => {
                panic!("missing prepared call result for non-ZST scalar destination")
            }
            _ => {
                // ZST and memory destinations have no direct call SSA result to write here.
            }
        }
    }

    if target.is_some()
        && destination_place.projection.lookup(&fx.ra_body().projection_store).is_empty()
    {
        fx.set_drop_flag(destination_place.local);
    }

    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    } else {
        fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
    }
}

#[derive(Clone)]
struct CallArgument {
    cval: CValue,
    is_owned: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CallOperandLowering {
    Standard,
    RustCall,
}

impl CallOperandLowering {
    fn from_rust_call_abi(is_rust_call_abi: bool) -> Self {
        if is_rust_call_abi { Self::RustCall } else { Self::Standard }
    }
}

fn codegen_call_argument_operand(
    fx: &mut FunctionCx<'_, impl Module>,
    operand: &Operand,
) -> CallArgument {
    let is_owned = operand_consumes_ownership(fx, &operand.kind);
    CallArgument {
        cval: codegen_operand(fx, &operand.kind),
        is_owned,
    }
}

fn lower_call_operands_for_abi(
    fx: &mut FunctionCx<'_, impl Module>,
    args: &[Operand],
) -> Vec<CallArgument> {
    args.iter().map(|arg| codegen_call_argument_operand(fx, arg)).collect()
}

fn lower_rust_call_tuple_operands_for_abi(
    fx: &mut FunctionCx<'_, impl Module>,
    args: &[Operand],
) -> Vec<CallArgument> {
    let (tuple_operand, prefix_operands) =
        args.split_last().expect("rust-call tuple lowering requires at least one argument");

    let tuple_ty = operand_ty(fx.db(), fx.ra_body(), &tuple_operand.kind);
    let TyKind::Tuple(tuple_fields) = tuple_ty.as_ref().kind() else {
        panic!("rust-call tuple lowering expected final tuple argument, got {tuple_ty:?}");
    };

    let mut lowered = prefix_operands
        .iter()
        .map(|operand| codegen_call_argument_operand(fx, operand))
        .collect::<Vec<_>>();

    let tuple_arg = codegen_call_argument_operand(fx, tuple_operand);

    let tuple_layout = tuple_arg.cval.layout.clone();
    let tuple_ptr = tuple_arg.cval.clone().force_stack(fx);
    let tuple_place = CPlace::for_ptr(tuple_ptr, tuple_layout);

    lowered.reserve(tuple_fields.as_slice().len());
    for (idx, field_ty) in tuple_fields.iter().enumerate() {
        let field_layout = fx
            .db()
            .layout_of_ty(field_ty.store(), fx.env().clone())
            .expect("rust-call tuple field layout");
        let cval = if field_layout.is_zst() {
            CValue::zst(field_layout)
        } else {
            tuple_place.place_field(fx, idx, field_layout).to_cvalue(fx)
        };
        lowered.push(CallArgument { cval, is_owned: tuple_arg.is_owned });
    }

    lowered
}

fn lower_rust_call_operands_for_abi(
    fx: &mut FunctionCx<'_, impl Module>,
    args: &[Operand],
    expected_arg_count: usize,
) -> Vec<CallArgument> {
    if !matches!(args, [_] | [_, _]) {
        panic!(
            "rust-call ABI requires one or two operands (self + tuple), got {}",
            args.len(),
        );
    }

    let lowered = lower_rust_call_tuple_operands_for_abi(fx, args);

    assert_eq!(
        lowered.len(),
        expected_arg_count,
        "rust-call argument count mismatch after tuple lowering: lowered={} expected={} args_len={}",
        lowered.len(),
        expected_arg_count,
        args.len(),
    );
    lowered
}

fn lower_call_operands_with_lowering(
    fx: &mut FunctionCx<'_, impl Module>,
    args: &[Operand],
    arg_abis: &[abi::ArgAbi],
    operand_lowering: CallOperandLowering,
) -> Vec<CallArgument> {
    match operand_lowering {
        CallOperandLowering::Standard => lower_call_operands_for_abi(fx, args),
        CallOperandLowering::RustCall => {
            lower_rust_call_operands_for_abi(fx, args, arg_abis.len())
        }
    }
}

fn append_lowered_call_args(
    fx: &mut FunctionCx<'_, impl Module>,
    call_args: &mut Vec<Value>,
    lowered_args: &[CallArgument],
    arg_abis: &[abi::ArgAbi],
    c_variadic: bool,
) {
    if c_variadic {
        assert!(
            lowered_args.len() >= arg_abis.len(),
            "c-variadic call has fewer lowered args than fixed ABI args: lowered={} fixed={}",
            lowered_args.len(),
            arg_abis.len(),
        );
    } else {
        assert_eq!(
            lowered_args.len(),
            arg_abis.len(),
            "call argument count mismatch after ABI lowering: lowered={} abi={}",
            lowered_args.len(),
            arg_abis.len(),
        );
    }

    for (arg, arg_abi) in lowered_args.iter().zip(arg_abis) {
        call_args.extend(abi::pass_mode::adjust_arg_for_abi(
            fx,
            arg.cval.clone(),
            arg_abi,
            arg.is_owned,
        ));
    }

    // Extra c-variadic arguments are lowered without fixed ArgAbi entries and
    // follow C default argument promotions.
    if c_variadic {
        for arg in &lowered_args[arg_abis.len()..] {
            call_args.push(lower_c_variadic_extra_arg(fx, arg));
        }
    }
}

fn lower_c_variadic_extra_arg(fx: &mut FunctionCx<'_, impl Module>, arg: &CallArgument) -> Value {
    let BackendRepr::Scalar(scalar) = arg.cval.layout.backend_repr else {
        panic!(
            "unsupported c-variadic extra argument layout: {:?}",
            arg.cval.layout.backend_repr,
        );
    };

    let mut value = arg.cval.clone().load_scalar(fx);
    let value_ty = fx.bcx.func.dfg.value_type(value);
    let is_signed_small_int = matches!(scalar.primitive(), Primitive::Int(_, true));

    // C default argument promotions for varargs.
    value = match value_ty {
        types::F32 => fx.bcx.ins().fpromote(types::F64, value),
        types::I8 | types::I16 => {
            if is_signed_small_int {
                fx.bcx.ins().sextend(types::I32, value)
            } else {
                fx.bcx.ins().uextend(types::I32, value)
            }
        }
        _ => value,
    };

    value
}

fn adjust_c_variadic_signature_and_args(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_name: &str,
    sig_ref: cranelift_codegen::ir::SigRef,
    fixed_param_count: usize,
    call_args: &mut Vec<Value>,
) {
    assert!(
        call_args.len() >= fixed_param_count,
        "c-variadic call passed fewer args than fixed params for {callee_name}: fixed={} args={}",
        fixed_param_count,
        call_args.len(),
    );

    let mut sig_params = fx.bcx.func.dfg.signatures[sig_ref].params.clone();
    if sig_params.len() < call_args.len() {
        for arg in &call_args[sig_params.len()..] {
            sig_params.push(AbiParam::new(fx.bcx.func.dfg.value_type(*arg)));
        }
    } else if sig_params.len() > call_args.len() {
        let pad_param_tys: Vec<_> =
            sig_params[call_args.len()..].iter().map(|param| param.value_type).collect();
        for param_ty in pad_param_tys {
            assert!(
                param_ty.is_int(),
                "unsupported c-variadic pad type for {callee_name}: {:?}",
                param_ty,
            );
            call_args.push(fx.bcx.ins().iconst(param_ty, 0));
        }
    }

    for (arg, param) in call_args.iter().zip(sig_params.iter()) {
        let arg_ty = fx.bcx.func.dfg.value_type(*arg);
        assert_eq!(
            arg_ty, param.value_type,
            "c-variadic arg type mismatch for {callee_name}: arg={:?} param={:?}",
            arg_ty, param.value_type
        );
    }

    fx.bcx.func.dfg.signatures[sig_ref].params = sig_params;
}

// ---------------------------------------------------------------------------
// Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_terminator(fx: &mut FunctionCx<'_, impl Module>, term: &TerminatorKind) {
    match term {
        TerminatorKind::Return => {
            let ret_place = fx.local_place(hir_ty::mir::return_slot()).clone();
            codegen_return(fx, &ret_place);
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
fn codegen_drop(fx: &mut FunctionCx<'_, impl Module>, place: &Place, target: BasicBlockId) {
    let target_block = fx.clif_block(target);
    let body = fx.ra_body();
    let ty = place_ty(fx.db(), body, place);

    let interner = DbInterner::new_with(fx.db(), fx.local_crate());
    if !hir_ty::drop::has_drop_glue_mono(interner, ty.as_ref()) {
        fx.bcx.ins().jump(target_block, &[]);
        return;
    }

    // Check drop flag: skip if the local has been moved out.
    // Only simple locals (no projections) have drop flags.
    let projections = place.projection.lookup(&fx.ra_body().projection_store);
    let mut clear_flag_after_drop = false;
    if projections.is_empty() {
        let local_idx = place.local.into_raw().into_u32();
        if let Some(&flag_var) = fx.drop_flags.get(&local_idx) {
            clear_flag_after_drop = true;
            let flag_val = fx.bcx.use_var(flag_var);
            let do_drop_block = fx.bcx.create_block();
            fx.bcx.ins().brif(flag_val, do_drop_block, &[], target_block, &[]);
            fx.bcx.switch_to_block(do_drop_block);
        }
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
    let direct_drop = lang_items
        .Drop
        .and_then(|drop_trait| resolve_drop_impl(fx.db(), fx.local_crate(), drop_trait, &ty));
    let needs_field_drops = type_has_droppable_fields(fx.db(), fx.local_crate(), &ty);

    let fn_name = if let (Some(drop_func_id), false) = (direct_drop, needs_field_drops) {
        // Simple case: just call Drop::drop directly
        let generic_args = drop_impl_generic_args(fx.db(), fx.local_crate(), &ty);
        symbol_mangling::mangle_function(
            fx.db(),
            drop_func_id,
            generic_args.as_ref(),
            fx.ext_crate_disambiguators(),
        )
    } else {
        // Needs recursive field drops — use drop_in_place glue
        symbol_mangling::mangle_drop_in_place(fx.db(), ty.as_ref(), fx.ext_crate_disambiguators())
    };

    let callee_id =
        fx.module.declare_function(&fn_name, Linkage::Import, &drop_sig).expect("declare drop fn");
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
    fx.bcx.ins().call(callee_ref, &[ptr]);

    if clear_flag_after_drop {
        fx.clear_drop_flag(place.local);
    }

    fx.bcx.ins().jump(target_block, &[]);
}

/// Check whether a type has fields that themselves need dropping.
/// Returns true if any field (struct fields, tuple elements, enum variant
/// fields, closure captures) has drop glue, requiring a `drop_in_place`
/// wrapper rather than a simple `Drop::drop` call.
fn type_has_droppable_fields(db: &dyn HirDatabase, krate: base_db::Crate, ty: &StoredTy) -> bool {
    let interner = DbInterner::new_with(db, krate);
    match ty.as_ref().kind() {
        TyKind::Adt(adt_def, subst) => {
            let adt_id = adt_def.inner().id;
            match adt_id {
                hir_def::AdtId::StructId(id) => {
                    use hir_def::signatures::StructFlags;
                    if db
                        .struct_signature(id)
                        .flags
                        .intersects(StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA)
                    {
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
            captures
                .iter()
                .any(|capture| hir_ty::drop::has_drop_glue_mono(interner, capture.ty(db, subst)))
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

/// Build generic args for calling a resolved `Drop::drop` impl method.
///
/// `TyKind::Adt` substitutions include regions, but impl-method
/// monomorphization expects type/const entries only.
fn drop_impl_generic_args(
    db: &dyn HirDatabase,
    local_crate: base_db::Crate,
    ty: &StoredTy,
) -> StoredGenericArgs {
    let interner = DbInterner::new_with(db, local_crate);
    match ty.as_ref().kind() {
        TyKind::Adt(_, subst) => {
            GenericArgs::new_from_iter(interner, subst.iter().filter(|arg| arg.region().is_none()))
                .store()
        }
        _ => GenericArgs::empty(interner).store(),
    }
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
                        CallOperandLowering::Standard,
                        args,
                        destination,
                        target,
                    );
                }
                CallableDefId::StructId(struct_id) => {
                    codegen_adt_constructor_call(
                        fx,
                        VariantId::StructId(struct_id),
                        generic_args,
                        args,
                        destination,
                        target,
                    );
                }
                CallableDefId::EnumVariantId(variant_id) => {
                    codegen_adt_constructor_call(
                        fx,
                        VariantId::EnumVariantId(variant_id),
                        generic_args,
                        args,
                        destination,
                        target,
                    );
                }
            }
        }
        TyKind::FnPtr(sig_tys, header) => {
            let fn_ptr_is_rust_call_abi = matches!(header.abi, hir_ty::FnAbi::RustCall);
            codegen_fn_ptr_call(
                fx,
                func,
                &sig_tys,
                fn_ptr_is_rust_call_abi,
                CallOperandLowering::from_rust_call_abi(fn_ptr_is_rust_call_abi),
                header.c_variadic,
                args,
                destination,
                target,
            );
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
    let dest = codegen_place(fx, destination);
    if !dest.layout.is_zst() {
        let dest_ty = {
            let body = fx.ra_body();
            place_ty(fx.db(), body, destination)
        };
        let variant_idx = variant_id_to_idx(fx.db(), variant_id);
        let field_vals: Vec<_> = args.iter().map(|op| codegen_operand(fx, &op.kind)).collect();
        codegen_adt_fields(fx, variant_idx, &field_vals, &dest_ty, dest);
    }

    if destination.projection.lookup(&fx.ra_body().projection_store).is_empty() {
        fx.set_drop_flag(destination.local);
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
    fn_ptr_is_rust_call_abi: bool,
    operand_lowering: CallOperandLowering,
    c_variadic: bool,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Load the fn pointer value
    let fn_ptr_cval = codegen_operand(fx, &func_operand.kind);
    let fn_ptr = fn_ptr_cval.load_scalar(fx);

    let sig_tys_inner = sig_tys.clone().skip_binder();
    let source_return_ty = {
        let output = sig_tys_inner.output();
        (!output.is_never()).then(|| output.store())
    };
    let callee_abi = abi::fn_abi_for_fn_ptr(
        fx.isa,
        fx.db(),
        fx.dl,
        fx.env(),
        sig_tys,
        fn_ptr_is_rust_call_abi,
        c_variadic,
    )
    .expect("fn pointer ABI");

    let dest = codegen_place(fx, destination);
    let is_sret_return = matches!(
        callee_abi.ret.mode,
        PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
    );

    let sig_ref = fx.bcx.import_signature(callee_abi.sig.clone());

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

    let lowered_args =
        lower_call_operands_with_lowering(fx, args, &callee_abi.args, operand_lowering);
    append_lowered_call_args(
        fx,
        &mut call_args,
        &lowered_args,
        &callee_abi.args,
        callee_abi.c_variadic,
    );

    if callee_abi.c_variadic {
        adjust_c_variadic_signature_and_args(
            fx,
            "fn_ptr_call",
            sig_ref,
            callee_abi.sig.params.len(),
            &mut call_args,
        );
    }

    // Emit indirect call
    let call = fx.bcx.ins().call_indirect(sig_ref, fn_ptr, &call_args);
    let call_result = prepare_call_result_cvalue(
        fx,
        call,
        &callee_abi.ret,
        &dest,
        source_return_ty.as_ref(),
        destination,
    );
    store_call_result_and_jump(fx, sret_slot, dest, call_result, destination, target);
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
    operand_lowering: CallOperandLowering,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Get closure MIR body to build the signature
    let closure_body = fx
        .db()
        .monomorphized_mir_body_for_closure(closure_id, closure_subst.clone(), fx.env().clone())
        .expect("closure MIR");
    let closure_ret_ty = closure_body.locals[hir_ty::mir::return_slot()].ty.clone();
    let callee_abi =
        abi::fn_abi_for_body(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body).expect("closure ABI");

    // Generate mangled name
    let closure_name = symbol_mangling::mangle_closure(
        fx.db(),
        closure_id,
        closure_subst.as_ref(),
        fx.ext_crate_disambiguators(),
    );

    // Declare in module (Import linkage — defined elsewhere in same module)
    let callee_id = fx
        .module
        .declare_function(&closure_name, Linkage::Import, &callee_abi.sig)
        .expect("declare closure");
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    let lowered_args =
        lower_call_operands_with_lowering(fx, args, &callee_abi.args, operand_lowering);

    assert!(!lowered_args.is_empty(), "closure call is missing receiver argument");

    // Destination
    let dest = codegen_place(fx, destination);
    let is_sret_return = matches!(
        callee_abi.ret.mode,
        PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
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

    append_lowered_call_args(fx, &mut call_args, &lowered_args, &callee_abi.args, false);

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);
    let call_result = prepare_call_result_cvalue(
        fx,
        call,
        &callee_abi.ret,
        &dest,
        Some(&closure_ret_ty),
        destination,
    );
    store_call_result_and_jump(fx, sret_slot, dest, call_result, destination, target);
}

fn method_output_ty(
    db: &dyn HirDatabase,
    func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
) -> StoredTy {
    callable_output_ty(db, func_id, generic_args)
        .expect("derive pseudo-method unexpectedly returned never")
}

fn callable_output_ty(
    db: &dyn HirDatabase,
    func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
) -> Option<StoredTy> {
    let interner = DbInterner::new_no_crate(db);
    let fn_sig = if generic_args.is_empty() {
        db.callable_item_signature(func_id.into()).skip_binder().skip_binder()
    } else {
        db.callable_item_signature(func_id.into()).instantiate(interner, generic_args).skip_binder()
    };
    let output = *fn_sig.inputs_and_output.as_slice().split_last().unwrap().0;
    (!output.is_never()).then(|| output.store())
}

fn trait_method_substs_for_derive_call<'db>(
    db: &'db dyn HirDatabase,
    local_crate: base_db::Crate,
    trait_method_func_id: hir_def::FunctionId,
    self_ty: &StoredTy,
    include_rhs_self: bool,
) -> GenericArgs<'db> {
    let interner = DbInterner::new_with(db, local_crate);
    let identity = GenericArgs::identity_for_item(interner, trait_method_func_id.into());
    assert!(!identity.is_empty(), "trait method has no generic args: {trait_method_func_id:?}");

    let mut args: Vec<_> = identity.iter().collect();
    args[0] = self_ty.as_ref().into();
    if include_rhs_self && args.len() > 1 {
        args[1] = self_ty.as_ref().into();
    }
    GenericArgs::new_from_iter(interner, args)
}

fn shared_ref_layout_for_pointee(
    fx: &mut FunctionCx<'_, impl Module>,
    pointee_ty: &StoredTy,
) -> LayoutArc {
    let interner = DbInterner::new_no_crate(fx.db());
    let ref_ty = hir_ty::next_solver::Ty::new_ref(
        interner,
        hir_ty::next_solver::Region::new_static(interner),
        pointee_ty.as_ref(),
        hir_ty::next_solver::Mutability::Not,
    )
    .store();
    fx.db().layout_of_ty(ref_ty, fx.env().clone()).expect("shared reference layout")
}

fn bool_layout_ty(dl: &TargetDataLayout, bool_layout: &LayoutArc) -> Type {
    let BackendRepr::Scalar(bool_scalar) = bool_layout.backend_repr else {
        panic!("expected scalar bool layout, got {:?}", bool_layout.backend_repr);
    };
    scalar_to_clif_type(dl, &bool_scalar)
}

fn bool_cvalue_to_bool_scalar(
    fx: &mut FunctionCx<'_, impl Module>,
    bool_cvalue: CValue,
    bool_ty: Type,
) -> Value {
    let bool_val = bool_cvalue.load_scalar(fx);
    if fx.bcx.func.dfg.value_type(bool_val) == bool_ty {
        bool_val
    } else {
        let is_true = codegen_icmp_imm(fx, IntCC::NotEqual, bool_val, 0);
        let one = fx.bcx.ins().iconst(bool_ty, 1);
        let zero = fx.bcx.ins().iconst(bool_ty, 0);
        fx.bcx.ins().select(is_true, one, zero)
    }
}

fn enum_discriminant_layout(
    fx: &mut FunctionCx<'_, impl Module>,
    enum_layout: &LayoutArc,
) -> LayoutArc {
    use rustc_abi::Variants;
    match &enum_layout.variants {
        Variants::Multiple { tag, .. } => tag_scalar_layout(fx.dl, tag),
        Variants::Single { .. } => {
            panic!("single-variant enum has no runtime discriminant")
        }
        Variants::Empty => panic!("empty enum has no runtime discriminant"),
    }
}

fn enum_variant_idx_by_name(
    db: &dyn HirDatabase,
    enum_id: hir_def::EnumId,
    wanted: &str,
) -> VariantIdx {
    for (idx, (_, name, _)) in enum_id.enum_variants(db).variants.iter().enumerate() {
        if name.as_str() == wanted {
            return VariantIdx::from_u32(idx as u32);
        }
    }
    panic!("enum variant `{wanted}` not found for enum {enum_id:?}")
}

fn enum_default_variant_idx(db: &dyn HirDatabase, enum_id: hir_def::EnumId) -> VariantIdx {
    let variants = &enum_id.enum_variants(db).variants;
    let mut default_variant_idx = None;

    for (idx, (variant_id, _, _)) in variants.iter().enumerate() {
        let attrs = AttrFlags::query(db, (*variant_id).into());
        if attrs.contains(AttrFlags::HAS_DEFAULT_ATTR) {
            assert!(
                default_variant_idx.is_none(),
                "builtin Default::default found multiple #[default] variants for enum {enum_id:?}",
            );
            default_variant_idx = Some(VariantIdx::from_u32(idx as u32));
        }
    }

    default_variant_idx.unwrap_or_else(|| {
        panic!("builtin Default::default found no #[default] variant for enum {enum_id:?}")
    })
}

#[derive(Clone)]
struct OptionOrderingInfo {
    option_ty: StoredTy,
    option_layout: LayoutArc,
    option_none_idx: VariantIdx,
    option_some_idx: VariantIdx,
    option_none_discr: u128,
    ordering_ty: StoredTy,
    ordering_layout: LayoutArc,
    ordering_less_idx: VariantIdx,
    ordering_equal_idx: VariantIdx,
    ordering_greater_idx: VariantIdx,
    ordering_less_discr: u128,
    ordering_equal_discr: u128,
}

fn option_ordering_info(
    fx: &mut FunctionCx<'_, impl Module>,
    option_ty: StoredTy,
    option_layout: LayoutArc,
) -> OptionOrderingInfo {
    let TyKind::Adt(option_adt, option_args) = option_ty.as_ref().kind() else {
        panic!("partial_cmp must return Option<Ordering>, got {option_ty:?}");
    };
    let hir_def::AdtId::EnumId(option_enum_id) = option_adt.inner().id else {
        panic!("partial_cmp must return enum Option<Ordering>, got {option_ty:?}");
    };

    let option_none_idx = enum_variant_idx_by_name(fx.db(), option_enum_id, "None");
    let option_some_idx = enum_variant_idx_by_name(fx.db(), option_enum_id, "Some");
    let option_none_discr = enum_variant_discriminant_bits(fx.db(), &option_ty, option_none_idx);

    let ordering_ty = option_args.type_at(0).store();
    let ordering_layout =
        fx.db().layout_of_ty(ordering_ty.clone(), fx.env().clone()).expect("Ordering layout");
    let TyKind::Adt(ordering_adt, _) = ordering_ty.as_ref().kind() else {
        panic!("partial_cmp inner type must be Ordering enum, got {ordering_ty:?}");
    };
    let hir_def::AdtId::EnumId(ordering_enum_id) = ordering_adt.inner().id else {
        panic!("partial_cmp inner type must be Ordering enum, got {ordering_ty:?}");
    };

    let ordering_less_idx = enum_variant_idx_by_name(fx.db(), ordering_enum_id, "Less");
    let ordering_equal_idx = enum_variant_idx_by_name(fx.db(), ordering_enum_id, "Equal");
    let ordering_greater_idx = enum_variant_idx_by_name(fx.db(), ordering_enum_id, "Greater");
    let ordering_less_discr =
        enum_variant_discriminant_bits(fx.db(), &ordering_ty, ordering_less_idx);
    let ordering_equal_discr =
        enum_variant_discriminant_bits(fx.db(), &ordering_ty, ordering_equal_idx);

    OptionOrderingInfo {
        option_ty,
        option_layout,
        option_none_idx,
        option_some_idx,
        option_none_discr,
        ordering_ty,
        ordering_layout,
        ordering_less_idx,
        ordering_equal_idx,
        ordering_greater_idx,
        ordering_less_discr,
        ordering_equal_discr,
    }
}

fn write_option_ordering_variant(
    fx: &mut FunctionCx<'_, impl Module>,
    info: &OptionOrderingInfo,
    dest: &CPlace,
    ordering_variant: Option<VariantIdx>,
) {
    match ordering_variant {
        None => codegen_adt_fields(fx, info.option_none_idx, &[], &info.option_ty, dest.clone()),
        Some(ordering_variant) => {
            let ordering_place = CPlace::new_stack_slot(fx, info.ordering_layout.clone());
            codegen_adt_fields(
                fx,
                ordering_variant,
                &[],
                &info.ordering_ty,
                ordering_place.clone(),
            );
            let ordering_val = ordering_place.to_cvalue(fx);
            codegen_adt_fields(
                fx,
                info.option_some_idx,
                &[ordering_val],
                &info.option_ty,
                dest.clone(),
            );
        }
    }
}

fn write_option_ordering_state(
    fx: &mut FunctionCx<'_, impl Module>,
    info: &OptionOrderingInfo,
    dest: &CPlace,
    state: Value,
) {
    // 0 = None, 1 = Equal, 2 = Less, 3 = Greater
    let none_block = fx.bcx.create_block();
    let equal_block = fx.bcx.create_block();
    let less_block = fx.bcx.create_block();
    let greater_block = fx.bcx.create_block();
    let done_block = fx.bcx.create_block();

    let is_none = fx.bcx.ins().icmp_imm(IntCC::Equal, state, 0);
    let non_none = fx.bcx.create_block();
    fx.bcx.ins().brif(is_none, none_block, &[], non_none, &[]);

    fx.bcx.switch_to_block(non_none);
    let is_equal = fx.bcx.ins().icmp_imm(IntCC::Equal, state, 1);
    let non_equal = fx.bcx.create_block();
    fx.bcx.ins().brif(is_equal, equal_block, &[], non_equal, &[]);

    fx.bcx.switch_to_block(non_equal);
    let is_less = fx.bcx.ins().icmp_imm(IntCC::Equal, state, 2);
    fx.bcx.ins().brif(is_less, less_block, &[], greater_block, &[]);

    fx.bcx.switch_to_block(none_block);
    write_option_ordering_variant(fx, info, dest, None);
    fx.bcx.ins().jump(done_block, &[]);

    fx.bcx.switch_to_block(equal_block);
    write_option_ordering_variant(fx, info, dest, Some(info.ordering_equal_idx));
    fx.bcx.ins().jump(done_block, &[]);

    fx.bcx.switch_to_block(less_block);
    write_option_ordering_variant(fx, info, dest, Some(info.ordering_less_idx));
    fx.bcx.ins().jump(done_block, &[]);

    fx.bcx.switch_to_block(greater_block);
    write_option_ordering_variant(fx, info, dest, Some(info.ordering_greater_idx));
    fx.bcx.ins().jump(done_block, &[]);

    fx.bcx.switch_to_block(done_block);
}

fn decode_option_ordering_state(
    fx: &mut FunctionCx<'_, impl Module>,
    info: &OptionOrderingInfo,
    value: CValue,
) -> Value {
    // 0 = None, 1 = Equal, 2 = Less, 3 = Greater
    let option_place = CPlace::for_ptr(value.force_stack(fx), info.option_layout.clone());
    let option_discr_layout = enum_discriminant_layout(fx, &info.option_layout);
    let option_discr =
        codegen_get_discriminant(fx, &option_place, &info.option_ty, &option_discr_layout);
    let is_none = codegen_icmp_imm(fx, IntCC::Equal, option_discr, info.option_none_discr as i128);

    let some_layout = variant_layout(&info.option_layout, info.option_some_idx);
    let some_place = option_place.downcast_variant(some_layout);
    let ordering_field = some_place.place_field(fx, 0, info.ordering_layout.clone());
    let ordering_discr_layout = enum_discriminant_layout(fx, &info.ordering_layout);
    let ordering_discr =
        codegen_get_discriminant(fx, &ordering_field, &info.ordering_ty, &ordering_discr_layout);

    let is_less =
        codegen_icmp_imm(fx, IntCC::Equal, ordering_discr, info.ordering_less_discr as i128);
    let is_equal =
        codegen_icmp_imm(fx, IntCC::Equal, ordering_discr, info.ordering_equal_discr as i128);

    let none_state = fx.bcx.ins().iconst(types::I8, 0);
    let equal_state = fx.bcx.ins().iconst(types::I8, 1);
    let less_state = fx.bcx.ins().iconst(types::I8, 2);
    let greater_state = fx.bcx.ins().iconst(types::I8, 3);

    let non_less_state = fx.bcx.ins().select(is_equal, equal_state, greater_state);
    let some_state = fx.bcx.ins().select(is_less, less_state, non_less_state);
    fx.bcx.ins().select(is_none, none_state, some_state)
}

fn codegen_direct_call_from_cvalues_into_dest(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
    args: &[CValue],
    dest: &CPlace,
) {
    let is_extern =
        matches!(callee_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
    let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();

    let interner = DbInterner::new_no_crate(fx.db());
    let empty_args = GenericArgs::empty(interner);
    let (callee_sig, callee_name) = if is_extern {
        let sig =
            build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id, empty_args)
                .expect("extern fn sig");
        let name = extern_fn_symbol_name(fx.db(), callee_func_id);
        (sig, name)
    } else if let Ok(callee_body) = fx.db().monomorphized_mir_body(
        callee_func_id.into(),
        generic_args.store(),
        fx.env().clone(),
    ) {
        let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body).expect("callee sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args,
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    } else if is_cross_crate {
        let sig =
            build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id, generic_args)
                .expect("cross-crate fn sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args,
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    } else {
        panic!("failed to get local callee MIR for {callee_func_id:?}");
    };

    let callee_id = fx
        .module
        .declare_function(&callee_name, Linkage::Import, &callee_sig)
        .expect("declare callee");
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    let is_sret_return = !dest.layout.is_zst()
        && !matches!(
            dest.layout.backend_repr,
            BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
        );

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
        if arg.layout.is_zst() {
            continue;
        }
        match arg.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = arg.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            BackendRepr::Scalar(_) => {
                call_args.push(arg.load_scalar(fx));
            }
            _ => {
                let ptr = arg.clone().force_stack(fx);
                call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
            }
        }
    }

    assert_eq!(
        call_args.len(),
        callee_sig.params.len(),
        "direct call ABI mismatch for {callee_name}: params={} args={} callee={:?}",
        callee_sig.params.len(),
        call_args.len(),
        callee_func_id,
    );

    let call = fx.bcx.ins().call(callee_ref, &call_args);
    if let Some(slot) = sret_slot {
        let slot_val = slot.to_cvalue(fx);
        dest.write_cvalue(fx, slot_val);
        return;
    }

    if dest.layout.is_zst() {
        return;
    }

    let results = fx.bcx.inst_results(call);
    match dest.layout.backend_repr {
        BackendRepr::Scalar(_) => {
            dest.write_cvalue(fx, CValue::by_val(results[0], dest.layout.clone()));
        }
        BackendRepr::ScalarPair(_, _) => {
            dest.write_cvalue(fx, CValue::by_val_pair(results[0], results[1], dest.layout.clone()));
        }
        _ => {}
    }
}

fn codegen_trait_method_call_returning_cvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    trait_method_func_id: hir_def::FunctionId,
    self_ty: StoredTy,
    include_rhs_self: bool,
    args: &[CValue],
) -> CValue {
    let method_substs = trait_method_substs_for_derive_call(
        fx.db(),
        fx.local_crate(),
        trait_method_func_id,
        &self_ty,
        include_rhs_self,
    );
    let (resolved_callable, resolved_args) =
        fx.db().lookup_impl_method(fx.env().as_ref(), trait_method_func_id, method_substs);
    let resolved_args = resolved_args.store();

    let output_ty = method_output_ty(fx.db(), trait_method_func_id, method_substs);
    let output_layout = fx
        .db()
        .layout_of_ty(output_ty.clone(), fx.env().clone())
        .expect("trait method return layout");
    let output_place = CPlace::new_stack_slot(fx, output_layout.clone());

    match resolved_callable {
        Either::Left(resolved_id) => codegen_direct_call_from_cvalues_into_dest(
            fx,
            resolved_id,
            resolved_args.as_ref(),
            args,
            &output_place,
        ),
        Either::Right((derive_impl_id, derive_method)) => codegen_builtin_derive_method_impl(
            fx,
            derive_impl_id,
            derive_method,
            resolved_args.as_ref(),
            self_ty,
            output_ty,
            args,
            &output_place,
        ),
    }

    output_place.to_cvalue(fx)
}

fn codegen_builtin_derive_method_impl(
    fx: &mut FunctionCx<'_, impl Module>,
    derive_impl_id: BuiltinDeriveImplId,
    derive_method: BuiltinDeriveImplMethod,
    _generic_args: GenericArgs<'_>,
    self_ty: StoredTy,
    result_ty: StoredTy,
    args: &[CValue],
    dest: &CPlace,
) {
    let trait_method_func_id =
        derive_method.trait_method(fx.db(), derive_impl_id).expect("builtin derive trait method");

    match derive_method {
        BuiltinDeriveImplMethod::default => {
            assert!(args.is_empty(), "builtin Default::default expects no arguments");

            let TyKind::Adt(adt_id, adt_args) = self_ty.as_ref().kind() else {
                panic!("builtin Default::default expects ADT self type, got {self_ty:?}");
            };
            let interner = DbInterner::new_no_crate(fx.db());

            match adt_id.inner().id {
                hir_def::AdtId::StructId(struct_id) => {
                    for (field_idx, (_, field_ty)) in
                        fx.db().field_types(struct_id.into()).iter().enumerate()
                    {
                        let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                        let field_layout = fx
                            .db()
                            .layout_of_ty(field_ty.clone(), fx.env().clone())
                            .expect("builtin Default::default struct field layout");
                        let field_default = codegen_trait_method_call_returning_cvalue(
                            fx,
                            trait_method_func_id,
                            field_ty,
                            false,
                            &[],
                        );
                        let dest_field = dest.place_field(fx, field_idx, field_layout);
                        dest_field.write_cvalue(fx, field_default);
                    }
                }
                hir_def::AdtId::EnumId(enum_id) => {
                    let variants = &enum_id.enum_variants(fx.db()).variants;
                    let variant_idx = enum_default_variant_idx(fx.db(), enum_id);
                    let variant_layout = variant_layout(&dest.layout, variant_idx);
                    let dest_variant = dest.downcast_variant(variant_layout);
                    let variant_id = variants[variant_idx.as_usize()].0;

                    for (field_idx, (_, field_ty)) in
                        fx.db().field_types(variant_id.into()).iter().enumerate()
                    {
                        let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                        let field_layout = fx
                            .db()
                            .layout_of_ty(field_ty.clone(), fx.env().clone())
                            .expect("builtin Default::default enum field layout");
                        let field_default = codegen_trait_method_call_returning_cvalue(
                            fx,
                            trait_method_func_id,
                            field_ty,
                            false,
                            &[],
                        );
                        let dest_field = dest_variant.place_field(fx, field_idx, field_layout);
                        dest_field.write_cvalue(fx, field_default);
                    }

                    codegen_set_discriminant(fx, dest, &self_ty, variant_idx);
                }
                hir_def::AdtId::UnionId(_) => {
                    panic!("builtin Default::default is not supported for unions")
                }
            }
        }
        BuiltinDeriveImplMethod::clone => {
            assert_eq!(args.len(), 1, "builtin Clone::clone expects one receiver argument");

            let self_ref = args[0].clone();
            let BackendRepr::Scalar(_) = self_ref.layout.backend_repr else {
                panic!("builtin Clone::clone expects a thin &Self receiver");
            };
            let self_ptr = self_ref.load_scalar(fx);
            let src_place = CPlace::for_ptr(pointer::Pointer::new(self_ptr), dest.layout.clone());

            let TyKind::Adt(adt_id, adt_args) = self_ty.as_ref().kind() else {
                panic!("builtin Clone::clone expects ADT self type, got {self_ty:?}");
            };
            let interner = DbInterner::new_no_crate(fx.db());

            match adt_id.inner().id {
                hir_def::AdtId::StructId(struct_id) => {
                    for (field_idx, (_, field_ty)) in
                        fx.db().field_types(struct_id.into()).iter().enumerate()
                    {
                        let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                        let field_layout = fx
                            .db()
                            .layout_of_ty(field_ty.clone(), fx.env().clone())
                            .expect("builtin Clone::clone struct field layout");

                        let src_field = src_place.place_field(fx, field_idx, field_layout.clone());
                        let dest_field = dest.place_field(fx, field_idx, field_layout);

                        let ref_layout = shared_ref_layout_for_pointee(fx, &field_ty);
                        let src_ref = src_field.place_ref(fx, ref_layout);
                        let cloned_field = codegen_trait_method_call_returning_cvalue(
                            fx,
                            trait_method_func_id,
                            field_ty,
                            false,
                            &[src_ref],
                        );
                        dest_field.write_cvalue(fx, cloned_field);
                    }
                }
                hir_def::AdtId::UnionId(_) => {
                    let src_val = src_place.to_cvalue(fx);
                    dest.write_cvalue(fx, src_val);
                }
                hir_def::AdtId::EnumId(enum_id) => {
                    use rustc_abi::Variants;
                    match &dest.layout.variants {
                        Variants::Single { index } => {
                            let variant_idx = *index;
                            let variant_layout = variant_layout(&dest.layout, variant_idx);
                            let src_variant = src_place.downcast_variant(variant_layout.clone());
                            let dest_variant = dest.downcast_variant(variant_layout);
                            let variant_id =
                                enum_id.enum_variants(fx.db()).variants[variant_idx.as_usize()].0;
                            for (field_idx, (_, field_ty)) in
                                fx.db().field_types(variant_id.into()).iter().enumerate()
                            {
                                let field_ty =
                                    field_ty.get().instantiate(interner, adt_args).store();
                                let field_layout = fx
                                    .db()
                                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                                    .expect("builtin Clone::clone enum field layout");

                                let src_field =
                                    src_variant.place_field(fx, field_idx, field_layout.clone());
                                let dest_field =
                                    dest_variant.place_field(fx, field_idx, field_layout);

                                let ref_layout = shared_ref_layout_for_pointee(fx, &field_ty);
                                let src_ref = src_field.place_ref(fx, ref_layout);
                                let cloned_field = codegen_trait_method_call_returning_cvalue(
                                    fx,
                                    trait_method_func_id,
                                    field_ty,
                                    false,
                                    &[src_ref],
                                );
                                dest_field.write_cvalue(fx, cloned_field);
                            }
                        }
                        Variants::Multiple { .. } => {
                            let discr_layout = enum_discriminant_layout(fx, &dest.layout);
                            let discr =
                                codegen_get_discriminant(fx, &src_place, &self_ty, &discr_layout);
                            let discr_ty = fx.bcx.func.dfg.value_type(discr);

                            let done_block = fx.bcx.create_block();
                            let trap_block = fx.bcx.create_block();
                            let mut switch = Switch::new();
                            let mut variant_blocks = Vec::new();

                            for (variant_i, _) in
                                enum_id.enum_variants(fx.db()).variants.iter().enumerate()
                            {
                                let variant_idx = VariantIdx::from_u32(variant_i as u32);
                                let discr_bits =
                                    enum_variant_discriminant_bits(fx.db(), &self_ty, variant_idx);
                                let discr_bits = truncate_bits_to_clif_int_ty(discr_bits, discr_ty);
                                let block = fx.bcx.create_block();
                                switch.set_entry(discr_bits, block);
                                variant_blocks.push((variant_idx, block));
                            }

                            switch.emit(&mut fx.bcx, discr, trap_block);

                            for (variant_idx, block) in variant_blocks {
                                fx.bcx.switch_to_block(block);
                                let variant_layout = variant_layout(&dest.layout, variant_idx);
                                let src_variant =
                                    src_place.downcast_variant(variant_layout.clone());
                                let dest_variant = dest.downcast_variant(variant_layout);
                                let variant_id = enum_id.enum_variants(fx.db()).variants
                                    [variant_idx.as_usize()]
                                .0;

                                for (field_idx, (_, field_ty)) in
                                    fx.db().field_types(variant_id.into()).iter().enumerate()
                                {
                                    let field_ty =
                                        field_ty.get().instantiate(interner, adt_args).store();
                                    let field_layout = fx
                                        .db()
                                        .layout_of_ty(field_ty.clone(), fx.env().clone())
                                        .expect("builtin Clone::clone enum field layout");

                                    let src_field = src_variant.place_field(
                                        fx,
                                        field_idx,
                                        field_layout.clone(),
                                    );
                                    let dest_field =
                                        dest_variant.place_field(fx, field_idx, field_layout);

                                    let ref_layout = shared_ref_layout_for_pointee(fx, &field_ty);
                                    let src_ref = src_field.place_ref(fx, ref_layout);
                                    let cloned_field = codegen_trait_method_call_returning_cvalue(
                                        fx,
                                        trait_method_func_id,
                                        field_ty,
                                        false,
                                        &[src_ref],
                                    );
                                    dest_field.write_cvalue(fx, cloned_field);
                                }

                                codegen_set_discriminant(fx, dest, &self_ty, variant_idx);
                                fx.bcx.ins().jump(done_block, &[]);
                            }

                            fx.bcx.switch_to_block(trap_block);
                            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

                            fx.bcx.switch_to_block(done_block);
                        }
                        Variants::Empty => {
                            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());
                        }
                    }
                }
            }
        }
        BuiltinDeriveImplMethod::fmt => {
            if !dest.layout.is_zst() {
                let TyKind::Adt(adt, _) = result_ty.as_ref().kind() else {
                    panic!("builtin Debug::fmt must return Result, got: {result_ty:?}")
                };
                let hir_def::AdtId::EnumId(_) = adt.inner().id else {
                    panic!("builtin Debug::fmt must return enum Result, got: {result_ty:?}")
                };
                // `core::fmt::Result` is `Result<(), fmt::Error>`; variant 0 is `Ok`.
                codegen_set_discriminant(fx, dest, &result_ty, VariantIdx::from_u32(0));
            }
        }
        BuiltinDeriveImplMethod::eq => {
            assert_eq!(args.len(), 2, "builtin PartialEq::eq expects two receiver arguments");

            let self_layout = fx
                .db()
                .layout_of_ty(self_ty.clone(), fx.env().clone())
                .expect("PartialEq self layout");
            let lhs_ref = args[0].clone();
            let rhs_ref = args[1].clone();
            let BackendRepr::Scalar(_) = lhs_ref.layout.backend_repr else {
                panic!("builtin PartialEq::eq expects a thin &Self lhs receiver");
            };
            let BackendRepr::Scalar(_) = rhs_ref.layout.backend_repr else {
                panic!("builtin PartialEq::eq expects a thin &Self rhs receiver");
            };

            let lhs_ptr = lhs_ref.load_scalar(fx);
            let rhs_ptr = rhs_ref.load_scalar(fx);
            let lhs_val = CPlace::for_ptr(pointer::Pointer::new(lhs_ptr), self_layout.clone());
            let rhs_val = CPlace::for_ptr(pointer::Pointer::new(rhs_ptr), self_layout.clone());

            let bool_ty = bool_layout_ty(fx.dl, &dest.layout);
            let mut all_eq = fx.bcx.ins().iconst(bool_ty, 1);

            match self_ty.as_ref().kind() {
                TyKind::Adt(adt_id, adt_args) => {
                    let interner = DbInterner::new_no_crate(fx.db());
                    match adt_id.inner().id {
                        hir_def::AdtId::StructId(struct_id) => {
                            for (field_idx, (_, field_ty)) in
                                fx.db().field_types(struct_id.into()).iter().enumerate()
                            {
                                let field_ty =
                                    field_ty.get().instantiate(interner, adt_args).store();
                                let field_layout = fx
                                    .db()
                                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                                    .expect("PartialEq struct field layout");

                                let lhs_field =
                                    lhs_val.place_field(fx, field_idx, field_layout.clone());
                                let rhs_field = rhs_val.place_field(fx, field_idx, field_layout);
                                let ref_layout = shared_ref_layout_for_pointee(fx, &field_ty);
                                let lhs_ref = lhs_field.place_ref(fx, ref_layout.clone());
                                let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                let eq_res = codegen_trait_method_call_returning_cvalue(
                                    fx,
                                    trait_method_func_id,
                                    field_ty,
                                    true,
                                    &[lhs_ref, rhs_ref],
                                );
                                let eq_scalar = bool_cvalue_to_bool_scalar(fx, eq_res, bool_ty);
                                all_eq = fx.bcx.ins().band(all_eq, eq_scalar);
                            }
                        }
                        hir_def::AdtId::UnionId(_) => {
                            all_eq = fx.bcx.ins().iconst(bool_ty, 1);
                        }
                        hir_def::AdtId::EnumId(enum_id) => {
                            use rustc_abi::Variants;
                            match &self_layout.variants {
                                Variants::Single { index } => {
                                    let variant_idx = *index;
                                    let variant_layout = variant_layout(&self_layout, variant_idx);
                                    let lhs_variant =
                                        lhs_val.downcast_variant(variant_layout.clone());
                                    let rhs_variant = rhs_val.downcast_variant(variant_layout);
                                    let variant_id = enum_id.enum_variants(fx.db()).variants
                                        [variant_idx.as_usize()]
                                    .0;
                                    for (field_idx, (_, field_ty)) in
                                        fx.db().field_types(variant_id.into()).iter().enumerate()
                                    {
                                        let field_ty =
                                            field_ty.get().instantiate(interner, adt_args).store();
                                        let field_layout = fx
                                            .db()
                                            .layout_of_ty(field_ty.clone(), fx.env().clone())
                                            .expect("PartialEq enum field layout");

                                        let lhs_field = lhs_variant.place_field(
                                            fx,
                                            field_idx,
                                            field_layout.clone(),
                                        );
                                        let rhs_field =
                                            rhs_variant.place_field(fx, field_idx, field_layout);
                                        let ref_layout =
                                            shared_ref_layout_for_pointee(fx, &field_ty);
                                        let lhs_ref = lhs_field.place_ref(fx, ref_layout.clone());
                                        let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                        let eq_res = codegen_trait_method_call_returning_cvalue(
                                            fx,
                                            trait_method_func_id,
                                            field_ty,
                                            true,
                                            &[lhs_ref, rhs_ref],
                                        );
                                        let eq_scalar =
                                            bool_cvalue_to_bool_scalar(fx, eq_res, bool_ty);
                                        all_eq = fx.bcx.ins().band(all_eq, eq_scalar);
                                    }
                                }
                                Variants::Multiple { .. } => {
                                    let discr_layout = enum_discriminant_layout(fx, &self_layout);
                                    let lhs_discr = codegen_get_discriminant(
                                        fx,
                                        &lhs_val,
                                        &self_ty,
                                        &discr_layout,
                                    );
                                    let rhs_discr = codegen_get_discriminant(
                                        fx,
                                        &rhs_val,
                                        &self_ty,
                                        &discr_layout,
                                    );
                                    let discr_ty = fx.bcx.func.dfg.value_type(lhs_discr);

                                    let discr_eq =
                                        fx.bcx.ins().icmp(IntCC::Equal, lhs_discr, rhs_discr);
                                    let fields_eq_place =
                                        CPlace::new_stack_slot(fx, dest.layout.clone());
                                    let zero = fx.bcx.ins().iconst(bool_ty, 0);
                                    fields_eq_place.write_cvalue(
                                        fx,
                                        CValue::by_val(zero, dest.layout.clone()),
                                    );

                                    let done_block = fx.bcx.create_block();
                                    let trap_block = fx.bcx.create_block();
                                    let mut switch = Switch::new();
                                    let mut variant_blocks = Vec::new();

                                    for (variant_i, _) in
                                        enum_id.enum_variants(fx.db()).variants.iter().enumerate()
                                    {
                                        let variant_idx = VariantIdx::from_u32(variant_i as u32);
                                        let discr_bits = enum_variant_discriminant_bits(
                                            fx.db(),
                                            &self_ty,
                                            variant_idx,
                                        );
                                        let discr_bits =
                                            truncate_bits_to_clif_int_ty(discr_bits, discr_ty);
                                        let block = fx.bcx.create_block();
                                        switch.set_entry(discr_bits, block);
                                        variant_blocks.push((variant_idx, block));
                                    }

                                    switch.emit(&mut fx.bcx, lhs_discr, trap_block);

                                    for (variant_idx, block) in variant_blocks {
                                        fx.bcx.switch_to_block(block);
                                        let variant_layout =
                                            variant_layout(&self_layout, variant_idx);
                                        let lhs_variant =
                                            lhs_val.downcast_variant(variant_layout.clone());
                                        let rhs_variant = rhs_val.downcast_variant(variant_layout);
                                        let variant_id = enum_id.enum_variants(fx.db()).variants
                                            [variant_idx.as_usize()]
                                        .0;
                                        let mut variant_eq = fx.bcx.ins().iconst(bool_ty, 1);

                                        for (field_idx, (_, field_ty)) in fx
                                            .db()
                                            .field_types(variant_id.into())
                                            .iter()
                                            .enumerate()
                                        {
                                            let field_ty = field_ty
                                                .get()
                                                .instantiate(interner, adt_args)
                                                .store();
                                            let field_layout = fx
                                                .db()
                                                .layout_of_ty(field_ty.clone(), fx.env().clone())
                                                .expect("PartialEq enum field layout");

                                            let lhs_field = lhs_variant.place_field(
                                                fx,
                                                field_idx,
                                                field_layout.clone(),
                                            );
                                            let rhs_field = rhs_variant.place_field(
                                                fx,
                                                field_idx,
                                                field_layout,
                                            );
                                            let ref_layout =
                                                shared_ref_layout_for_pointee(fx, &field_ty);
                                            let lhs_ref =
                                                lhs_field.place_ref(fx, ref_layout.clone());
                                            let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                            let eq_res = codegen_trait_method_call_returning_cvalue(
                                                fx,
                                                trait_method_func_id,
                                                field_ty,
                                                true,
                                                &[lhs_ref, rhs_ref],
                                            );
                                            let eq_scalar =
                                                bool_cvalue_to_bool_scalar(fx, eq_res, bool_ty);
                                            variant_eq = fx.bcx.ins().band(variant_eq, eq_scalar);
                                        }

                                        fields_eq_place.write_cvalue(
                                            fx,
                                            CValue::by_val(variant_eq, dest.layout.clone()),
                                        );
                                        fx.bcx.ins().jump(done_block, &[]);
                                    }

                                    fx.bcx.switch_to_block(trap_block);
                                    fx.bcx
                                        .ins()
                                        .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

                                    fx.bcx.switch_to_block(done_block);

                                    let one = fx.bcx.ins().iconst(bool_ty, 1);
                                    let zero = fx.bcx.ins().iconst(bool_ty, 0);
                                    let discr_eq_scalar = fx.bcx.ins().select(discr_eq, one, zero);
                                    let fields_eq = fields_eq_place.to_cvalue(fx).load_scalar(fx);
                                    all_eq = fx.bcx.ins().band(discr_eq_scalar, fields_eq);
                                }
                                Variants::Empty => {
                                    all_eq = fx.bcx.ins().iconst(bool_ty, 0);
                                }
                            }
                        }
                    }
                }
                _ => panic!("builtin PartialEq::eq expects ADT self type, got {self_ty:?}"),
            }

            dest.write_cvalue(fx, CValue::by_val(all_eq, dest.layout.clone()));
        }
        BuiltinDeriveImplMethod::partial_cmp => {
            assert_eq!(
                args.len(),
                2,
                "builtin PartialOrd::partial_cmp expects two receiver arguments"
            );

            let self_layout = fx
                .db()
                .layout_of_ty(self_ty.clone(), fx.env().clone())
                .expect("PartialOrd self layout");
            let lhs_ref = args[0].clone();
            let rhs_ref = args[1].clone();
            let BackendRepr::Scalar(_) = lhs_ref.layout.backend_repr else {
                panic!("builtin PartialOrd::partial_cmp expects a thin &Self lhs receiver");
            };
            let BackendRepr::Scalar(_) = rhs_ref.layout.backend_repr else {
                panic!("builtin PartialOrd::partial_cmp expects a thin &Self rhs receiver");
            };

            let lhs_ptr = lhs_ref.load_scalar(fx);
            let rhs_ptr = rhs_ref.load_scalar(fx);
            let lhs_val = CPlace::for_ptr(pointer::Pointer::new(lhs_ptr), self_layout.clone());
            let rhs_val = CPlace::for_ptr(pointer::Pointer::new(rhs_ptr), self_layout.clone());

            let info = option_ordering_info(fx, result_ty.clone(), dest.layout.clone());

            match self_ty.as_ref().kind() {
                TyKind::Adt(adt_id, adt_args) => {
                    let interner = DbInterner::new_no_crate(fx.db());
                    match adt_id.inner().id {
                        hir_def::AdtId::StructId(struct_id) => {
                            let mut state = fx.bcx.ins().iconst(types::I8, 1);
                            for (field_idx, (_, field_ty)) in
                                fx.db().field_types(struct_id.into()).iter().enumerate()
                            {
                                let field_ty =
                                    field_ty.get().instantiate(interner, adt_args).store();
                                let field_layout = fx
                                    .db()
                                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                                    .expect("PartialOrd struct field layout");
                                let lhs_field =
                                    lhs_val.place_field(fx, field_idx, field_layout.clone());
                                let rhs_field = rhs_val.place_field(fx, field_idx, field_layout);

                                let ref_layout = shared_ref_layout_for_pointee(fx, &field_ty);
                                let lhs_ref = lhs_field.place_ref(fx, ref_layout.clone());
                                let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                let cmp_res = codegen_trait_method_call_returning_cvalue(
                                    fx,
                                    trait_method_func_id,
                                    field_ty,
                                    true,
                                    &[lhs_ref, rhs_ref],
                                );
                                let cmp_state = decode_option_ordering_state(fx, &info, cmp_res);
                                let is_running_equal =
                                    fx.bcx.ins().icmp_imm(IntCC::Equal, state, 1);
                                state = fx.bcx.ins().select(is_running_equal, cmp_state, state);
                            }
                            write_option_ordering_state(fx, &info, dest, state);
                        }
                        hir_def::AdtId::UnionId(_) => {
                            write_option_ordering_variant(
                                fx,
                                &info,
                                dest,
                                Some(info.ordering_equal_idx),
                            );
                        }
                        hir_def::AdtId::EnumId(enum_id) => {
                            use rustc_abi::Variants;
                            match &self_layout.variants {
                                Variants::Single { index } => {
                                    let variant_idx = *index;
                                    let variant_layout = variant_layout(&self_layout, variant_idx);
                                    let lhs_variant =
                                        lhs_val.downcast_variant(variant_layout.clone());
                                    let rhs_variant = rhs_val.downcast_variant(variant_layout);
                                    let variant_id = enum_id.enum_variants(fx.db()).variants
                                        [variant_idx.as_usize()]
                                    .0;

                                    let mut state = fx.bcx.ins().iconst(types::I8, 1);
                                    for (field_idx, (_, field_ty)) in
                                        fx.db().field_types(variant_id.into()).iter().enumerate()
                                    {
                                        let field_ty =
                                            field_ty.get().instantiate(interner, adt_args).store();
                                        let field_layout = fx
                                            .db()
                                            .layout_of_ty(field_ty.clone(), fx.env().clone())
                                            .expect("PartialOrd enum field layout");
                                        let lhs_field = lhs_variant.place_field(
                                            fx,
                                            field_idx,
                                            field_layout.clone(),
                                        );
                                        let rhs_field =
                                            rhs_variant.place_field(fx, field_idx, field_layout);

                                        let ref_layout =
                                            shared_ref_layout_for_pointee(fx, &field_ty);
                                        let lhs_ref = lhs_field.place_ref(fx, ref_layout.clone());
                                        let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                        let cmp_res = codegen_trait_method_call_returning_cvalue(
                                            fx,
                                            trait_method_func_id,
                                            field_ty,
                                            true,
                                            &[lhs_ref, rhs_ref],
                                        );
                                        let cmp_state =
                                            decode_option_ordering_state(fx, &info, cmp_res);
                                        let is_running_equal =
                                            fx.bcx.ins().icmp_imm(IntCC::Equal, state, 1);
                                        state =
                                            fx.bcx.ins().select(is_running_equal, cmp_state, state);
                                    }

                                    write_option_ordering_state(fx, &info, dest, state);
                                }
                                Variants::Multiple { .. } => {
                                    let discr_layout = enum_discriminant_layout(fx, &self_layout);
                                    let lhs_discr = codegen_get_discriminant(
                                        fx,
                                        &lhs_val,
                                        &self_ty,
                                        &discr_layout,
                                    );
                                    let rhs_discr = codegen_get_discriminant(
                                        fx,
                                        &rhs_val,
                                        &self_ty,
                                        &discr_layout,
                                    );
                                    let switch_discr_ty = fx.bcx.func.dfg.value_type(lhs_discr);

                                    let is_eq =
                                        fx.bcx.ins().icmp(IntCC::Equal, lhs_discr, rhs_discr);
                                    let discr_scalar_ty = fx.bcx.func.dfg.value_type(lhs_discr);
                                    let discr_signed = match discr_layout.backend_repr {
                                        BackendRepr::Scalar(s) => {
                                            matches!(s.primitive(), Primitive::Int(_, true))
                                        }
                                        _ => false,
                                    };
                                    let is_lt = if discr_signed {
                                        fx.bcx.ins().icmp(
                                            IntCC::SignedLessThan,
                                            lhs_discr,
                                            rhs_discr,
                                        )
                                    } else {
                                        fx.bcx.ins().icmp(
                                            IntCC::UnsignedLessThan,
                                            lhs_discr,
                                            rhs_discr,
                                        )
                                    };
                                    let is_gt = if discr_signed {
                                        fx.bcx.ins().icmp(
                                            IntCC::SignedGreaterThan,
                                            lhs_discr,
                                            rhs_discr,
                                        )
                                    } else {
                                        fx.bcx.ins().icmp(
                                            IntCC::UnsignedGreaterThan,
                                            lhs_discr,
                                            rhs_discr,
                                        )
                                    };

                                    let done_block = fx.bcx.create_block();
                                    let same_variant_block = fx.bcx.create_block();
                                    let trap_block = fx.bcx.create_block();

                                    fx.bcx.ins().brif(
                                        is_eq,
                                        same_variant_block,
                                        &[],
                                        done_block,
                                        &[],
                                    );

                                    fx.bcx.switch_to_block(done_block);
                                    let less_state = fx.bcx.ins().iconst(types::I8, 2);
                                    let greater_state = fx.bcx.ins().iconst(types::I8, 3);
                                    let equal_state = fx.bcx.ins().iconst(types::I8, 1);
                                    let non_lt_state =
                                        fx.bcx.ins().select(is_gt, greater_state, equal_state);
                                    let discr_state =
                                        fx.bcx.ins().select(is_lt, less_state, non_lt_state);
                                    write_option_ordering_state(fx, &info, dest, discr_state);
                                    let finished_block = fx.bcx.create_block();
                                    fx.bcx.ins().jump(finished_block, &[]);

                                    fx.bcx.switch_to_block(same_variant_block);
                                    let mut switch = Switch::new();
                                    let mut variant_blocks = Vec::new();
                                    for (variant_i, _) in
                                        enum_id.enum_variants(fx.db()).variants.iter().enumerate()
                                    {
                                        let variant_idx = VariantIdx::from_u32(variant_i as u32);
                                        let discr_bits = enum_variant_discriminant_bits(
                                            fx.db(),
                                            &self_ty,
                                            variant_idx,
                                        );
                                        let discr_bits = truncate_bits_to_clif_int_ty(
                                            discr_bits,
                                            switch_discr_ty,
                                        );
                                        let block = fx.bcx.create_block();
                                        switch.set_entry(discr_bits, block);
                                        variant_blocks.push((variant_idx, block));
                                    }
                                    switch.emit(&mut fx.bcx, lhs_discr, trap_block);

                                    for (variant_idx, block) in variant_blocks {
                                        fx.bcx.switch_to_block(block);
                                        let variant_layout =
                                            variant_layout(&self_layout, variant_idx);
                                        let lhs_variant =
                                            lhs_val.downcast_variant(variant_layout.clone());
                                        let rhs_variant = rhs_val.downcast_variant(variant_layout);
                                        let variant_id = enum_id.enum_variants(fx.db()).variants
                                            [variant_idx.as_usize()]
                                        .0;

                                        let mut state = fx.bcx.ins().iconst(types::I8, 1);
                                        for (field_idx, (_, field_ty)) in fx
                                            .db()
                                            .field_types(variant_id.into())
                                            .iter()
                                            .enumerate()
                                        {
                                            let field_ty = field_ty
                                                .get()
                                                .instantiate(interner, adt_args)
                                                .store();
                                            let field_layout = fx
                                                .db()
                                                .layout_of_ty(field_ty.clone(), fx.env().clone())
                                                .expect("PartialOrd enum field layout");
                                            let lhs_field = lhs_variant.place_field(
                                                fx,
                                                field_idx,
                                                field_layout.clone(),
                                            );
                                            let rhs_field = rhs_variant.place_field(
                                                fx,
                                                field_idx,
                                                field_layout,
                                            );

                                            let ref_layout =
                                                shared_ref_layout_for_pointee(fx, &field_ty);
                                            let lhs_ref =
                                                lhs_field.place_ref(fx, ref_layout.clone());
                                            let rhs_ref = rhs_field.place_ref(fx, ref_layout);
                                            let cmp_res =
                                                codegen_trait_method_call_returning_cvalue(
                                                    fx,
                                                    trait_method_func_id,
                                                    field_ty,
                                                    true,
                                                    &[lhs_ref, rhs_ref],
                                                );
                                            let cmp_state =
                                                decode_option_ordering_state(fx, &info, cmp_res);
                                            let is_running_equal =
                                                fx.bcx.ins().icmp_imm(IntCC::Equal, state, 1);
                                            state = fx.bcx.ins().select(
                                                is_running_equal,
                                                cmp_state,
                                                state,
                                            );
                                        }

                                        write_option_ordering_state(fx, &info, dest, state);
                                        fx.bcx.ins().jump(finished_block, &[]);
                                    }

                                    fx.bcx.switch_to_block(trap_block);
                                    fx.bcx
                                        .ins()
                                        .trap(cranelift_codegen::ir::TrapCode::user(1).unwrap());

                                    fx.bcx.switch_to_block(finished_block);

                                    let _ = discr_scalar_ty;
                                }
                                Variants::Empty => {
                                    write_option_ordering_variant(fx, &info, dest, None);
                                }
                            }
                        }
                    }
                }
                _ => {
                    panic!("builtin PartialOrd::partial_cmp expects ADT self type, got {self_ty:?}")
                }
            }
        }
        _ => panic!(
            "unsupported builtin-derive method during codegen: {:?}::{:?}",
            derive_impl_id, derive_method
        ),
    }
}

/// Lower calls that resolve to builtin-derive pseudo methods.
///
/// These methods do not have concrete MIR bodies or symbols, so codegen must
/// handle them directly.
fn codegen_builtin_derive_method_call(
    fx: &mut FunctionCx<'_, impl Module>,
    derive_impl_id: BuiltinDeriveImplId,
    derive_method: BuiltinDeriveImplMethod,
    generic_args: GenericArgs<'_>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    let dest = codegen_place(fx, destination);
    let arg_values: Vec<_> = args.iter().map(|arg| codegen_operand(fx, &arg.kind)).collect();
    let result_ty = place_ty(fx.db(), fx.ra_body(), destination);

    let self_ty = if let Some(first_arg) = args.first() {
        operand_ty(fx.db(), fx.ra_body(), &first_arg.kind)
            .as_ref()
            .builtin_deref(true)
            .map(|ty| ty.store())
            .or_else(|| (!generic_args.is_empty()).then(|| generic_args.type_at(0).store()))
            .expect("builtin derive method receiver must be a reference")
    } else if matches!(derive_method, BuiltinDeriveImplMethod::default) {
        // `Default::default` is an associated function with no receiver.
        result_ty.clone()
    } else {
        assert!(
            !generic_args.is_empty(),
            "builtin derive method missing generic args: impl={derive_impl_id:?} method={derive_method:?} args_len={} result_ty={result_ty:?}",
            args.len(),
        );
        generic_args.type_at(0).store()
    };

    codegen_builtin_derive_method_impl(
        fx,
        derive_impl_id,
        derive_method,
        generic_args,
        self_ty,
        result_ty,
        &arg_values,
        &dest,
    );

    if target.is_some() && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
    {
        fx.set_drop_flag(destination.local);
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
    call_operand_lowering: CallOperandLowering,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // `core::ptr::drop_in_place` is a lang-item shim with intentionally
    // recursive source; rustc replaces it with drop glue. Mirror that here
    // by lowering to our generated `drop_in_place::<T>` glue directly.
    let lang_items = hir_def::lang_item::lang_items(fx.db(), fx.local_crate());
    if Some(callee_func_id) == lang_items.DropInPlace {
        assert_eq!(args.len(), 1, "drop_in_place expects 1 argument");
        assert!(generic_args.len() > 0, "drop_in_place requires pointee generic arg");

        let pointee_ty = generic_args.type_at(0).store();
        let interner = DbInterner::new_with(fx.db(), fx.local_crate());
        if hir_ty::drop::has_drop_glue_mono(interner, pointee_ty.as_ref()) {
            let arg = codegen_operand(fx, &args[0].kind);
            match arg.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    let ptr = arg.load_scalar(fx);

                    let mut drop_sig = Signature::new(fx.isa.default_call_conv());
                    drop_sig.params.push(AbiParam::new(fx.pointer_type));
                    let fn_name = symbol_mangling::mangle_drop_in_place(
                        fx.db(),
                        pointee_ty.as_ref(),
                        fx.ext_crate_disambiguators(),
                    );
                    let callee_id = fx
                        .module
                        .declare_function(&fn_name, Linkage::Import, &drop_sig)
                        .expect("declare drop_in_place glue");
                    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
                    fx.bcx.ins().call(callee_ref, &[ptr]);
                }
                BackendRepr::ScalarPair(_, _) => {
                    let (data_ptr, metadata) = arg.load_scalar_pair(fx);
                    match pointee_ty.as_ref().kind() {
                        TyKind::Dynamic(..) => {
                            let drop_fn =
                                fx.bcx.ins().load(fx.pointer_type, vtable_memflags(), metadata, 0);
                            let is_null = fx.bcx.ins().icmp_imm(IntCC::Equal, drop_fn, 0);

                            let skip_drop = fx.bcx.create_block();
                            let do_drop = fx.bcx.create_block();
                            fx.bcx.ins().brif(is_null, skip_drop, &[], do_drop, &[]);

                            fx.bcx.switch_to_block(do_drop);
                            let mut drop_sig = Signature::new(fx.isa.default_call_conv());
                            drop_sig.params.push(AbiParam::new(fx.pointer_type));
                            let drop_sig_ref = fx.bcx.import_signature(drop_sig);
                            fx.bcx.ins().call_indirect(drop_sig_ref, drop_fn, &[data_ptr]);
                            fx.bcx.ins().jump(skip_drop, &[]);

                            fx.bcx.switch_to_block(skip_drop);
                        }
                        TyKind::Slice(elem_ty) => {
                            if !hir_ty::drop::has_drop_glue_mono(interner, elem_ty) {
                                // Nothing to do; this only happens for degenerate mono paths.
                                // `has_drop_glue_mono([T])` implies `has_drop_glue_mono(T)`.
                            } else {
                                let elem_ty_stored = elem_ty.store();
                                let elem_layout = fx
                                    .db()
                                    .layout_of_ty(elem_ty_stored.clone(), fx.env().clone())
                                    .expect("layout for slice element drop");

                                let mut drop_sig = Signature::new(fx.isa.default_call_conv());
                                drop_sig.params.push(AbiParam::new(fx.pointer_type));
                                let drop_fn_name = symbol_mangling::mangle_drop_in_place(
                                    fx.db(),
                                    elem_ty,
                                    fx.ext_crate_disambiguators(),
                                );
                                let drop_func_id = fx
                                    .module
                                    .declare_function(&drop_fn_name, Linkage::Import, &drop_sig)
                                    .expect("declare slice element drop_in_place");
                                let drop_callee_ref =
                                    fx.module.declare_func_in_func(drop_func_id, fx.bcx.func);

                                let loop_block = fx.bcx.create_block();
                                let body_block = fx.bcx.create_block();
                                let done_block = fx.bcx.create_block();

                                fx.bcx.append_block_param(loop_block, fx.pointer_type);
                                let zero = fx.bcx.ins().iconst(fx.pointer_type, 0);
                                fx.bcx.ins().jump(loop_block, &[BlockArg::Value(zero)]);

                                fx.bcx.switch_to_block(loop_block);
                                let idx = fx.bcx.block_params(loop_block)[0];
                                let finished = fx.bcx.ins().icmp(
                                    IntCC::UnsignedGreaterThanOrEqual,
                                    idx,
                                    metadata,
                                );
                                fx.bcx.ins().brif(finished, done_block, &[], body_block, &[]);

                                fx.bcx.switch_to_block(body_block);
                                let elem_ptr = if elem_layout.size.bytes() == 0 {
                                    data_ptr
                                } else {
                                    let elem_size = fx
                                        .bcx
                                        .ins()
                                        .iconst(fx.pointer_type, elem_layout.size.bytes() as i64);
                                    let byte_offset = fx.bcx.ins().imul(idx, elem_size);
                                    fx.bcx.ins().iadd(data_ptr, byte_offset)
                                };
                                fx.bcx.ins().call(drop_callee_ref, &[elem_ptr]);
                                let next_idx = fx.bcx.ins().iadd_imm(idx, 1);
                                fx.bcx.ins().jump(loop_block, &[BlockArg::Value(next_idx)]);

                                fx.bcx.switch_to_block(done_block);
                            }
                        }
                        _ => {
                            panic!(
                                "drop_in_place for unsized type with drop glue is not supported yet: {:?}",
                                pointee_ty.as_ref().kind()
                            );
                        }
                    }
                }
                _ => panic!("drop_in_place argument must be a pointer"),
            }
        }

        if let Some(target) = target {
            let block = fx.clif_block(*target);
            fx.bcx.ins().jump(block, &[]);
        }
        return;
    }

    if codegen_intrinsic_call(fx, callee_func_id, generic_args, args, destination, target) {
        return;
    }

    let (callee_func_id, generic_args) =
        if let ItemContainerId::TraitId(trait_id) = callee_func_id.loc(fx.db()).container {
            // Check for virtual dispatch: trait method called on dyn Trait
            let interner = DbInterner::new_no_crate(fx.db());
            if hir_ty::method_resolution::is_dyn_method(
                interner,
                fx.env().param_env(),
                callee_func_id,
                generic_args,
            )
            .is_some()
            {
                codegen_virtual_call(
                    fx,
                    callee_func_id,
                    trait_id,
                    generic_args,
                    args,
                    destination,
                    target,
                );
                return;
            }

            let receiver_ty = args
                .first()
                .map(|self_arg| operand_ty(fx.db(), fx.ra_body(), &self_arg.kind).as_ref())
                .or_else(|| (!generic_args.is_empty()).then(|| generic_args.type_at(0)));

            let trait_method_sig = fx.db().function_signature(callee_func_id);
            let trait_method_name = trait_method_sig.name.clone();
            let trait_method_is_rust_call = trait_method_sig.abi == Some(sym::rust_dash_call);

            // Fn/FnMut/FnOnce calls are often encoded with receiver information only in the
            // first MIR argument operand. Deriving callable dispatch from that operand keeps us
            // off unresolved trait-item imports like `FnOnce::call_once`.
            if is_fn_trait_method(fx.db(), trait_id, trait_method_name.as_str()) {
                if let Some(self_ty) = receiver_ty {
                    match peel_ref_layers(self_ty).kind() {
                        TyKind::Closure(closure_id, closure_subst) => {
                            codegen_closure_call(
                                fx,
                                closure_id.0,
                                closure_subst.store(),
                                CallOperandLowering::from_rust_call_abi(trait_method_is_rust_call),
                                args,
                                destination,
                                target,
                            );
                            return;
                        }
                        TyKind::FnDef(def, fn_args) => match def.0 {
                            CallableDefId::FunctionId(fn_id) => {
                                codegen_direct_call(
                                    fx,
                                    fn_id,
                                    fn_args,
                                    CallOperandLowering::from_rust_call_abi(trait_method_is_rust_call),
                                    args.get(1..).unwrap_or(&[]),
                                    destination,
                                    target,
                                );
                                return;
                            }
                            CallableDefId::StructId(struct_id) => {
                                codegen_adt_constructor_call(
                                    fx,
                                    VariantId::StructId(struct_id),
                                    fn_args,
                                    args.get(1..).unwrap_or(&[]),
                                    destination,
                                    target,
                                );
                                return;
                            }
                            CallableDefId::EnumVariantId(variant_id) => {
                                codegen_adt_constructor_call(
                                    fx,
                                    VariantId::EnumVariantId(variant_id),
                                    fn_args,
                                    args.get(1..).unwrap_or(&[]),
                                    destination,
                                    target,
                                );
                                return;
                            }
                        },
                        TyKind::FnPtr(sig_tys, header) if !args.is_empty() => {
                            let fn_ptr_is_rust_call_abi =
                                matches!(header.abi, hir_ty::FnAbi::RustCall);
                            let operand_lowering = if trait_method_is_rust_call {
                                CallOperandLowering::RustCall
                            } else {
                                CallOperandLowering::from_rust_call_abi(fn_ptr_is_rust_call_abi)
                            };
                            codegen_fn_ptr_call(
                                fx,
                                &args[0],
                                &sig_tys,
                                fn_ptr_is_rust_call_abi,
                                operand_lowering,
                                header.c_variadic,
                                args.get(1..).unwrap_or(&[]),
                                destination,
                                target,
                            );
                            return;
                        }
                        _ => {}
                    }
                }
            }

            // Calls like `FnMut::call_mut` on `&mut dyn FnMut(..)` often resolve to the
            // blanket reference impl in `core::ops::function::impls`, but the actual runtime
            // dispatch must still happen through the inner dyn vtable.
            if let Some(self_ty) = receiver_ty
                && let Some(self_pointee) = self_ty.builtin_deref(true)
                && self_pointee.dyn_trait() == Some(trait_id)
                && matches!(
                    fx.db()
                        .layout_of_ty(self_ty.store(), fx.env().clone())
                        .expect("self layout for dyn ref dispatch")
                        .backend_repr,
                    BackendRepr::ScalarPair(_, _)
                )
            {
                codegen_virtual_call(
                    fx,
                    callee_func_id,
                    trait_id,
                    generic_args,
                    args,
                    destination,
                    target,
                );
                return;
            }

            // Static trait dispatch: resolve to concrete impl method when possible.
            let resolved_method =
                match fx.db().lookup_impl_method(fx.env().as_ref(), callee_func_id, generic_args) {
                    (Either::Left(resolved_id), resolved_args) => {
                        Either::Left((resolved_id, resolved_args.store()))
                    }
                    (Either::Right((derive_impl_id, derive_method)), resolved_args) => {
                        Either::Right((derive_impl_id, derive_method, resolved_args.store()))
                    }
                };
            match resolved_method {
                Either::Left((resolved_id, resolved_args)) => (resolved_id, resolved_args),
                Either::Right((derive_impl_id, derive_method, resolved_args)) => {
                    codegen_builtin_derive_method_call(
                        fx,
                        derive_impl_id,
                        derive_method,
                        resolved_args.as_ref(),
                        args,
                        destination,
                        target,
                    );
                    return;
                }
            }
        } else {
            (callee_func_id, generic_args.store())
        };

    let callee_is_rust_call_abi =
        fx.db().function_signature(callee_func_id).abi == Some(sym::rust_dash_call);
    let operand_lowering = if callee_is_rust_call_abi {
        CallOperandLowering::RustCall
    } else {
        call_operand_lowering
    };
    let source_return_ty = callable_output_ty(fx.db(), callee_func_id, generic_args.as_ref());
    // Check if this is an extern function (no MIR available)
    let is_extern =
        matches!(callee_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
    let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();

    let interner = DbInterner::new_no_crate(fx.db());
    let empty_args = GenericArgs::empty(interner);
    let (callee_abi, callee_name) = if is_extern {
        // Extern functions: build signature from type info, use raw symbol name
        let fn_abi = abi::fn_abi_for_fn_item_from_ty(
            fx.isa,
            fx.db(),
            fx.dl,
            fx.env(),
            callee_func_id,
            empty_args,
        )
        .expect("extern fn ABI");
        let name = extern_fn_symbol_name(fx.db(), callee_func_id);
        (fn_abi, name)
    } else if let Ok(callee_body) = fx.db().monomorphized_mir_body(
        callee_func_id.into(),
        generic_args.clone(),
        fx.env().clone(),
    ) {
        // Prefer r-a MIR whenever available, including cross-crate bodies.
        let fn_abi = abi::fn_abi_for_body(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body)
            .expect("callee ABI");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        (fn_abi, name)
    } else if is_cross_crate {
        // Fall back to symbol-only cross-crate calls when MIR lowering is unavailable.
        let fn_abi = abi::fn_abi_for_fn_item_from_ty(
            fx.isa,
            fx.db(),
            fx.dl,
            fx.env(),
            callee_func_id,
            generic_args.as_ref(),
        )
        .expect("cross-crate fn ABI");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        (fn_abi, name)
    } else {
        panic!("failed to get local callee MIR for {:?}", callee_func_id);
    };

    // Declare callee in module (Import linkage — it may be defined elsewhere or in same module)
    let callee_id = fx
        .module
        .declare_function(&callee_name, Linkage::Import, &callee_abi.sig)
        .expect("declare callee");

    // Import into current function to get a FuncRef
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);
    let fixed_param_count = callee_abi.sig.params.len();

    // Determine destination layout to check for sret return
    let dest = codegen_place(fx, destination);
    let is_sret_return = matches!(
        callee_abi.ret.mode,
        PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
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

    let lowered_args =
        lower_call_operands_with_lowering(fx, args, &callee_abi.args, operand_lowering);
    append_lowered_call_args(
        fx,
        &mut call_args,
        &lowered_args,
        &callee_abi.args,
        callee_abi.c_variadic,
    );

    if callee_abi.c_variadic {
        let sig_ref = fx.bcx.func.dfg.ext_funcs[callee_ref].signature;
        adjust_c_variadic_signature_and_args(
            fx,
            &callee_name,
            sig_ref,
            fixed_param_count,
            &mut call_args,
        );
    } else {
        assert_eq!(
            call_args.len(),
            callee_abi.sig.params.len(),
            "direct call ABI mismatch for {callee_name}: params={} args={} callee={:?} generic_args={:?}",
            callee_abi.sig.params.len(),
            call_args.len(),
            callee_func_id,
            generic_args,
        );
    }

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);
    let call_result = prepare_call_result_cvalue(
        fx,
        call,
        &callee_abi.ret,
        &dest,
        source_return_ty.as_ref(),
        destination,
    );
    store_call_result_and_jump(fx, sret_slot, dest, call_result, destination, target);
}

/// Virtual dispatch: load fn ptr from vtable, call indirectly.
/// Reference: cg_clif/src/vtable.rs `get_ptr_and_method_ref` + cg_clif/src/abi/mod.rs:525-543
fn codegen_virtual_call(
    fx: &mut FunctionCx<'_, impl Module>,
    callee_func_id: hir_def::FunctionId,
    trait_id: TraitId,
    generic_args: GenericArgs<'_>,
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

    let is_rust_call_abi =
        fx.db().function_signature(callee_func_id).abi == Some(sym::rust_dash_call);
    let operand_lowering = CallOperandLowering::from_rust_call_abi(is_rust_call_abi);
    let source_return_ty = callable_output_ty(fx.db(), callee_func_id, generic_args);
    let expected_abi = abi::fn_abi_for_fn_item_from_ty(
        fx.isa,
        fx.db(),
        fx.dl,
        fx.env(),
        callee_func_id,
        generic_args,
    )
    .expect("virtual method ABI");
    let mut expected_abi_sig = expected_abi.sig.clone();
    let receiver_mode = expected_abi
        .args
        .first()
        .map(|arg| arg.mode.clone())
        .expect("virtual call ABI unexpectedly has no receiver argument");
    if let Some(receiver_param_idx) = expected_abi_sig
        .params
        .iter()
        .position(|param| param.purpose != ArgumentPurpose::StructReturn)
    {
        // Vtable methods take `self` as a thin data pointer.
        expected_abi_sig.params[receiver_param_idx].value_type = fx.pointer_type;

        // If the receiver ABI was wide (pair/unsized indirect), drop metadata lane.
        if matches!(
            receiver_mode,
            PassMode::Pair(_, _)
                | PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ }
        ) {
            expected_abi_sig.params.remove(receiver_param_idx + 1);
        }
    }
    let lowered_args =
        lower_call_operands_with_lowering(fx, args, &expected_abi.args, operand_lowering);
    let self_arg = lowered_args.first().expect("virtual call requires receiver argument");
    assert!(
        !expected_abi.args.is_empty(),
        "virtual call ABI unexpectedly has no receiver argument",
    );

    // Get self arg (&dyn Trait = ScalarPair(data_ptr, vtable_ptr))
    let self_cval = self_arg.cval.clone();
    let (data_ptr, vtable_ptr) = self_cval.load_scalar_pair(fx);

    // Load fn ptr from vtable
    let fn_ptr =
        fx.bcx.ins().load(fx.pointer_type, vtable_memflags(), vtable_ptr, vtable_offset as i32);

    let dest = codegen_place(fx, destination);
    let is_sret_return = matches!(
        expected_abi.ret.mode,
        PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
    );

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
    append_lowered_call_args(
        fx,
        &mut call_args,
        &lowered_args[1..],
        &expected_abi.args[1..],
        false,
    );

    assert_eq!(
        call_args.len(),
        expected_abi_sig.params.len(),
        "virtual call ABI mismatch: params={} args={} callee={:?}",
        expected_abi_sig.params.len(),
        call_args.len(),
        callee_func_id,
    );

    // Emit indirect call
    let sig_ref = fx.bcx.import_signature(expected_abi_sig);
    let call = fx.bcx.ins().call_indirect(sig_ref, fn_ptr, &call_args);
    let call_result = prepare_call_result_cvalue(
        fx,
        call,
        &expected_abi.ret,
        &dest,
        source_return_ty.as_ref(),
        destination,
    );
    store_call_result_and_jump(fx, sret_slot, dest, call_result, destination, target);
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
            let pointee_layout = fx
                .db()
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
            let pointee_layout = fx
                .db()
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
        "align_of" | "min_align_of" | "pref_align_of" => {
            let layout = generic_ty_layout.clone().expect("align_of: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64))
        }
        "size_of_val" => {
            assert_eq!(args.len(), 1, "size_of_val expects 1 arg");

            let pointee_ty = generic_ty.clone().expect("size_of_val requires a generic arg");
            let pointee_layout = generic_ty_layout.clone().expect("size_of_val: layout error");
            let ptr = codegen_operand(fx, &args[0].kind);
            let metadata = match ptr.layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => Some(ptr.load_scalar_pair(fx).1),
                _ => None,
            };

            let (size, _) = size_and_align_of_pointee(fx, pointee_ty, &pointee_layout, metadata);
            Some(size)
        }
        "min_align_of_val" => {
            let layout = generic_ty_layout.clone().expect("min_align_of_val: layout error");
            Some(fx.bcx.ins().iconst(fx.pointer_type, layout.align.abi.bytes() as i64))
        }
        "align_of_val" => {
            assert_eq!(args.len(), 1, "align_of_val expects 1 arg");

            let pointee_ty = generic_ty.clone().expect("align_of_val requires a generic arg");
            let pointee_layout = generic_ty_layout.clone().expect("align_of_val: layout error");
            let ptr = codegen_operand(fx, &args[0].kind);
            let metadata = match ptr.layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => Some(ptr.load_scalar_pair(fx).1),
                _ => None,
            };

            let (_, align) = size_and_align_of_pointee(fx, pointee_ty, &pointee_layout, metadata);
            Some(align)
        }
        "needs_drop" => {
            let generic_ty = generic_ty.as_ref().expect("needs_drop requires a generic arg");
            let interner = DbInterner::new_with(fx.db(), fx.local_crate());
            let result = hir_ty::drop::has_drop_glue_mono(interner, generic_ty.as_ref());
            Some(fx.bcx.ins().iconst(types::I8, i64::from(result)))
        }
        "ptr_metadata" => {
            assert_eq!(args.len(), 1, "ptr_metadata expects 1 arg");
            let ptr = codegen_operand(fx, &args[0].kind);
            match ptr.layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => {
                    let (_, metadata) = ptr.load_scalar_pair(fx);
                    Some(metadata)
                }
                // Thin pointers have `()` metadata; destination is ZST so no write needed.
                _ => None,
            }
        }
        "discriminant_value" => {
            assert_eq!(args.len(), 1, "discriminant_value expects 1 arg");

            let enum_ty = generic_ty.expect("discriminant_value requires a generic arg");
            let enum_layout =
                generic_ty_layout.clone().expect("discriminant_value generic layout error");
            let dest_layout = codegen_place(fx, destination).layout.clone();
            let BackendRepr::Scalar(dest_scalar) = dest_layout.backend_repr else {
                panic!("discriminant_value destination must be scalar");
            };
            let dest_ty = scalar_to_clif_type(fx.dl, &dest_scalar);

            if !matches!(enum_ty.as_ref().kind(), TyKind::Adt(adt, _) if matches!(adt.inner().id, hir_def::AdtId::EnumId(_)))
            {
                Some(iconst_from_bits(&mut fx.bcx, dest_ty, 0))
            } else {
                let enum_ref = codegen_operand(fx, &args[0].kind);
                let enum_ptr = match enum_ref.layout.backend_repr {
                    BackendRepr::Scalar(_) => enum_ref.load_scalar(fx),
                    _ => panic!("discriminant_value expects a thin &T receiver"),
                };
                let enum_place = CPlace::for_ptr(pointer::Pointer::new(enum_ptr), enum_layout);
                Some(codegen_get_discriminant(fx, &enum_place, &enum_ty, &dest_layout))
            }
        }
        "aggregate_raw_ptr" => {
            assert_eq!(args.len(), 2, "aggregate_raw_ptr expects 2 args");

            let data = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let dest = codegen_place(fx, destination);
            match dest.layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => {
                    let metadata = codegen_operand(fx, &args[1].kind).load_scalar(fx);
                    dest.write_cvalue(fx, CValue::by_val_pair(data, metadata, dest.layout.clone()));
                }
                BackendRepr::Scalar(_) => {
                    dest.write_cvalue(fx, CValue::by_val(data, dest.layout.clone()));
                }
                _ if dest.layout.is_zst() => {}
                _ => panic!("aggregate_raw_ptr destination has unexpected repr"),
            }

            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "slice_get_unchecked" | "slice_get_unchecked_mut" => {
            assert_eq!(args.len(), 2, "{name} expects 2 args");

            let slice = codegen_operand(fx, &args[0].kind);
            let index = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let data_ptr = match slice.layout.backend_repr {
                BackendRepr::ScalarPair(_, _) => slice.load_scalar_pair(fx).0,
                BackendRepr::Scalar(_) => slice.load_scalar(fx),
                _ => panic!("{name} first argument must be pointer-like"),
            };

            let arg0_ty = operand_ty(fx.db(), body, &args[0].kind);
            let pointee = arg0_ty
                .as_ref()
                .builtin_deref(true)
                .expect("slice_get_unchecked first argument must be pointer/reference");
            let elem_ty = match pointee.kind() {
                TyKind::Slice(elem) | TyKind::Array(elem, _) => elem.store(),
                _ => panic!("{name} first argument pointee must be slice/array, got {pointee:?}"),
            };
            let elem_layout = fx
                .db()
                .layout_of_ty(elem_ty, fx.env().clone())
                .expect("slice_get_unchecked element layout error");

            let index = codegen_intcast(fx, index, fx.pointer_type, false);
            let byte_offset = if elem_layout.size.bytes() == 1 {
                index
            } else {
                fx.bcx.ins().imul_imm(index, elem_layout.size.bytes() as i64)
            };
            Some(fx.bcx.ins().iadd(data_ptr, byte_offset))
        }
        "type_id" => {
            let ty = generic_ty.as_ref().expect("type_id requires a generic arg");
            let ty_dbg = format!("{:?}", ty.as_ref());
            let lo = stable_hash64_with_seed(ty_dbg.as_bytes(), 0xcbf29ce484222325);
            let hi = stable_hash64_with_seed(ty_dbg.as_bytes(), 0x9e3779b185ebca87);
            let type_id_bits = ((hi as u128) << 64) | (lo as u128);
            Some(iconst_from_bits(&mut fx.bcx, types::I128, type_id_bits))
        }
        "type_name" => {
            // Requires emitting a stable &'static str constant.
            return false;
        }
        "ub_checks" => {
            // `cfg(ub_checks)` is unstable in this crate. For our std-JIT harness,
            // treat this as disabled by default.
            Some(fx.bcx.ins().iconst(types::I8, 0))
        }
        "overflow_checks" => {
            // Mirrors `core::intrinsics::overflow_checks` semantics.
            Some(fx.bcx.ins().iconst(types::I8, i64::from(cfg!(debug_assertions))))
        }
        "const_eval_select" => {
            assert_eq!(args.len(), 3, "const_eval_select expects 3 args");

            let runtime_ty = operand_ty(fx.db(), body, &args[2].kind);
            let (runtime_func_id, runtime_generic_args) = match runtime_ty.as_ref().kind() {
                TyKind::FnDef(def, runtime_generic_args) => {
                    let CallableDefId::FunctionId(runtime_func_id) = def.0 else {
                        panic!("const_eval_select runtime callable must be a function")
                    };
                    (runtime_func_id, runtime_generic_args)
                }
                _ => panic!("unsupported const_eval_select runtime callable: {:?}", runtime_ty),
            };

            // `const_eval_select` passes captured arguments as a tuple in `args[0]`.
            // Runtime callables expect those as regular function parameters, so unpack.
            let tuple_ty = operand_ty(fx.db(), body, &args[0].kind);
            let TyKind::Tuple(tuple_fields) = tuple_ty.as_ref().kind() else {
                panic!("const_eval_select ARG must be a tuple, got: {:?}", tuple_ty)
            };
            let tuple_fields: Vec<_> =
                tuple_fields.iter().map(|field_ty| field_ty.store()).collect();

            let tuple_cval = codegen_operand(fx, &args[0].kind);
            let tuple_layout = tuple_cval.layout.clone();
            let tuple_ptr = tuple_cval.force_stack(fx);
            let mut runtime_args = Vec::with_capacity(tuple_fields.len());
            for (idx, field_ty) in tuple_fields.iter().enumerate() {
                let field_layout = fx
                    .db()
                    .layout_of_ty(field_ty.clone(), fx.env().clone())
                    .expect("tuple field layout");
                if field_layout.is_zst() {
                    runtime_args.push(CValue::zst(field_layout));
                    continue;
                }
                let field_offset = tuple_layout.fields.offset(idx).bytes() as i64;
                let field_ptr = tuple_ptr.offset_i64(&mut fx.bcx, fx.pointer_type, field_offset);
                runtime_args.push(CValue::by_ref(field_ptr, field_layout));
            }

            // Declare the runtime callable similarly to normal direct calls.
            let is_extern =
                matches!(runtime_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
            let is_cross_crate = runtime_func_id.krate(fx.db()) != fx.local_crate();
            let interner = DbInterner::new_no_crate(fx.db());
            let empty_args = GenericArgs::empty(interner);
            let (callee_abi, callee_name) = if is_extern {
                let fn_abi = abi::fn_abi_for_fn_item_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    runtime_func_id,
                    empty_args,
                )
                .expect("const_eval_select extern fn ABI");
                let name = extern_fn_symbol_name(fx.db(), runtime_func_id);
                (fn_abi, name)
            } else if let Ok(callee_body) = fx.db().monomorphized_mir_body(
                runtime_func_id.into(),
                runtime_generic_args.store(),
                fx.env().clone(),
            ) {
                let fn_abi = abi::fn_abi_for_body(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body)
                    .expect("callee ABI");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    runtime_func_id,
                    runtime_generic_args,
                    fx.ext_crate_disambiguators(),
                );
                (fn_abi, name)
            } else if is_cross_crate {
                let fn_abi = abi::fn_abi_for_fn_item_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    runtime_func_id,
                    runtime_generic_args,
                )
                .expect("const_eval_select cross-crate fn ABI");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    runtime_func_id,
                    runtime_generic_args,
                    fx.ext_crate_disambiguators(),
                );
                (fn_abi, name)
            } else {
                panic!(
                    "failed to get const_eval_select runtime callee MIR for {runtime_func_id:?}"
                );
            };

            let callee_id = fx
                .module
                .declare_function(&callee_name, Linkage::Import, &callee_abi.sig)
                .expect("declare const_eval_select runtime callee");
            let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

            let dest = codegen_place(fx, destination);
            let is_sret_return = matches!(
                callee_abi.ret.mode,
                PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
            );

            let mut call_args = Vec::new();
            let sret_slot = if is_sret_return {
                let slot = CPlace::new_stack_slot(fx, dest.layout.clone());
                let ptr = slot.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
                call_args.push(ptr);
                Some(slot)
            } else {
                None
            };

            assert_eq!(
                runtime_args.len(),
                callee_abi.args.len(),
                "const_eval_select runtime arg ABI mismatch: args={} abi={}",
                runtime_args.len(),
                callee_abi.args.len(),
            );
            for (arg, arg_abi) in runtime_args.into_iter().zip(callee_abi.args.iter()) {
                call_args.extend(abi::pass_mode::adjust_arg_for_abi(fx, arg, arg_abi, false));
            }

            let call = fx.bcx.ins().call(callee_ref, &call_args);
            let call_result =
                prepare_call_result_cvalue(fx, call, &callee_abi.ret, &dest, None, destination);
            store_call_result_and_jump(fx, sret_slot, dest, call_result, destination, target);
            return true;
        }

        // --- memory operations ---
        "copy_nonoverlapping" => {
            assert_eq!(args.len(), 3, "copy_nonoverlapping expects 3 args");
            let src = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let dst = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            let layout = generic_ty_layout.clone().expect("copy_nonoverlapping: layout error");
            let elem_size = layout.size.bytes();
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };
            let tc = fx.module.target_config();
            fx.bcx.call_memcpy(tc, dst, src, byte_amount);
            None
        }
        "typed_swap_nonoverlapping" => {
            assert_eq!(args.len(), 2, "typed_swap_nonoverlapping expects 2 args");
            let x_ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let y_ptr = codegen_operand(fx, &args[1].kind).load_scalar(fx);

            let layout =
                generic_ty_layout.clone().expect("typed_swap_nonoverlapping: layout error");
            if !layout.is_zst() {
                // Keep this byte-oriented to avoid representation-specific loads/stores
                // when swapping larger memory-repr values (e.g. hashbrown internals).
                let byte_count = fx.bcx.ins().iconst(fx.pointer_type, layout.size.bytes() as i64);
                let tmp = CPlace::new_stack_slot(fx, layout.clone());
                let tmp_ptr = tmp.to_ptr().get_addr(&mut fx.bcx, fx.pointer_type);
                let tc = fx.module.target_config();

                fx.bcx.call_memcpy(tc, tmp_ptr, x_ptr, byte_count);
                fx.bcx.call_memcpy(tc, x_ptr, y_ptr, byte_count);
                fx.bcx.call_memcpy(tc, y_ptr, tmp_ptr, byte_count);
            }
            None
        }
        "copy" => {
            assert_eq!(args.len(), 3, "copy expects 3 args");
            let src = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let dst = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            let layout = generic_ty_layout.clone().expect("copy: layout error");
            let elem_size = layout.size.bytes();
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };
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
            let byte_amount =
                if elem_size != 1 { fx.bcx.ins().imul_imm(count, elem_size as i64) } else { count };
            let tc = fx.module.target_config();
            fx.bcx.call_memset(tc, dst, val, byte_amount);
            None
        }
        "compare_bytes" => {
            assert_eq!(args.len(), 3, "compare_bytes expects 3 args");
            let lhs_ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let rhs_ptr = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let byte_count = codegen_operand(fx, &args[2].kind).load_scalar(fx);

            // Match upstream cg_clif behavior: lower to a memcmp libcall.
            Some(codegen_libcall1(
                fx,
                "memcmp",
                &[fx.pointer_type, fx.pointer_type, fx.pointer_type],
                types::I32,
                &[lhs_ptr, rhs_ptr, byte_count],
            ))
        }
        "three_way_compare" => {
            assert_eq!(args.len(), 2, "three_way_compare expects 2 args");

            let lhs = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let rhs = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            let signed = matches!(arg_ty.as_ref().kind(), TyKind::Int(_));

            let lt_cc = if signed { IntCC::SignedLessThan } else { IntCC::UnsignedLessThan };
            let gt_cc = if signed { IntCC::SignedGreaterThan } else { IntCC::UnsignedGreaterThan };
            let is_lt = fx.bcx.ins().icmp(lt_cc, lhs, rhs);
            let is_gt = fx.bcx.ins().icmp(gt_cc, lhs, rhs);

            let dest = codegen_place(fx, destination);
            let BackendRepr::Scalar(order_scalar) = dest.layout.backend_repr else {
                panic!("three_way_compare destination must be scalar")
            };
            let order_ty = scalar_to_clif_type(fx.dl, &order_scalar);
            let less = fx.bcx.ins().iconst(order_ty, -1);
            let equal = fx.bcx.ins().iconst(order_ty, 0);
            let greater = fx.bcx.ins().iconst(order_ty, 1);
            let non_lt = fx.bcx.ins().select(is_gt, greater, equal);
            let ordering = fx.bcx.ins().select(is_lt, less, non_lt);

            dest.write_cvalue(fx, CValue::by_val(ordering, dest.layout.clone()));
            if target.is_some()
                && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
            {
                fx.set_drop_flag(destination.local);
            }
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "read_via_copy" => {
            assert_eq!(args.len(), 1, "read_via_copy expects 1 arg");
            let src_ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let src_layout = generic_ty_layout.clone().expect("read_via_copy: layout error");
            let src = CValue::by_ref(pointer::Pointer::new(src_ptr), src_layout);
            let dest = codegen_place(fx, destination);
            dest.write_cvalue(fx, src);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "write_via_move" => {
            assert_eq!(args.len(), 2, "write_via_move expects 2 args");
            let dst_ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let value = codegen_operand(fx, &args[1].kind);
            let dst = CPlace::for_ptr(pointer::Pointer::new(dst_ptr), value.layout.clone());
            dst.write_cvalue(fx, value);
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "box_new" => {
            assert_eq!(args.len(), 1, "box_new expects 1 arg");

            let pointee_layout = generic_ty_layout.clone().expect("box_new: layout error");
            let value = codegen_operand(fx, &args[0].kind);

            let ptr = codegen_box_alloc(fx, &pointee_layout);
            let heap_place = CPlace::for_ptr(pointer::Pointer::new(ptr), pointee_layout);
            heap_place.write_cvalue(fx, value);

            Some(ptr)
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
        "assume"
        | "assert_inhabited"
        | "assert_zero_valid"
        | "assert_mem_uninitialized_valid"
        | "cold_path" => {
            None // no-op
        }
        "likely" | "unlikely" => {
            assert_eq!(args.len(), 1);
            let val = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            Some(val) // pass through
        }
        "select_unpredictable" => {
            assert_eq!(args.len(), 3, "select_unpredictable expects 3 args");

            let cond_raw = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let cond_ty = fx.bcx.func.dfg.value_type(cond_raw);
            let cond = if cond_ty.is_int() {
                fx.bcx.ins().icmp_imm(IntCC::NotEqual, cond_raw, 0)
            } else {
                cond_raw
            };

            let if_true = codegen_operand(fx, &args[1].kind);
            let if_false = codegen_operand(fx, &args[2].kind);
            let dest = codegen_place(fx, destination);

            if !dest.layout.is_zst() {
                let true_ptr = if_true.force_stack(fx).get_addr(&mut fx.bcx, fx.pointer_type);
                let false_ptr = if_false.force_stack(fx).get_addr(&mut fx.bcx, fx.pointer_type);
                let selected_ptr = fx.bcx.ins().select(cond, true_ptr, false_ptr);
                let selected = CValue::by_ref(pointer::Pointer::new(selected_ptr), dest.layout.clone());
                dest.write_cvalue(fx, selected);
            }

            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "is_val_statically_known" => {
            assert_eq!(args.len(), 1);
            // This intrinsic is explicitly non-deterministic; returning false
            // is always semantically valid and avoids leaving unresolved imports.
            Some(fx.bcx.ins().iconst(types::I8, 0))
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
        "saturating_add" | "saturating_sub" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let lhs = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let rhs = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let arg_ty = operand_ty(fx.db(), body, &args[0].kind);
            if !matches!(arg_ty.as_ref().kind(), TyKind::Int(_) | TyKind::Uint(_)) {
                panic!("{name} intrinsic expects integer arguments, got {arg_ty:?}");
            }

            let signed = ty_is_signed_int(arg_ty);
            let op = if name == "saturating_add" { BinOp::Add } else { BinOp::Sub };
            Some(codegen_saturating_int_binop(fx, &op, lhs, rhs, signed))
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
        "add_with_overflow" | "sub_with_overflow" | "mul_with_overflow" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let lhs = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let rhs = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let signed = ty_is_signed_int(operand_ty(fx.db(), body, &args[0].kind));

            let op = match name {
                "add_with_overflow" => BinOp::AddWithOverflow,
                "sub_with_overflow" => BinOp::SubWithOverflow,
                "mul_with_overflow" => BinOp::MulWithOverflow,
                _ => unreachable!(),
            };

            let (result, has_overflow) = codegen_checked_int_binop(fx, &op, lhs, rhs, signed);
            let dest = codegen_place(fx, destination);
            let BackendRepr::ScalarPair(_, overflow_scalar) = dest.layout.backend_repr else {
                panic!("{name} intrinsic must return ScalarPair");
            };
            let overflow_ty = scalar_to_clif_type(fx.dl, &overflow_scalar);
            let has_overflow = if fx.bcx.func.dfg.value_type(has_overflow) == overflow_ty {
                has_overflow
            } else {
                let one = fx.bcx.ins().iconst(overflow_ty, 1);
                let zero = fx.bcx.ins().iconst(overflow_ty, 0);
                fx.bcx.ins().select(has_overflow, one, zero)
            };

            dest.write_cvalue(fx, CValue::by_val_pair(result, has_overflow, dest.layout.clone()));
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
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

        // --- abort / unreachable ---
        "abort" | "unreachable" => {
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
            if destination.projection.lookup(&fx.ra_body().projection_store).is_empty() {
                fx.set_drop_flag(destination.local);
            }
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }

        // --- atomics ---
        _ if name == "atomic_fence"
            || name.starts_with("atomic_fence_")
            || name == "atomic_singlethreadfence"
            || name.starts_with("atomic_singlethreadfence_") =>
        {
            assert_eq!(args.len(), 0, "{name} intrinsic expects 0 args");
            fx.bcx.ins().fence();
            None
        }
        "atomic_load" => {
            assert_eq!(args.len(), 1, "atomic_load intrinsic expects 1 arg");

            let ty = generic_ty.expect("atomic_load requires a type generic arg");
            if !ty_is_atomic_int_or_ptr(&ty) {
                panic!("atomic_load requires integer or raw pointer type, got {:?}", ty);
            }
            let layout = generic_ty_layout.clone().expect("atomic_load: layout error");
            let clif_ty = atomic_scalar_clif_ty(fx.dl, &layout);
            if clif_ty == types::I128 {
                panic!("128-bit atomics are not yet supported");
            }

            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            Some(fx.bcx.ins().atomic_load(clif_ty, MemFlags::trusted(), ptr))
        }
        "atomic_store" => {
            assert_eq!(args.len(), 2, "atomic_store intrinsic expects 2 args");

            let ty = generic_ty.expect("atomic_store requires a type generic arg");
            if !ty_is_atomic_int_or_ptr(&ty) {
                panic!("atomic_store requires integer or raw pointer type, got {:?}", ty);
            }
            let layout = generic_ty_layout.clone().expect("atomic_store: layout error");
            let clif_ty = atomic_scalar_clif_ty(fx.dl, &layout);
            if clif_ty == types::I128 {
                panic!("128-bit atomics are not yet supported");
            }

            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let val = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            if fx.bcx.func.dfg.value_type(val) != clif_ty {
                panic!(
                    "atomic_store value type mismatch: expected {clif_ty:?}, got {:?}",
                    fx.bcx.func.dfg.value_type(val)
                );
            }

            fx.bcx.ins().atomic_store(MemFlags::trusted(), val, ptr);
            None
        }
        "atomic_xchg" => {
            assert_eq!(args.len(), 2, "atomic_xchg intrinsic expects 2 args");

            let ty = generic_ty.expect("atomic_xchg requires a type generic arg");
            if !ty_is_atomic_int_or_ptr(&ty) {
                panic!("atomic_xchg requires integer or raw pointer type, got {:?}", ty);
            }
            let layout = generic_ty_layout.clone().expect("atomic_xchg: layout error");
            let clif_ty = atomic_scalar_clif_ty(fx.dl, &layout);
            if clif_ty == types::I128 {
                panic!("128-bit atomics are not yet supported");
            }

            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let new = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            if fx.bcx.func.dfg.value_type(new) != clif_ty {
                panic!(
                    "atomic_xchg value type mismatch: expected {clif_ty:?}, got {:?}",
                    fx.bcx.func.dfg.value_type(new)
                );
            }

            Some(fx.bcx.ins().atomic_rmw(clif_ty, MemFlags::trusted(), AtomicRmwOp::Xchg, ptr, new))
        }
        "atomic_cxchg" | "atomic_cxchgweak" => {
            assert_eq!(args.len(), 3, "{name} intrinsic expects 3 args");

            let ty = generic_ty.expect("atomic_cxchg requires a type generic arg");
            if !ty_is_atomic_int_or_ptr(&ty) {
                panic!("{name} requires integer or raw pointer type, got {:?}", ty);
            }
            let layout = generic_ty_layout.clone().expect("atomic_cxchg: layout error");
            let clif_ty = atomic_scalar_clif_ty(fx.dl, &layout);
            if clif_ty == types::I128 {
                panic!("128-bit atomics are not yet supported");
            }

            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let test_old = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            let new = codegen_operand(fx, &args[2].kind).load_scalar(fx);
            if fx.bcx.func.dfg.value_type(test_old) != clif_ty {
                panic!(
                    "{name} expected compare value type {clif_ty:?}, got {:?}",
                    fx.bcx.func.dfg.value_type(test_old)
                );
            }
            if fx.bcx.func.dfg.value_type(new) != clif_ty {
                panic!(
                    "{name} expected new value type {clif_ty:?}, got {:?}",
                    fx.bcx.func.dfg.value_type(new)
                );
            }

            let old = fx.bcx.ins().atomic_cas(MemFlags::trusted(), ptr, test_old, new);
            let is_eq = fx.bcx.ins().icmp(IntCC::Equal, old, test_old);

            let dest = codegen_place(fx, destination);
            let BackendRepr::ScalarPair(_, success_scalar) = dest.layout.backend_repr else {
                panic!("{name} intrinsic must return ScalarPair");
            };
            let success_ty = scalar_to_clif_type(fx.dl, &success_scalar);
            let success = if fx.bcx.func.dfg.value_type(is_eq) == success_ty {
                is_eq
            } else {
                let one = fx.bcx.ins().iconst(success_ty, 1);
                let zero = fx.bcx.ins().iconst(success_ty, 0);
                fx.bcx.ins().select(is_eq, one, zero)
            };

            dest.write_cvalue(fx, CValue::by_val_pair(old, success, dest.layout.clone()));
            if target.is_some()
                && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
            {
                fx.set_drop_flag(destination.local);
            }
            if let Some(target) = target {
                let block = fx.clif_block(*target);
                fx.bcx.ins().jump(block, &[]);
            }
            return true;
        }
        "atomic_xadd" | "atomic_xsub" | "atomic_and" | "atomic_or" | "atomic_xor"
        | "atomic_nand" | "atomic_max" | "atomic_umax" | "atomic_min" | "atomic_umin" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let ty = generic_ty.expect("atomic rmw intrinsic requires a type generic arg");
            let valid_ty = match name {
                "atomic_xadd" | "atomic_xsub" | "atomic_and" | "atomic_or" | "atomic_xor"
                | "atomic_nand" => ty_is_atomic_int(&ty),
                "atomic_max" | "atomic_min" => ty_is_signed_int(ty.clone()),
                "atomic_umax" | "atomic_umin" => ty_is_atomic_unsigned_int(&ty),
                _ => unreachable!(),
            };
            if !valid_ty {
                panic!("{name} received unsupported type {:?}", ty);
            }

            let layout = generic_ty_layout.clone().expect("atomic rmw: layout error");
            let clif_ty = atomic_scalar_clif_ty(fx.dl, &layout);
            if clif_ty == types::I128 {
                panic!("128-bit atomics are not yet supported");
            }

            let ptr = codegen_operand(fx, &args[0].kind).load_scalar(fx);
            let src = codegen_operand(fx, &args[1].kind).load_scalar(fx);
            if fx.bcx.func.dfg.value_type(src) != clif_ty {
                panic!(
                    "{name} value type mismatch: expected {clif_ty:?}, got {:?}",
                    fx.bcx.func.dfg.value_type(src)
                );
            }

            let op = match name {
                "atomic_xadd" => AtomicRmwOp::Add,
                "atomic_xsub" => AtomicRmwOp::Sub,
                "atomic_and" => AtomicRmwOp::And,
                "atomic_or" => AtomicRmwOp::Or,
                "atomic_xor" => AtomicRmwOp::Xor,
                "atomic_nand" => AtomicRmwOp::Nand,
                "atomic_max" => AtomicRmwOp::Smax,
                "atomic_umax" => AtomicRmwOp::Umax,
                "atomic_min" => AtomicRmwOp::Smin,
                "atomic_umin" => AtomicRmwOp::Umin,
                _ => unreachable!(),
            };

            Some(fx.bcx.ins().atomic_rmw(clif_ty, MemFlags::trusted(), op, ptr, src))
        }

        _ => {
            return false;
        }
    };

    // Write result to destination
    let dest = codegen_place(fx, destination);
    if let Some(val) = result {
        if !dest.layout.is_zst() {
            dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
        }
    }

    if target.is_some() && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
    {
        fx.set_drop_flag(destination.local);
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
    flags_builder.set("enable_llvm_abi_extensions", "true").unwrap();
    let flags = settings::Flags::new(flags_builder);

    let isa_builder = cranelift_native::builder().expect("host ISA not supported");
    isa_builder.finish(flags).expect("failed to build ISA")
}

/// Build a Cranelift signature from a MIR body's locals (return type + params).
pub fn build_fn_sig(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
) -> Result<Signature, String> {
    let fn_abi = abi::fn_abi_for_body(isa, db, dl, env, body)?;
    Ok(fn_abi.sig)
}

/// Build a Cranelift signature from a function's type information (via `callable_item_signature`)
/// instead of from a MIR body. Needed for extern and cross-crate functions where we don't have MIR.
///
/// For generic cross-crate functions, pass the concrete generic args to substitute
/// type parameters. For non-generic functions, pass empty args.
fn build_fn_sig_from_ty(
    isa: &dyn TargetIsa,
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
) -> Result<Signature, String> {
    let fn_abi = abi::fn_abi_for_fn_item_from_ty(isa, db, dl, env, func_id, generic_args)?;
    Ok(fn_abi.sig)
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
) -> Result<FuncId, String> {
    let pointer_type = pointer_ty(dl);
    let fn_abi = abi::fn_abi_for_body(isa, db, dl, env, body)?;
    let sig = fn_abi.sig.clone();

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
                    let size =
                        u32::try_from(local_layout.size.bytes()).expect("stack slot too large");
                    let align_shift = {
                        let a = local_layout.align.abi.bytes();
                        assert!(a.is_power_of_two());
                        a.trailing_zeros() as u8
                    };
                    let slot =
                        bcx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            size,
                            align_shift,
                        ));
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
                _ if local_layout.is_zst() => CPlace::for_ptr(
                    pointer::Pointer::dangling(local_layout.align.abi),
                    local_layout,
                ),
                _ => {
                    // Non-scalar, non-ZST: allocate stack slot
                    let size =
                        u32::try_from(local_layout.size.bytes()).expect("stack slot too large");
                    let align_shift = {
                        let a = local_layout.align.abi.bytes();
                        assert!(a.is_power_of_two());
                        a.trailing_zeros() as u8
                    };
                    let slot =
                        bcx.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            size,
                            align_shift,
                        ));
                    CPlace::for_ptr(pointer::Pointer::stack_slot(slot), local_layout)
                }
            };
            local_map.push(place);
        }

        // Detect sret (indirect return) from ABI mode.
        let is_sret = matches!(
            fn_abi.ret.mode,
            PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ }
        );

        // Wire function parameters to their locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;

        // If sret, block param 0 is the sret pointer — override return slot
        if is_sret {
            let sret_ptr = block_params[param_idx];
            param_idx += 1;
            let ret_idx = hir_ty::mir::return_slot().into_raw().into_u32() as usize;
            let ret_layout =
                fn_abi.ret.layout.as_ref().expect("sret return ABI missing return layout").clone();
            local_map[ret_idx] = CPlace::for_ptr(pointer::Pointer::new(sret_ptr), ret_layout);
        }

        assert_eq!(
            body.param_locals.len(),
            fn_abi.args.len(),
            "function ABI arg count mismatch for {fn_name}",
        );
        for (&param_local, param_abi) in body.param_locals.iter().zip(fn_abi.args.iter()) {
            let param_idx_local = param_local.into_raw().into_u32() as usize;
            let place = &local_map[param_idx_local];
            match param_abi.mode {
                PassMode::Ignore => {}
                PassMode::Direct(_) => {
                    let param_val = block_params[param_idx];
                    param_idx += 1;
                    if place.is_register() {
                        place.def_var(0, param_val, &mut bcx);
                    } else {
                        // Address-taken scalar: store parameter into stack slot.
                        let mut flags = MemFlags::new();
                        flags.set_notrap();
                        place.to_ptr().store(&mut bcx, param_val, flags);
                    }
                }
                PassMode::Pair(_, _) => {
                    let val0 = block_params[param_idx];
                    let val1 = block_params[param_idx + 1];
                    param_idx += 2;
                    if place.is_register() {
                        place.def_var(0, val0, &mut bcx);
                        place.def_var(1, val1, &mut bcx);
                    } else {
                        // Address-taken scalar pair: store both parts into stack slot
                        let mut flags = MemFlags::new();
                        flags.set_notrap();
                        let ptr = place.to_ptr();
                        ptr.store(&mut bcx, val0, flags);
                        let BackendRepr::ScalarPair(ref a, ref b) = place.layout.backend_repr
                        else {
                            unreachable!()
                        };
                        let b_off = value_and_place::scalar_pair_b_offset(dl, *a, *b);
                        ptr.offset_i64(&mut bcx, pointer_type, b_off).store(&mut bcx, val1, flags);
                    }
                }
                PassMode::Cast { .. } => {
                    panic!("PassMode::Cast params are not supported in function prelude yet");
                }
                PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ } => {
                    panic!("unsized by-value params are not supported yet");
                }
                PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ } => {
                    // Memory-repr param: block param is a pointer to caller-owned data.
                    // Materialize the parameter into the callee local place so
                    // subsequent local mutations don't alias caller storage.
                    let ptr_val = block_params[param_idx];
                    param_idx += 1;
                    let dst = place.to_ptr().get_addr(&mut bcx, pointer_type);
                    let byte_count =
                        bcx.ins().iconst(pointer_type, place.layout.size.bytes() as i64);
                    bcx.call_memcpy(module.target_config(), dst, ptr_val, byte_count);
                }
            }
        }
        assert_eq!(
            param_idx,
            block_params.len(),
            "function parameter ABI consumed {} block params, expected {} for {fn_name}",
            param_idx,
            block_params.len(),
        );

        // Create drop flags for locals that have drop glue.
        // r-a's MIR lacks drop elaboration, so Drop terminators fire even
        // for locals that have been moved out. Drop flags (0=dead, 1=live)
        // let codegen_drop skip drops on moved-out locals.
        let mut drop_flags = HashMap::new();
        {
            let interner = DbInterner::new_with(db, local_crate);
            for (local_id, local) in body.locals.iter() {
                let local_ty = &local.ty;
                if hir_ty::drop::has_drop_glue_mono(interner, local_ty.as_ref()) {
                    let var = bcx.declare_var(types::I8);
                    let zero = bcx.ins().iconst(types::I8, 0);
                    bcx.def_var(var, zero);
                    drop_flags.insert(local_id.into_raw().into_u32(), var);
                }
            }
            // Parameters are live on entry.
            for &param_local in &body.param_locals {
                let idx = param_local.into_raw().into_u32();
                if let Some(&var) = drop_flags.get(&idx) {
                    let one = bcx.ins().iconst(types::I8, 1);
                    bcx.def_var(var, one);
                }
            }
        }

        let mut fx = FunctionCx {
            bcx,
            module,
            isa,
            fn_abi: fn_abi.clone(),
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
            drop_flags,
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
    module.define_function(func_id, &mut ctx).map_err(|e| {
        format!("define_function: {e}\n-- function: {fn_name} --\n{}", ctx.func.display(),)
    })?;

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
        &mut module,
        &*isa,
        db,
        dl,
        env,
        body,
        fn_name,
        Linkage::Export,
        local_crate,
        &empty_map,
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
    module.define_function(cmain_func_id, &mut ctx).map_err(|e| format!("define main: {e}"))?;

    Ok(())
}

/// When an unsizing coercion `&T → &dyn Trait` is found, discover the impl
/// methods that will be placed in the vtable and add them to the work queue.
fn collect_unsize_metadata_dependencies(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    local_crate: base_db::Crate,
    source_pointee: StoredTy,
    target_pointee: StoredTy,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    drop_types: &mut std::collections::HashSet<StoredTy>,
) {
    let (source_tail, target_tail) = lockstep_unsized_tails(db, source_pointee, target_pointee);
    let Some(trait_id) = target_tail.as_ref().dyn_trait() else {
        return;
    };

    // dyn->dyn metadata propagation doesn't construct a concrete vtable.
    if matches!(source_tail.as_ref().kind(), TyKind::Dynamic(..)) {
        return;
    }

    if let TyKind::Closure(closure_id, closure_subst) = source_tail.as_ref().kind() {
        let key = (closure_id.0, closure_subst.store());
        if closure_visited.insert(key.clone()) {
            closure_result.push(key);
        }
    }

    // Slot 0 stores drop glue for the concrete source tail.
    collect_drop_info(db, local_crate, &source_tail, queue, drop_types);

    // Slots 3+ store trait method implementations.
    let trait_items = trait_id.trait_items(db);
    for (_trait_method_name, trait_item) in trait_items.items.iter() {
        let AssocItemId::FunctionId(trait_method_func_id) = trait_item else { continue };
        let trait_method_subst =
            trait_method_substs_for_receiver(db, local_crate, *trait_method_func_id, &source_tail);
        if let (Either::Left(impl_func_id), impl_generic_args) =
            db.lookup_impl_method(env.as_ref(), *trait_method_func_id, trait_method_subst)
        {
            queue.push_back((impl_func_id, impl_generic_args.store()));
        }
    }
}

/// When an unsizing coercion `&T → &dyn Trait` is found, discover the impl
/// methods that will be placed in the vtable and add them to the work queue.
fn collect_vtable_methods(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    operand: &Operand,
    target_ty: &StoredTy,
    local_crate: base_db::Crate,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    drop_types: &mut std::collections::HashSet<StoredTy>,
) {
    // Extract the concrete source type
    let from_ty = operand_ty(db, body, &operand.kind);
    let Some(source_pointee) = from_ty.as_ref().builtin_deref(true) else { return };
    let Some(target_pointee) = target_ty.as_ref().builtin_deref(true) else { return };

    collect_unsize_metadata_dependencies(
        db,
        env,
        local_crate,
        source_pointee.store(),
        target_pointee.store(),
        queue,
        closure_visited,
        closure_result,
        drop_types,
    );
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

    let fn_name = symbol_mangling::mangle_drop_in_place(db, ty.as_ref(), ext_crate_disambiguators);
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

                let generic_args = drop_impl_generic_args(db, local_crate, ty);
                let drop_fn_name = symbol_mangling::mangle_function(
                    db,
                    drop_func_id,
                    generic_args.as_ref(),
                    ext_crate_disambiguators,
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
                            for (field_idx, (_, field_ty_binder)) in field_types.iter().enumerate()
                            {
                                let field_ty = field_ty_binder.get().instantiate(interner, subst);
                                if hir_ty::drop::has_drop_glue_mono(interner, field_ty) {
                                    let offset = layout.fields.offset(field_idx).bytes() as i64;
                                    let field_ptr = bcx.ins().iadd_imm(self_ptr, offset);
                                    let field_ty_stored = field_ty.store();
                                    let name = symbol_mangling::mangle_drop_in_place(
                                        db,
                                        field_ty_stored.as_ref(),
                                        ext_crate_disambiguators,
                                    );
                                    let callee = module
                                        .declare_function(&name, Linkage::Import, &field_drop_sig)
                                        .expect("declare field drop_in_place");
                                    let callee_ref = module.declare_func_in_func(callee, bcx.func);
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
    module.define_function(func_id, &mut ctx).map_err(|e| format!("define drop_in_place: {e}"))?;

    Ok(func_id)
}

/// Collect all non-extern, non-intrinsic monomorphized function instances
/// reachable from `root` by walking MIR call terminators. Returns them in BFS
/// order with `root` first. Each entry is a `(FunctionId, StoredGenericArgs)` pair
/// representing a specific monomorphization.
///
/// Also returns:
/// - closures discovered via `AggregateKind::Closure` in statements
/// - types that need `drop_in_place` glue functions
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
    collect_reachable_fns_with_body_filter(db, env, root, local_crate, |_func_id, _generic_args| {
        true
    })
}

/// Same as `collect_reachable_fns`, but allows callers to skip MIR body
/// traversal for selected function instances while still retaining them in
/// the reachable function result list.
fn collect_reachable_fns_with_body_filter<F>(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    root: hir_def::FunctionId,
    local_crate: base_db::Crate,
    mut should_scan_body: F,
) -> (
    Vec<(hir_def::FunctionId, StoredGenericArgs)>,
    Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    Vec<StoredTy>,
)
where
    F: FnMut(hir_def::FunctionId, &StoredGenericArgs) -> bool,
{
    use std::collections::{HashSet, VecDeque};

    let interner = hir_ty::next_solver::DbInterner::new_no_crate(db);
    let empty_args = GenericArgs::empty(interner).store();

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

        // Skip trait methods with no MIR (abstract methods). Keep default
        // trait methods that do have MIR bodies.
        if matches!(func_id.loc(db).container, ItemContainerId::TraitId(_))
            && db.monomorphized_mir_body(func_id.into(), generic_args.clone(), env.clone()).is_err()
        {
            continue;
        }

        result.push((func_id, generic_args.clone()));

        if !should_scan_body(func_id, &generic_args) {
            continue;
        }

        // Get monomorphized MIR and scan for direct callees
        let Ok(body) = db.monomorphized_mir_body(func_id.into(), generic_args, env.clone()) else {
            continue;
        };

        scan_body_for_callees(
            db,
            env,
            &body,
            interner,
            local_crate,
            &mut queue,
            &mut closure_visited,
            &mut closure_result,
            &mut drop_types,
        );
    }

    // Also scan closure bodies for callees and nested closures
    let mut i = 0;
    while i < closure_result.len() {
        let (closure_id, closure_subst) = closure_result[i].clone();
        i += 1;
        let Ok(closure_body) =
            db.monomorphized_mir_body_for_closure(closure_id, closure_subst, env.clone())
        else {
            continue;
        };
        scan_body_for_callees(
            db,
            env,
            &closure_body,
            interner,
            local_crate,
            &mut queue,
            &mut closure_visited,
            &mut closure_result,
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
            if matches!(func_id.loc(db).container, ItemContainerId::TraitId(_))
                && db
                    .monomorphized_mir_body(func_id.into(), generic_args.clone(), env.clone())
                    .is_err()
            {
                continue;
            }
            result.push((func_id, generic_args.clone()));
            if !should_scan_body(func_id, &generic_args) {
                continue;
            }
            let Ok(body) = db.monomorphized_mir_body(func_id.into(), generic_args, env.clone())
            else {
                continue;
            };
            scan_body_for_callees(
                db,
                env,
                &body,
                interner,
                local_crate,
                &mut queue,
                &mut closure_visited,
                &mut closure_result,
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
            let drop_args = drop_impl_generic_args(db, local_crate, ty);
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
                    if db
                        .struct_signature(id)
                        .flags
                        .intersects(StructFlags::IS_MANUALLY_DROP | StructFlags::IS_PHANTOM_DATA)
                    {
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

fn collect_builtin_derive_trait_call_dependency(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    local_crate: base_db::Crate,
    trait_method_func_id: hir_def::FunctionId,
    self_ty: StoredTy,
    include_rhs_self: bool,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
) {
    let method_substs = trait_method_substs_for_derive_call(
        db,
        local_crate,
        trait_method_func_id,
        &self_ty,
        include_rhs_self,
    );
    let (resolved_callable, resolved_args) =
        db.lookup_impl_method(env.as_ref(), trait_method_func_id, method_substs);
    match resolved_callable {
        Either::Left(resolved_id) => {
            queue.push_back((resolved_id, resolved_args.store()));
        }
        Either::Right((derive_impl_id, derive_method)) => {
            collect_builtin_derive_method_dependencies(
                db,
                env,
                local_crate,
                derive_impl_id,
                derive_method,
                self_ty,
                queue,
            );
        }
    }
}

fn collect_builtin_derive_field_trait_calls(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    local_crate: base_db::Crate,
    trait_method_func_id: hir_def::FunctionId,
    self_ty: StoredTy,
    include_rhs_self: bool,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
) {
    let TyKind::Adt(adt_id, adt_args) = self_ty.as_ref().kind() else {
        return;
    };

    let interner = DbInterner::new_with(db, local_crate);
    match adt_id.inner().id {
        hir_def::AdtId::StructId(struct_id) => {
            for (_, field_ty) in db.field_types(struct_id.into()).iter() {
                let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                collect_builtin_derive_trait_call_dependency(
                    db,
                    env,
                    local_crate,
                    trait_method_func_id,
                    field_ty,
                    include_rhs_self,
                    queue,
                );
            }
        }
        hir_def::AdtId::UnionId(_) => {}
        hir_def::AdtId::EnumId(enum_id) => {
            for &(variant_id, _, _) in &enum_id.enum_variants(db).variants {
                for (_, field_ty) in db.field_types(variant_id.into()).iter() {
                    let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                    collect_builtin_derive_trait_call_dependency(
                        db,
                        env,
                        local_crate,
                        trait_method_func_id,
                        field_ty,
                        include_rhs_self,
                        queue,
                    );
                }
            }
        }
    }
}

fn collect_builtin_derive_method_dependencies(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    local_crate: base_db::Crate,
    derive_impl_id: BuiltinDeriveImplId,
    derive_method: BuiltinDeriveImplMethod,
    self_ty: StoredTy,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
) {
    let Some(trait_method_func_id) = derive_method.trait_method(db, derive_impl_id) else {
        return;
    };

    match derive_method {
        BuiltinDeriveImplMethod::default => {
            let TyKind::Adt(adt_id, adt_args) = self_ty.as_ref().kind() else {
                return;
            };

            let interner = DbInterner::new_with(db, local_crate);
            match adt_id.inner().id {
                hir_def::AdtId::StructId(struct_id) => {
                    for (_, field_ty) in db.field_types(struct_id.into()).iter() {
                        let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                        collect_builtin_derive_trait_call_dependency(
                            db,
                            env,
                            local_crate,
                            trait_method_func_id,
                            field_ty,
                            false,
                            queue,
                        );
                    }
                }
                hir_def::AdtId::EnumId(enum_id) => {
                    let variants = &enum_id.enum_variants(db).variants;
                    let default_variant_idx = enum_default_variant_idx(db, enum_id);
                    let default_variant_id = variants[default_variant_idx.as_usize()].0;
                    for (_, field_ty) in db.field_types(default_variant_id.into()).iter() {
                        let field_ty = field_ty.get().instantiate(interner, adt_args).store();
                        collect_builtin_derive_trait_call_dependency(
                            db,
                            env,
                            local_crate,
                            trait_method_func_id,
                            field_ty,
                            false,
                            queue,
                        );
                    }
                }
                hir_def::AdtId::UnionId(_) => {}
            }
        }
        BuiltinDeriveImplMethod::clone => {
            collect_builtin_derive_field_trait_calls(
                db,
                env,
                local_crate,
                trait_method_func_id,
                self_ty,
                false,
                queue,
            );
        }
        BuiltinDeriveImplMethod::eq | BuiltinDeriveImplMethod::partial_cmp => {
            collect_builtin_derive_field_trait_calls(
                db,
                env,
                local_crate,
                trait_method_func_id,
                self_ty,
                true,
                queue,
            );
        }
        BuiltinDeriveImplMethod::fmt => {}
        _ => {}
    }
}

fn const_addr_points_to_alloc(memory_map: &hir_ty::MemoryMap<'_>, addr: usize) -> bool {
    match memory_map {
        hir_ty::MemoryMap::Empty => false,
        hir_ty::MemoryMap::Simple(data) => addr <= data.len(),
        hir_ty::MemoryMap::Complex(cm) => cm.memory_iter().any(|(base, bytes)| {
            let Some(end) = base.checked_add(bytes.len()) else {
                return false;
            };
            addr >= *base && addr <= end
        }),
    }
}

fn read_usize_le(bytes: &[u8], offset: usize) -> Option<usize> {
    let ptr_size = std::mem::size_of::<usize>();
    let end = offset.checked_add(ptr_size)?;
    let raw = bytes.get(offset..end)?;
    Some(match ptr_size {
        8 => {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(raw);
            u64::from_le_bytes(buf) as usize
        }
        4 => {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(raw);
            u32::from_le_bytes(buf) as usize
        }
        _ => unreachable!("unsupported pointer width"),
    })
}

fn enqueue_callable_from_mapped_ty(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    interner: DbInterner,
    mapped_ty: hir_ty::next_solver::Ty<'_>,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
) {
    match mapped_ty.kind() {
        TyKind::FnDef(def, callee_args) => {
            let CallableDefId::FunctionId(mut callee_id) = def.0 else {
                return;
            };
            let mut resolved_args = callee_args;

            if let ItemContainerId::TraitId(_) = callee_id.loc(db).container
                && hir_ty::method_resolution::is_dyn_method(
                    interner,
                    env.param_env(),
                    callee_id,
                    callee_args,
                )
                .is_none()
            {
                match db.lookup_impl_method(env.as_ref(), callee_id, callee_args) {
                    (Either::Left(impl_id), impl_args) => {
                        callee_id = impl_id;
                        resolved_args = impl_args;
                    }
                    (Either::Right(_), _) => return,
                }
            }

            queue.push_back((callee_id, resolved_args.store()));
        }
        TyKind::Closure(closure_id, closure_subst) => {
            let key = (closure_id.0, closure_subst.store());
            if closure_visited.insert(key.clone()) {
                closure_result.push(key);
            }
        }
        _ => {}
    }
}

fn collect_const_operand_callables(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    interner: DbInterner,
    operand: &Operand,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
) {
    let OperandKind::Constant { konst, ty } = &operand.kind else {
        return;
    };
    let Ok(layout) = db.layout_of_ty(ty.clone(), env.clone()) else {
        return;
    };
    let value = resolve_const_value(db, konst.as_ref());
    let const_bytes = value.value.inner();
    let memory_map = &const_bytes.memory_map;

    let mut maybe_collect = |offset: usize| {
        let Some(addr) = read_usize_le(&const_bytes.memory, offset) else {
            return;
        };
        if addr == 0 || const_addr_points_to_alloc(memory_map, addr) {
            return;
        }
        let Ok(mapped_ty) = memory_map.vtable_ty(addr) else {
            return;
        };
        enqueue_callable_from_mapped_ty(
            db,
            env,
            interner,
            mapped_ty,
            queue,
            closure_visited,
            closure_result,
        );
    };

    match &layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            if matches!(scalar.primitive(), Primitive::Pointer(_)) {
                maybe_collect(0);
            }
        }
        BackendRepr::ScalarPair(a, b) => {
            if matches!(a.primitive(), Primitive::Pointer(_)) {
                maybe_collect(0);
            }
            if matches!(b.primitive(), Primitive::Pointer(_)) {
                maybe_collect(layout.fields.offset(1).bytes() as usize);
            }
        }
        _ => {}
    }
}

fn collect_const_rvalue_callables(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    interner: DbInterner,
    rvalue: &Rvalue,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
) {
    let mut collect_operand = |operand: &Operand| {
        collect_const_operand_callables(
            db,
            env,
            interner,
            operand,
            queue,
            closure_visited,
            closure_result,
        );
    };

    match rvalue {
        Rvalue::Use(operand)
        | Rvalue::Repeat(operand, _)
        | Rvalue::Cast(_, operand, _)
        | Rvalue::UnaryOp(_, operand)
        | Rvalue::ShallowInitBox(operand, _) => collect_operand(operand),
        Rvalue::BinaryOp(_, lhs, rhs) => {
            collect_operand(lhs);
            collect_operand(rhs);
        }
        Rvalue::Aggregate(_, operands) => {
            for operand in operands.iter() {
                collect_operand(operand);
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
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
    closure_visited: &mut std::collections::HashSet<(
        hir_ty::db::InternedClosureId,
        StoredGenericArgs,
    )>,
    closure_result: &mut Vec<(hir_ty::db::InternedClosureId, StoredGenericArgs)>,
    drop_types: &mut std::collections::HashSet<StoredTy>,
) {
    use hir_ty::mir::AggregateKind;

    for (_, bb) in body.basic_blocks.iter() {
        // Scan statements for coercions and closure constructions
        for stmt in &bb.statements {
            if let StatementKind::Assign(_, rvalue) = &stmt.kind {
                collect_const_rvalue_callables(
                    db,
                    env,
                    interner,
                    rvalue,
                    queue,
                    closure_visited,
                    closure_result,
                );
            }

            match &stmt.kind {
                StatementKind::Assign(_, Rvalue::Cast(cast_kind, operand, target_ty)) => {
                    match cast_kind {
                        // Unsizing coercions → discover vtable impl methods
                        CastKind::PointerCoercion(PointerCast::Unsize) => {
                            collect_vtable_methods(
                                db,
                                env,
                                body,
                                operand,
                                target_ty,
                                local_crate,
                                queue,
                                closure_visited,
                                closure_result,
                                drop_types,
                            );
                        }
                        // ReifyFnPointer: fn item → fn pointer. The target fn needs compilation.
                        CastKind::PointerCoercion(PointerCast::ReifyFnPointer) => {
                            let from_ty = operand_ty(db, body, &operand.kind);
                            if let TyKind::FnDef(def, callee_args) = from_ty.as_ref().kind() {
                                if let CallableDefId::FunctionId(callee_id) = def.0 {
                                    let mut resolved_id = callee_id;
                                    let mut resolved_args = callee_args;

                                    if let ItemContainerId::TraitId(_) = callee_id.loc(db).container
                                    {
                                        if hir_ty::method_resolution::is_dyn_method(
                                            interner,
                                            env.param_env(),
                                            callee_id,
                                            callee_args,
                                        )
                                        .is_none()
                                        {
                                            match db.lookup_impl_method(
                                                env.as_ref(),
                                                callee_id,
                                                callee_args,
                                            ) {
                                                (Either::Left(impl_id), impl_args) => {
                                                    resolved_id = impl_id;
                                                    resolved_args = impl_args;
                                                }
                                                (Either::Right(_), _) => {
                                                    continue;
                                                }
                                            }
                                        }
                                    }

                                    queue.push_back((resolved_id, resolved_args.store()));
                                }
                            }
                        }
                        // ClosureFnPointer: non-capturing closure -> fn pointer.
                        // The closure body is codegen'd as a standalone symbol.
                        CastKind::PointerCoercion(PointerCast::ClosureFnPointer(_)) => {
                            let from_ty = operand_ty(db, body, &operand.kind);
                            if let TyKind::Closure(closure_id, closure_subst) = from_ty.as_ref().kind()
                            {
                                let key = (closure_id.0, closure_subst.store());
                                if closure_visited.insert(key.clone()) {
                                    closure_result.push(key);
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
            TerminatorKind::SwitchInt { discr, .. } => {
                collect_const_operand_callables(
                    db,
                    env,
                    interner,
                    discr,
                    queue,
                    closure_visited,
                    closure_result,
                );
            }
            TerminatorKind::Assert { cond, .. } => {
                collect_const_operand_callables(
                    db,
                    env,
                    interner,
                    cond,
                    queue,
                    closure_visited,
                    closure_result,
                );
            }
            TerminatorKind::Call {
                func,
                args,
                destination,
                ..
            } => {
                collect_const_operand_callables(
                    db,
                    env,
                    interner,
                    func,
                    queue,
                    closure_visited,
                    closure_result,
                );
                for arg in args.iter() {
                    collect_const_operand_callables(
                        db,
                        env,
                        interner,
                        arg,
                        queue,
                        closure_visited,
                        closure_result,
                    );
                }

                // Call targets can be carried through locals (`Move`/`Copy`) after
                // `ReifyFnPointer`; inspect the operand type rather than requiring
                // a direct `Constant` operand to avoid missing reachable callees.
                let func_ty = operand_ty(db, body, &func.kind);
                let TyKind::FnDef(def, callee_args) = func_ty.as_ref().kind() else { continue };
                if let CallableDefId::FunctionId(callee_id) = def.0 {
                    if FunctionSignature::is_intrinsic(db, callee_id) {
                        // Runtime lowering of `const_eval_select` calls its runtime-arm
                        // function directly; ensure reachability includes that callee.
                        if db.function_signature(callee_id).name.as_str() == "const_eval_select"
                            && args.len() >= 3
                        {
                            let runtime_ty = operand_ty(db, body, &args[2].kind);
                            if let TyKind::FnDef(runtime_def, runtime_args) =
                                runtime_ty.as_ref().kind()
                                && let CallableDefId::FunctionId(runtime_func_id) = runtime_def.0
                            {
                                queue.push_back((runtime_func_id, runtime_args.store()));
                            }
                        }
                        continue;
                    }

                    let lang_items = hir_def::lang_item::lang_items(db, local_crate);
                    if Some(callee_id) == lang_items.DropInPlace {
                        if !callee_args.is_empty() {
                            let pointee_ty = callee_args.type_at(0).store();
                            collect_drop_info(db, local_crate, &pointee_ty, queue, drop_types);
                        }
                        continue;
                    }

                    let mut resolved_id = callee_id;
                    let mut resolved_args = callee_args;

                    // Skip virtual calls — trait methods on dyn types are dispatched
                    // through vtables, not compiled as standalone functions.
                    if let ItemContainerId::TraitId(trait_id) = callee_id.loc(db).container {
                        if hir_ty::method_resolution::is_dyn_method(
                            interner,
                            env.param_env(),
                            callee_id,
                            callee_args,
                        )
                        .is_some()
                        {
                            continue;
                        }

                        let receiver_ty = args
                            .first()
                            .map(|self_arg| operand_ty(db, body, &self_arg.kind).as_ref())
                            .or_else(|| (!callee_args.is_empty()).then(|| callee_args.type_at(0)));
                        let trait_method_name = db.function_signature(callee_id).name.clone();

                        // Mirror `codegen_direct_call` Fn-trait dispatch to avoid
                        // leaving real callees (e.g. fn-item shims) out of the queue.
                        if is_fn_trait_method(db, trait_id, trait_method_name.as_str())
                            && let Some(self_ty) = receiver_ty
                        {
                            match peel_ref_layers(self_ty).kind() {
                                TyKind::Closure(closure_id, closure_subst) => {
                                    let key = (closure_id.0, closure_subst.store());
                                    if closure_visited.insert(key.clone()) {
                                        closure_result.push(key);
                                    }
                                    continue;
                                }
                                TyKind::FnDef(def, fn_args) => {
                                    if let CallableDefId::FunctionId(fn_id) = def.0 {
                                        let mut resolved_fn_id = fn_id;
                                        let mut resolved_fn_args = fn_args;

                                        if let ItemContainerId::TraitId(_) = fn_id.loc(db).container {
                                            if hir_ty::method_resolution::is_dyn_method(
                                                interner,
                                                env.param_env(),
                                                fn_id,
                                                fn_args,
                                            )
                                            .is_none()
                                            {
                                                match db.lookup_impl_method(
                                                    env.as_ref(),
                                                    fn_id,
                                                    fn_args,
                                                ) {
                                                    (Either::Left(impl_id), impl_args) => {
                                                        resolved_fn_id = impl_id;
                                                        resolved_fn_args = impl_args;
                                                    }
                                                    (Either::Right(_), _) => {
                                                        continue;
                                                    }
                                                }
                                            }
                                        }

                                        queue.push_back((resolved_fn_id, resolved_fn_args.store()));
                                    }
                                    continue;
                                }
                                TyKind::FnPtr(_, _) => {
                                    // Indirect fn-pointer dispatch has no direct body to queue.
                                    continue;
                                }
                                _ => {}
                            }
                        }

                        match db.lookup_impl_method(env.as_ref(), callee_id, callee_args) {
                            (Either::Left(impl_id), impl_args) => {
                                resolved_id = impl_id;
                                resolved_args = impl_args;
                            }
                            (Either::Right((derive_impl_id, derive_method)), _) => {
                                let self_ty = args
                                    .first()
                                    .map(|self_arg| operand_ty(db, body, &self_arg.kind))
                                    .and_then(|ty| {
                                        ty.as_ref().builtin_deref(true).map(|it| it.store())
                                    })
                                    .or_else(|| {
                                        (!callee_args.is_empty())
                                            .then(|| callee_args.type_at(0).store())
                                    })
                                    .expect("builtin derive call receiver must be present");
                                collect_builtin_derive_method_dependencies(
                                    db,
                                    env,
                                    local_crate,
                                    derive_impl_id,
                                    derive_method,
                                    self_ty,
                                    queue,
                                );
                                continue;
                            }
                        }
                    }
                    queue.push_back((resolved_id, resolved_args.store()));

                    if let Some(source_return_ty) = callable_output_ty(db, resolved_id, resolved_args)
                    {
                        let dest_ty = place_ty(db, body, destination);
                        if let (Some(src_pointee), Some(dest_pointee)) = (
                            source_return_ty.as_ref().builtin_deref(true),
                            dest_ty.as_ref().builtin_deref(true),
                        ) {
                            collect_unsize_metadata_dependencies(
                                db,
                                env,
                                local_crate,
                                src_pointee.store(),
                                dest_pointee.store(),
                                queue,
                                closure_visited,
                                closure_result,
                                drop_types,
                            );
                        }
                    }
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

    // Discover and compile all reachable monomorphized function instances.
    let (reachable_fns, reachable_closures, drop_types) =
        collect_reachable_fns(db, env, main_func_id, local_crate);
    let mut user_main_id = None;
    let mut compiled_closure_symbols = std::collections::HashSet::new();
    let mut compiled_drop_symbols = std::collections::HashSet::new();

    for (func_id, generic_args) in &reachable_fns {
        let body = db
            .monomorphized_mir_body((*func_id).into(), generic_args.clone(), env.clone())
            .map_err(|e| format!("MIR error for reachable fn: {:?}", e))?;
        let fn_name = symbol_mangling::mangle_function(
            db,
            *func_id,
            generic_args.as_ref(),
            ext_crate_disambiguators,
        );
        let func_clif_id = compile_fn(
            &mut module,
            &*isa,
            db,
            dl,
            env,
            &body,
            &fn_name,
            Linkage::Export,
            local_crate,
            ext_crate_disambiguators,
        )?;

        if *func_id == main_func_id {
            user_main_id = Some(func_clif_id);
        }
    }

    // Compile reachable closure bodies
    for (closure_id, closure_subst) in &reachable_closures {
        let closure_name = symbol_mangling::mangle_closure(
            db,
            *closure_id,
            closure_subst.as_ref(),
            ext_crate_disambiguators,
        );
        if !compiled_closure_symbols.insert(closure_name.clone()) {
            continue;
        }
        let body = match db.monomorphized_mir_body_for_closure(
            *closure_id,
            closure_subst.clone(),
            env.clone(),
        ) {
            Ok(body) => body,
            Err(hir_ty::mir::MirLowerError::UnresolvedName(_)) => continue,
            Err(e) => return Err(format!("MIR error for closure {closure_name}: {e:?}")),
        };
        compile_fn(
            &mut module,
            &*isa,
            db,
            dl,
            env,
            &body,
            &closure_name,
            Linkage::Export,
            local_crate,
            ext_crate_disambiguators,
        )?;
    }

    // Compile drop_in_place glue functions
    for ty in &drop_types {
        let drop_name =
            symbol_mangling::mangle_drop_in_place(db, ty.as_ref(), ext_crate_disambiguators);
        if !compiled_drop_symbols.insert(drop_name) {
            continue;
        }
        compile_drop_in_place(
            &mut module,
            &*isa,
            db,
            dl,
            env,
            ty,
            local_crate,
            ext_crate_disambiguators,
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

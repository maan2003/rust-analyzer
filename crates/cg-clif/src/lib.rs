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
    AbiParam, ArgumentPurpose, AtomicRmwOp, Block, InstBuilder, MemFlags, Signature, Type, Value,
    types,
};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_module::{DataDescription, FuncId, Linkage, Module, ModuleError};
use cranelift_object::{ObjectBuilder, ObjectModule};
use either::Either;
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
use hir_ty::next_solver::infer::DbInternerInferExt;
use hir_ty::next_solver::{
    Const, ConstKind, DbInterner, GenericArgs, IntoKind, StoredGenericArgs, StoredTy, TyKind,
    ValueConst,
};
use hir_ty::traits::StoredParamEnvAndCrate;
use rac_abi::VariantIdx;
use rustc_abi::{BackendRepr, Primitive, Scalar, Size, TargetDataLayout};
use rustc_type_ir::elaborate::supertrait_def_ids;
use rustc_type_ir::inherent::{GenericArgs as _, Region as _, Ty as _};
use triomphe::Arc as TArc;

pub mod link;
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

fn iconst_from_bits(bcx: &mut FunctionBuilder<'_>, ty: Type, bits: u128) -> Value {
    if ty == types::I128 {
        let lsb = bcx.ins().iconst(types::I64, bits as u64 as i64);
        let msb = bcx.ins().iconst(types::I64, (bits >> 64) as u64 as i64);
        bcx.ins().iconcat(lsb, msb)
    } else {
        bcx.ins().iconst(ty, bits as u64 as i64)
    }
}

/// Append a return type to a Cranelift signature based on its layout.
/// Returns `true` if the return is memory-repr and needs an sret pointer.
pub(crate) fn append_ret_to_sig(
    sig: &mut Signature,
    dl: &TargetDataLayout,
    layout: &Layout,
) -> bool {
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
            false
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
            false
        }
        _ if layout.is_zst() => false,
        _ => {
            sig.params.push(AbiParam::special(
                pointer_ty(dl),
                cranelift_codegen::ir::ArgumentPurpose::StructReturn,
            ));
            true
        }
    }
}

/// Append a parameter to a Cranelift signature based on its layout.
pub(crate) fn append_param_to_sig(sig: &mut Signature, dl: &TargetDataLayout, layout: &Layout) {
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
            sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
        }
        _ if layout.is_zst() => {}
        _ => {
            sig.params.push(AbiParam::new(pointer_ty(dl)));
        }
    }
}

/// Emit a return instruction from the given return place.
/// Handles ZST, Scalar, ScalarPair, and Memory (sret) returns.
pub(crate) fn codegen_return(fx: &mut FunctionCx<'_, impl Module>, ret_place: &CPlace) {
    if ret_place.layout.is_zst() {
        fx.bcx.ins().return_(&[]);
    } else {
        let cval = ret_place.to_cvalue(fx);
        match ret_place.layout.backend_repr {
            BackendRepr::Scalar(_) => {
                let val = cval.load_scalar(fx);
                fx.bcx.ins().return_(&[val]);
            }
            BackendRepr::ScalarPair(_, _) => {
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
        let CallableDefId::FunctionId(callee_func_id) = def.0 else {
            panic!("ReifyFnPointer on non-function: {:?}", def);
        };
        let (callee_func_id, generic_args) = if let ItemContainerId::TraitId(_) =
            callee_func_id.loc(fx.db()).container
        {
            let interner = DbInterner::new_no_crate(fx.db());
            if hir_ty::method_resolution::is_dyn_method(
                interner,
                fx.env().param_env(),
                callee_func_id,
                generic_args,
            )
            .is_some()
            {
                // Keep trait item identity for dyn methods; they don't have a
                // concrete impl target at reify time.
                (callee_func_id, generic_args)
            } else {
                match fx.db().lookup_impl_method(fx.env().as_ref(), callee_func_id, generic_args) {
                    (Either::Left(resolved_id), resolved_args) => (resolved_id, resolved_args),
                    (Either::Right((derive_impl_id, derive_method)), _) => {
                        panic!(
                            "cannot reify builtin-derive method pointer: {:?}::{:?}",
                            derive_impl_id, derive_method
                        );
                    }
                }
            }
        } else {
            (callee_func_id, generic_args)
        };

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
            let name = fx.db().function_signature(callee_func_id).name.as_str().to_owned();
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

    let metadata = match target_pointee.kind() {
        TyKind::Dynamic(..) => match source_pointee.kind() {
            TyKind::Dynamic(..) => {
                from_meta.expect("dyn unsize from wide source requires metadata")
            }
            _ => {
                let trait_id = target_pointee
                    .dyn_trait()
                    .expect("dyn unsize target pointee must have a principal trait");
                get_or_create_vtable(fx, source_pointee.store(), trait_id)
            }
        },
        TyKind::Slice(_) | TyKind::Str => match source_pointee.kind() {
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
                "unsupported unsize to slice/str: source pointee kind {:?}",
                source_pointee.kind()
            ),
        },
        _ => panic!(
            "unsupported unsize target pointee kind {:?} (source {:?})",
            target_pointee.kind(),
            source_pointee.kind()
        ),
    };

    CValue::by_val_pair(data_ptr, metadata, result_layout.clone())
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
fn trait_method_substs_for_receiver<'db>(
    db: &'db dyn HirDatabase,
    local_crate: base_db::Crate,
    trait_method_func_id: hir_def::FunctionId,
    concrete_self_ty: &StoredTy,
) -> GenericArgs<'db> {
    let interner = DbInterner::new_with(db, local_crate);
    let identity = GenericArgs::identity_for_item(interner, trait_method_func_id.into());
    assert!(
        !identity.is_empty(),
        "trait method has no generic args: {:?}",
        trait_method_func_id
    );
    GenericArgs::new_from_iter(
        interner,
        std::iter::once(concrete_self_ty.as_ref().into()).chain(identity.iter().skip(1)),
    )
}

fn get_or_create_vtable(
    fx: &mut FunctionCx<'_, impl Module>,
    concrete_ty: StoredTy,
    trait_id: TraitId,
) -> Value {
    let ptr_size = fx.dl.pointer_size().bytes() as usize;

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
    let impl_id = find_trait_impl_for_simplified_ty(fx.db(), trait_id, &simplified, krate);

    // Closure traits (`Fn` / `FnMut` / `FnOnce`) are built-in and often don't
    // show up as explicit impl items in TraitImpls. In that case, wire the
    // vtable method slot directly to the closure body symbol.
    let closure_fallback = if impl_id.is_none() {
        match concrete_ty.as_ref().kind() {
            TyKind::Closure(closure_id, closure_subst) => {
                Some((closure_id.0, closure_subst.store()))
            }
            _ => None,
        }
    } else {
        None
    };

    assert!(
        impl_id.is_some() || closure_fallback.is_some(),
        "no impl found for vtable: trait={:?}, concrete_ty={:?}, simplified={:?}",
        trait_id,
        concrete_ty,
        simplified
    );

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

    // Slot 0: drop_in_place — null for now (no drop glue)
    // (already zeroed)

    // Slot 1: size
    vtable_bytes[ptr_size..ptr_size * 2]
        .copy_from_slice(&(concrete_size as u64).to_le_bytes()[..ptr_size]);

    // Slot 2: alignment
    vtable_bytes[ptr_size * 2..ptr_size * 3]
        .copy_from_slice(&(concrete_align as u64).to_le_bytes()[..ptr_size]);

    data.define(vtable_bytes.into_boxed_slice());

    // Slot 3+: trait method fn ptrs — emit as relocations
    let closure_func_id = closure_fallback.map(|(closure_id, closure_subst)| {
        let closure_body = fx
            .db()
            .monomorphized_mir_body_for_closure(closure_id, closure_subst, fx.env().clone())
            .expect("failed to get closure MIR for vtable");
        let closure_sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body)
            .expect("closure sig for vtable");
        let closure_name =
            symbol_mangling::mangle_closure(fx.db(), closure_id, fx.ext_crate_disambiguators());
        fx.module
            .declare_function(&closure_name, Linkage::Import, &closure_sig)
            .expect("declare closure vtable method")
    });

    for (method_idx, trait_method_func_id) in method_func_ids.iter().enumerate() {
        let trait_method_name = fx.db().function_signature(*trait_method_func_id).name.clone();
        let func_id = if impl_id.is_some() {
            let trait_method_subst = trait_method_substs_for_receiver(
                fx.db(),
                krate,
                *trait_method_func_id,
                &concrete_ty,
            );
            let (impl_func_id, impl_generic_args) =
                match fx
                    .db()
                    .lookup_impl_method(fx.env().as_ref(), *trait_method_func_id, trait_method_subst)
                {
                    (Either::Left(resolved_id), resolved_args) => (resolved_id, resolved_args),
                    (Either::Right((derive_impl_id, derive_method)), _) => {
                        panic!(
                            "unsupported builtin-derive vtable method: {:?}::{:?}",
                            derive_impl_id, derive_method
                        );
                    }
                };

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
        } else {
            let closure_func_id = closure_func_id.expect("closure vtable method func id");
            assert!(
                matches!(trait_method_name.as_str(), "call" | "call_mut" | "call_once"),
                "closure fallback cannot build vtable entry for trait method `{}`",
                trait_method_name.as_str()
            );
            closure_func_id
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
                    let ptr = place.place_field(fx, tag_field.as_usize(), tag_layout);
                    let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);
                    let discr_bits =
                        enum_variant_discriminant_bits(fx.db(), enum_ty, variant_index);
                    let discr_bits = tag.size(fx.dl).truncate(discr_bits);
                    let to = iconst_from_bits(&mut fx.bcx, tag_clif_ty, discr_bits);
                    ptr.write_cvalue(fx, CValue::by_val(to, ptr.layout.clone()));
                }
                TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
                    if variant_index != *untagged_variant {
                        let niche = place.place_field(fx, tag_field.as_usize(), tag_layout);
                        let niche_type = scalar_to_clif_type(fx.dl, tag);
                        let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                        let niche_value = (niche_value as u128).wrapping_add(*niche_start);
                        let niche_value = iconst_from_bits(&mut fx.bcx, niche_type, niche_value);
                        niche.write_cvalue(fx, CValue::by_val(niche_value, niche.layout.clone()));
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

    let ptr = if is_extern || !is_local_static {
        let data_id = fx
            .module
            .declare_data(&symbol_name, Linkage::Import, is_mutable, false)
            .unwrap_or_else(|e| panic!("declare imported static `{symbol_name}` failed: {e}"));
        let local_data_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
        fx.bcx.ins().symbol_value(fx.pointer_type, local_data_id)
    } else {
        let data_id = fx
            .module
            .declare_data(&symbol_name, Linkage::Local, is_mutable, false)
            .unwrap_or_else(|e| panic!("declare static `{symbol_name}` failed: {e}"));

        match db.const_eval_static(static_id) {
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
                    let val = match scalar.primitive() {
                        Primitive::Float(rustc_abi::Float::F32) => {
                            fx.bcx.ins().f32const(f32::from_bits(raw_bits as u32))
                        }
                        Primitive::Float(rustc_abi::Float::F64) => {
                            fx.bcx.ins().f64const(f64::from_bits(raw_bits as u64))
                        }
                        Primitive::Float(_) => todo!("f16/f128 constants"),
                        _ => {
                            let clif_ty = scalar_to_clif_type(fx.dl, &scalar);
                            iconst_from_bits(&mut fx.bcx, clif_ty, raw_bits)
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
                    let addr_to_gv = create_const_data_sections(fx, &const_memory_map);

                    let a_val = codegen_scalar_const(fx, &a_scalar, a_raw, &addr_to_gv);
                    let b_val = codegen_scalar_const(fx, &b_scalar, b_raw, &addr_to_gv);
                    CValue::by_val_pair(a_val, b_val, layout)
                }
                _ => {
                    // Memory-repr constant: store raw bytes in a data section
                    let const_memory = {
                        let val = resolve_const_value(fx.db(), konst.as_ref());
                        val.value.inner().memory.clone()
                    };
                    let mut data_desc = DataDescription::new();
                    data_desc.define(const_memory.clone());
                    data_desc.set_align(layout.align.abi.bytes());
                    let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                    fx.module.define_data(data_id, &data_desc).unwrap();
                    let local_id = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                    let ptr = fx.bcx.ins().symbol_value(fx.pointer_type, local_id);
                    CValue::by_ref(pointer::Pointer::new(ptr), layout)
                }
            }
        }
        OperandKind::Copy(place) => codegen_place(fx, place).to_cvalue(fx),
        OperandKind::Move(place) => {
            let val = codegen_place(fx, place).to_cvalue(fx);
            // Clear drop flag: the value has been moved out.
            if place.projection.lookup(&fx.ra_body().projection_store).is_empty() {
                fx.clear_drop_flag(place.local);
            }
            val
        }
        OperandKind::Static(static_id) => codegen_static_operand(fx, *static_id),
    }
}

/// Create anonymous data sections for each allocation in a `MemoryMap`,
/// returning a mapping from virtual address → Cranelift GlobalValue.
fn create_const_data_sections(
    fx: &mut FunctionCx<'_, impl Module>,
    memory_map: &hir_ty::MemoryMap<'_>,
) -> HashMap<usize, cranelift_codegen::ir::GlobalValue> {
    use hir_ty::MemoryMap;
    let mut map = HashMap::new();
    match memory_map {
        MemoryMap::Empty => {}
        MemoryMap::Simple(data) => {
            let mut data_desc = DataDescription::new();
            data_desc.define(data.to_vec().into_boxed_slice());
            let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
            fx.module.define_data(data_id, &data_desc).unwrap();
            let gv = fx.module.declare_data_in_func(data_id, fx.bcx.func);
            map.insert(0usize, gv);
        }
        MemoryMap::Complex(cm) => {
            for (addr, data) in cm.memory_iter() {
                let mut data_desc = DataDescription::new();
                data_desc.define(data.to_vec().into_boxed_slice());
                let data_id = fx.module.declare_anonymous_data(false, false).unwrap();
                fx.module.define_data(data_id, &data_desc).unwrap();
                let gv = fx.module.declare_data_in_func(data_id, fx.bcx.func);
                map.insert(*addr, gv);
            }
        }
    }
    map
}

/// Emit a Cranelift Value for a single scalar constant.
/// If the scalar is a pointer and the address matches an entry in `addr_to_gv`,
/// emits a `symbol_value` (relocation). Otherwise, emits an `iconst` or float const.
fn codegen_scalar_const(
    fx: &mut FunctionCx<'_, impl Module>,
    scalar: &Scalar,
    raw: u128,
    addr_to_gv: &HashMap<usize, cranelift_codegen::ir::GlobalValue>,
) -> Value {
    match scalar.primitive() {
        Primitive::Pointer(_) => {
            let addr = raw as usize;
            if let Some(&gv) = addr_to_gv.get(&addr) {
                fx.bcx.ins().symbol_value(fx.pointer_type, gv)
            } else if addr == 0 {
                iconst_from_bits(&mut fx.bcx, fx.pointer_type, 0)
            } else {
                // Unknown address — treat as raw integer (shouldn't normally happen)
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

/// Store a call's return value into `dest` and jump to the continuation block
/// (or trap for diverging calls). Shared by all call codegen paths.
fn store_call_result_and_jump(
    fx: &mut FunctionCx<'_, impl Module>,
    call: cranelift_codegen::ir::Inst,
    sret_slot: Option<CPlace>,
    dest: CPlace,
    destination_place: &Place,
    target: &Option<BasicBlockId>,
) {
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
    if projections.is_empty() {
        let local_idx = place.local.into_raw().into_u32();
        if let Some(&flag_var) = fx.drop_flags.get(&local_idx) {
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
        let param_layout =
            fx.db().layout_of_ty(param_ty.store(), fx.env().clone()).expect("fn ptr param layout");
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
    store_call_result_and_jump(fx, call, sret_slot, dest, destination, target);
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
    let closure_body = fx
        .db()
        .monomorphized_mir_body_for_closure(closure_id, closure_subst, fx.env().clone())
        .expect("closure MIR");
    let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &closure_body).expect("closure sig");

    // Generate mangled name
    let closure_name =
        symbol_mangling::mangle_closure(fx.db(), closure_id, fx.ext_crate_disambiguators());

    // Declare in module (Import linkage — defined elsewhere in same module)
    let callee_id =
        fx.module.declare_function(&closure_name, Linkage::Import, &sig).expect("declare closure");
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
    store_call_result_and_jump(fx, call, sret_slot, dest, destination, target);
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
    match derive_method {
        BuiltinDeriveImplMethod::clone => {
            assert_eq!(args.len(), 1, "builtin Clone::clone expects one receiver argument");

            let self_ty = if generic_args.is_empty() {
                let receiver_ref_ty = operand_ty(fx.db(), fx.ra_body(), &args[0].kind);
                receiver_ref_ty
                    .as_ref()
                    .builtin_deref(true)
                    .expect("builtin Clone::clone receiver must be a reference")
            } else {
                generic_args.type_at(0)
            };
            let interner = DbInterner::new_with(fx.db(), fx.local_crate());
            let infcx = interner.infer_ctxt().build(hir_ty::next_solver::TypingMode::PostAnalysis);
            assert!(
                infcx.type_is_copy_modulo_regions(fx.env().param_env(), self_ty),
                "builtin Clone::clone for non-Copy Self is not supported yet: {:?}",
                self_ty
            );

            let self_layout = fx
                .db()
                .layout_of_ty(self_ty.store(), fx.env().clone())
                .expect("layout for builtin Clone::clone receiver");
            let self_ref = codegen_operand(fx, &args[0].kind);
            let BackendRepr::Scalar(_) = self_ref.layout.backend_repr else {
                panic!("builtin Clone::clone expects a thin &Self receiver");
            };
            let self_ptr = self_ref.load_scalar(fx);
            let self_val = CValue::by_ref(pointer::Pointer::new(self_ptr), self_layout);

            let dest = codegen_place(fx, destination);
            dest.write_cvalue(fx, self_val);

            if target.is_some()
                && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
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
        BuiltinDeriveImplMethod::fmt => {
            // Builtin derive impls are pseudo impls with no MIR body. For Debug::fmt
            // we at least preserve the fmt::Result contract (`Ok(())`) so callers
            // can continue execution instead of crashing in codegen.
            assert_eq!(
                args.len(),
                2,
                "builtin Debug::fmt expects receiver and formatter arguments"
            );

            let dest = codegen_place(fx, destination);
            if !dest.layout.is_zst() {
                let result_ty = place_ty(fx.db(), fx.ra_body(), destination);
                let TyKind::Adt(adt, _) = result_ty.as_ref().kind() else {
                    panic!("builtin Debug::fmt must return Result, got: {:?}", result_ty)
                };
                let hir_def::AdtId::EnumId(_) = adt.inner().id else {
                    panic!("builtin Debug::fmt must return enum Result, got: {:?}", result_ty)
                };
                // `core::fmt::Result` is `Result<(), fmt::Error>`; variant 0 is `Ok`.
                codegen_set_discriminant(fx, &dest, &result_ty, VariantIdx::from_u32(0));
            }

            if target.is_some()
                && destination.projection.lookup(&fx.ra_body().projection_store).is_empty()
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
        _ => panic!(
            "unsupported builtin-derive method during codegen: {:?}::{:?}",
            derive_impl_id, derive_method
        ),
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
            let ptr = match arg.layout.backend_repr {
                BackendRepr::Scalar(_) => arg.load_scalar(fx),
                BackendRepr::ScalarPair(_, _) => {
                    panic!("drop_in_place for unsized type with drop glue is not supported yet")
                }
                _ => panic!("drop_in_place argument must be a pointer"),
            };

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
                codegen_virtual_call(fx, callee_func_id, trait_id, args, destination, target);
                return;
            }

            // Check for closure call: Fn::call / FnMut::call_mut / FnOnce::call_once
            // where self type is a closure — redirect to the closure's MIR body.
            if generic_args.len() > 0 {
                let self_ty = generic_args.type_at(0);
                if let TyKind::Closure(closure_id, closure_subst) = self_ty.kind() {
                    codegen_closure_call(
                        fx,
                        closure_id.0,
                        closure_subst.store(),
                        args,
                        destination,
                        target,
                    );
                    return;
                }

                // Calls like `FnMut::call_mut` on `&mut dyn FnMut(..)` often
                // resolve to the blanket reference impl in `core::ops::function::impls`,
                // but the actual runtime dispatch must still happen through the inner
                // dyn vtable. Route those straight to virtual dispatch.
                if let Some(self_pointee) = self_ty.builtin_deref(true)
                    && self_pointee.dyn_trait() == Some(trait_id)
                    && matches!(
                        fx.db()
                            .layout_of_ty(self_ty.store(), fx.env().clone())
                            .expect("self layout for dyn ref dispatch")
                            .backend_repr,
                        BackendRepr::ScalarPair(_, _)
                    )
                {
                    codegen_virtual_call(fx, callee_func_id, trait_id, args, destination, target);
                    return;
                }
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

    // Check if this is an extern function (no MIR available)
    let is_extern =
        matches!(callee_func_id.loc(fx.db()).container, ItemContainerId::ExternBlockId(_));
    let is_cross_crate = callee_func_id.krate(fx.db()) != fx.local_crate();

    let interner = DbInterner::new_no_crate(fx.db());
    let empty_args = GenericArgs::empty(interner);
    let (callee_sig, callee_name) = if is_extern {
        // Extern functions: build signature from type info, use raw symbol name
        let sig =
            build_fn_sig_from_ty(fx.isa, fx.db(), fx.dl, fx.env(), callee_func_id, empty_args)
                .expect("extern fn sig");
        let name = fx.db().function_signature(callee_func_id).name.as_str().to_owned();
        (sig, name)
    } else if let Ok(callee_body) = fx.db().monomorphized_mir_body(
        callee_func_id.into(),
        generic_args.clone(),
        fx.env().clone(),
    ) {
        // Prefer r-a MIR whenever available, including cross-crate bodies.
        let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body).expect("callee sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    } else if is_cross_crate {
        // Fall back to symbol-only cross-crate calls when MIR lowering is unavailable.
        let sig = build_fn_sig_from_ty(
            fx.isa,
            fx.db(),
            fx.dl,
            fx.env(),
            callee_func_id,
            generic_args.as_ref(),
        )
        .expect("cross-crate fn sig");
        let name = symbol_mangling::mangle_function(
            fx.db(),
            callee_func_id,
            generic_args.as_ref(),
            fx.ext_crate_disambiguators(),
        );
        (sig, name)
    } else {
        panic!("failed to get local callee MIR for {:?}", callee_func_id);
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
    store_call_result_and_jump(fx, call, sret_slot, dest, destination, target);
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
    let fn_ptr =
        fx.bcx.ins().load(fx.pointer_type, vtable_memflags(), vtable_ptr, vtable_offset as i32);

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
    store_call_result_and_jump(fx, call, sret_slot, dest, destination, target);
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
        "type_id" | "type_name" => {
            // These are complex; fall through to let them be unresolved
            // (they need const eval or string allocation)
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
            let (callee_sig, callee_name) = if is_extern {
                let sig = build_fn_sig_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    runtime_func_id,
                    empty_args,
                )
                .expect("const_eval_select extern fn sig");
                let name = fx.db().function_signature(runtime_func_id).name.as_str().to_owned();
                (sig, name)
            } else if let Ok(callee_body) = fx.db().monomorphized_mir_body(
                runtime_func_id.into(),
                runtime_generic_args.store(),
                fx.env().clone(),
            ) {
                let sig = build_fn_sig(fx.isa, fx.db(), fx.dl, fx.env(), &callee_body)
                    .expect("callee sig");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    runtime_func_id,
                    runtime_generic_args,
                    fx.ext_crate_disambiguators(),
                );
                (sig, name)
            } else if is_cross_crate {
                let sig = build_fn_sig_from_ty(
                    fx.isa,
                    fx.db(),
                    fx.dl,
                    fx.env(),
                    runtime_func_id,
                    runtime_generic_args,
                )
                .expect("const_eval_select cross-crate fn sig");
                let name = symbol_mangling::mangle_function(
                    fx.db(),
                    runtime_func_id,
                    runtime_generic_args,
                    fx.ext_crate_disambiguators(),
                );
                (sig, name)
            } else {
                panic!(
                    "failed to get const_eval_select runtime callee MIR for {runtime_func_id:?}"
                );
            };

            let callee_id = fx
                .module
                .declare_function(&callee_name, Linkage::Import, &callee_sig)
                .expect("declare const_eval_select runtime callee");
            let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

            let dest = codegen_place(fx, destination);
            let is_sret_return = !dest.layout.is_zst()
                && !matches!(
                    dest.layout.backend_repr,
                    BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _)
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

            for arg in runtime_args {
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
                        let ptr = arg.force_stack(fx);
                        call_args.push(ptr.get_addr(&mut fx.bcx, fx.pointer_type));
                    }
                }
            }

            let call = fx.bcx.ins().call(callee_ref, &call_args);
            store_call_result_and_jump(fx, call, sret_slot, dest, destination, target);
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
            if std::env::var_os("CG_CLIF_STD_JIT_TRACE").is_some() {
                eprintln!(
                    "std-jit: unhandled intrinsic `{}` (callee={:?}, generic_args={})",
                    name,
                    callee_func_id,
                    generic_args.len(),
                );
            }
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
    let mut sig = Signature::new(isa.default_call_conv());

    // Return type
    let ret_local = &body.locals[hir_ty::mir::return_slot()];
    let ret_layout = db
        .layout_of_ty(ret_local.ty.clone(), env.clone())
        .map_err(|e| format!("return type layout error: {:?}", e))?;
    append_ret_to_sig(&mut sig, dl, &ret_layout);

    // Parameter types
    for &param_local in &body.param_locals {
        let param = &body.locals[param_local];
        let param_layout = db
            .layout_of_ty(param.ty.clone(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        append_param_to_sig(&mut sig, dl, &param_layout);
    }

    Ok(sig)
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
    let mut sig = Signature::new(isa.default_call_conv());

    let interner = DbInterner::new_no_crate(db);
    let fn_sig = if generic_args.is_empty() {
        db.callable_item_signature(func_id.into()).skip_binder().skip_binder()
    } else {
        db.callable_item_signature(func_id.into()).instantiate(interner, generic_args).skip_binder()
    };

    // Return type — skip if `!` (never) or ZST
    let output = *fn_sig.inputs_and_output.as_slice().split_last().unwrap().0;
    if !output.is_never() {
        let ret_layout = db
            .layout_of_ty(output.store(), env.clone())
            .map_err(|e| format!("return type layout error: {:?}", e))?;
        append_ret_to_sig(&mut sig, dl, &ret_layout);
    }

    // Parameter types
    for &param_ty in fn_sig.inputs_and_output.inputs() {
        let param_layout = db
            .layout_of_ty(param_ty.store(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        append_param_to_sig(&mut sig, dl, &param_layout);
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
    let sig = build_fn_sig(isa, db, dl, env, body)?;

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
            local_map[ret_idx] = CPlace::for_ptr(pointer::Pointer::new(sret_ptr), ret_layout);
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
                        else {
                            unreachable!()
                        };
                        let b_off = value_and_place::scalar_pair_b_offset(dl, *a, *b);
                        ptr.offset_i64(&mut bcx, pointer_type, b_off).store(
                            &mut bcx,
                            block_params[param_idx + 1],
                            flags,
                        );
                    }
                    param_idx += 2;
                }
                _ => {
                    // Memory-repr param: block param is a pointer to the data
                    let ptr_val = block_params[param_idx];
                    param_idx += 1;
                    let layout = place.layout.clone();
                    local_map[param_idx_local] =
                        CPlace::for_ptr(pointer::Pointer::new(ptr_val), layout);
                }
            }
        }

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
fn collect_vtable_methods(
    db: &dyn HirDatabase,
    env: &StoredParamEnvAndCrate,
    body: &MirBody,
    operand: &Operand,
    target_ty: &StoredTy,
    local_crate: base_db::Crate,
    queue: &mut std::collections::VecDeque<(hir_def::FunctionId, StoredGenericArgs)>,
) {
    // Extract the trait from the target type
    let Some(pointee_ty) = target_ty.as_ref().builtin_deref(true) else { return };
    let Some(trait_id) = pointee_ty.dyn_trait() else { return };

    // Extract the concrete source type
    let from_ty = operand_ty(db, body, &operand.kind);
    let Some(source_pointee) = from_ty.as_ref().builtin_deref(true) else { return };

    let interner = DbInterner::new_no_crate(db);
    use rustc_type_ir::fast_reject::{TreatParams, simplify_type};
    let source_pointee_stored = source_pointee.store();
    let Some(simplified) =
        simplify_type(interner, source_pointee, TreatParams::InstantiateWithInfer)
    else {
        return;
    };
    if find_trait_impl_for_simplified_ty(db, trait_id, &simplified, local_crate).is_none() {
        return;
    }

    // Add all trait method implementations to the queue
    let trait_items = trait_id.trait_items(db);
    for (_trait_method_name, trait_item) in trait_items.items.iter() {
        let AssocItemId::FunctionId(trait_method_func_id) = trait_item else { continue };
        let trait_method_subst =
            trait_method_substs_for_receiver(db, local_crate, *trait_method_func_id, &source_pointee_stored);
        if let (Either::Left(impl_func_id), impl_generic_args) =
            db.lookup_impl_method(env.as_ref(), *trait_method_func_id, trait_method_subst)
        {
            queue.push_back((impl_func_id, impl_generic_args.store()));
        }
    }
}

fn find_trait_impl_for_simplified_ty<'db>(
    db: &'db dyn HirDatabase,
    trait_id: TraitId,
    simplified: &hir_ty::next_solver::SimplifiedType,
    local_crate: base_db::Crate,
) -> Option<hir_def::ImplId> {
    let mut search_order: Vec<base_db::Crate> = vec![local_crate];
    for &krate in db.all_crates().iter() {
        if krate != local_crate {
            search_order.push(krate);
        }
    }

    for krate in search_order {
        let trait_impls = TraitImpls::for_crate(db, krate);
        let (impl_ids, _) = trait_impls.for_trait_and_self_ty(trait_id, simplified);
        if let Some(&impl_id) = impl_ids.first() {
            return Some(impl_id);
        }
    }

    None
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
            TerminatorKind::Call { func, args, .. } => {
                let OperandKind::Constant { ty, .. } = &func.kind else { continue };
                let TyKind::FnDef(def, callee_args) = ty.as_ref().kind() else { continue };
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

                    let mut resolved_id = callee_id;
                    let mut resolved_args = callee_args;

                    // Skip virtual calls — trait methods on dyn types are dispatched
                    // through vtables, not compiled as standalone functions.
                    if let ItemContainerId::TraitId(_) = callee_id.loc(db).container {
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
                        // Closure calls: Fn::call / FnMut::call_mut / FnOnce::call_once
                        // where self type is a concrete closure — record the closure body,
                        // not the trait method.
                        if callee_args.len() > 0 {
                            let self_ty = callee_args.type_at(0);
                            if let TyKind::Closure(closure_id, closure_subst) = self_ty.kind() {
                                let key = (closure_id.0, closure_subst.store());
                                if closure_visited.insert(key.clone()) {
                                    closure_result.push(key);
                                }
                                continue;
                            }
                        }

                        match db.lookup_impl_method(env.as_ref(), callee_id, callee_args) {
                            (Either::Left(impl_id), impl_args) => {
                                resolved_id = impl_id;
                                resolved_args = impl_args;
                            }
                            (Either::Right(_), _) => {
                                continue;
                            }
                        }
                    }
                    queue.push_back((resolved_id, resolved_args.store()));
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
        let body = db
            .monomorphized_mir_body_for_closure(*closure_id, closure_subst.clone(), env.clone())
            .map_err(|e| format!("MIR error for closure: {:?}", e))?;
        let closure_name =
            symbol_mangling::mangle_closure(db, *closure_id, ext_crate_disambiguators);
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

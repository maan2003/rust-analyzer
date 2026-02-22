//! MIR → Cranelift IR codegen for rust-analyzer.
//!
//! Translates r-a's MIR representation to Cranelift IR and emits object files
//! via cranelift-object. Based on patterns from cg_clif (rustc's Cranelift backend).

use std::path::Path;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, Block, InstBuilder, MemFlags, Signature, Type, Value, types};
use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use hir_def::{CallableDefId, ItemContainerId};
use hir_def::signatures::FunctionSignature;
use hir_ty::db::HirDatabase;
use hir_ty::mir::{
    BasicBlockId, BinOp, CastKind, LocalId, MirBody, Operand, OperandKind, Place,
    ProjectionElem, Rvalue, StatementKind, TerminatorKind, UnOp,
};
use either::Either;
use hir_ty::next_solver::{Const, ConstKind, GenericArgs, IntoKind, StoredTy, TyKind};
use hir_ty::traits::StoredParamEnvAndCrate;
use la_arena::ArenaMap;
use rac_abi::VariantIdx;
use rustc_abi::{BackendRepr, Primitive, Scalar, Size, TargetDataLayout};
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

// ---------------------------------------------------------------------------
// FunctionCx: per-function codegen state
// ---------------------------------------------------------------------------

struct FunctionCx<'a, M: Module> {
    bcx: FunctionBuilder<'a>,
    module: &'a mut M,
    isa: &'a dyn TargetIsa,
    pointer_type: Type,
    db: &'a dyn HirDatabase,
    dl: &'a TargetDataLayout,
    env: StoredParamEnvAndCrate,

    /// MIR basic block → Cranelift block
    block_map: ArenaMap<BasicBlockId, Block>,
    /// MIR local → CPlace (SSA variable, variable pair, or stack slot)
    local_map: ArenaMap<LocalId, CPlace>,
}

impl<M: Module> FunctionCx<'_, M> {
    fn clif_block(&self, bb: BasicBlockId) -> Block {
        self.block_map[bb]
    }

    fn local_place(&self, local: LocalId) -> &CPlace {
        &self.local_map[local]
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
    body: &MirBody,
    kind: &CastKind,
    operand: &Operand,
    target_ty: &StoredTy,
    result_layout: &LayoutArc,
) -> CValue {
    let from_ty = operand_ty(fx.db, body, &operand.kind);
    let from_cval = codegen_operand(fx, body, &operand.kind);
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

// ---------------------------------------------------------------------------
// Place codegen
// ---------------------------------------------------------------------------

fn codegen_place(fx: &mut FunctionCx<'_, impl Module>, body: &MirBody, place: &Place) -> CPlace {
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
                let field_ty = field_type(fx.db, &cur_ty, field);
                let field_layout = fx
                    .db
                    .layout_of_ty(field_ty.clone(), fx.env.clone())
                    .expect("field layout error");

                cplace = cplace.place_field(fx, field_idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Deref => {
                let inner_ty = cur_ty
                    .as_ref()
                    .builtin_deref(true)
                    .expect("deref on non-pointer type");
                let inner_layout = fx
                    .db
                    .layout_of_ty(inner_ty.store(), fx.env.clone())
                    .expect("deref layout error");

                // Load the pointer value from the current place
                let ptr_val = cplace.to_cvalue(fx).load_scalar(fx);
                cplace = CPlace::for_ptr(pointer::Pointer::new(ptr_val), inner_layout);
                cur_ty = inner_ty.store();
            }
            ProjectionElem::Downcast(variant_idx) => {
                let variant_layout = variant_layout(
                    fx.db,
                    &cur_ty,
                    &cplace.layout,
                    *variant_idx,
                    &fx.env,
                );
                cplace = cplace.downcast_variant(variant_layout);
                // cur_ty stays the same (Downcast is just a type assertion)
            }
            ProjectionElem::ClosureField(idx) => {
                // Closure captures are stored as fields of the closure struct
                let field_ty = closure_field_type(fx.db, &cur_ty, *idx);
                let field_layout = fx
                    .db
                    .layout_of_ty(field_ty.clone(), fx.env.clone())
                    .expect("closure field layout error");
                cplace = cplace.place_field(fx, *idx, field_layout);
                cur_ty = field_ty;
            }
            ProjectionElem::Index(_) => todo!("Index projection"),
            ProjectionElem::ConstantIndex { .. } => todo!("ConstantIndex projection"),
            ProjectionElem::Subslice { .. } => todo!("Subslice projection"),
            ProjectionElem::OpaqueCast(_) => todo!("OpaqueCast projection"),
        }
    }
    cplace
}

/// Get the type of a field from a parent type.
fn field_type(
    _db: &dyn HirDatabase,
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
        TyKind::Adt(_adt_id, _args) => {
            todo!("field_type for ADT types (coming in M3.3)")
        }
        _ => todo!("field_type for {:?}", parent_ty.as_ref().kind()),
    }
}

/// Get the type of a closure field (captured variable).
fn closure_field_type(
    _db: &dyn HirDatabase,
    _closure_ty: &StoredTy,
    _idx: usize,
) -> StoredTy {
    todo!("closure_field_type (coming in M3.3)")
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

// ---------------------------------------------------------------------------
// Statement codegen
// ---------------------------------------------------------------------------

fn codegen_statement(fx: &mut FunctionCx<'_, impl Module>, body: &MirBody, stmt: &StatementKind) {
    match stmt {
        StatementKind::Assign(place, rvalue) => {
            codegen_assign(fx, body, place, rvalue);
        }
        StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {}
        StatementKind::Nop | StatementKind::FakeRead(_) => {}
        StatementKind::Deinit(_) => {}
        StatementKind::SetDiscriminant { .. } => todo!("SetDiscriminant"),
    }
}

fn codegen_assign(fx: &mut FunctionCx<'_, impl Module>, body: &MirBody, place: &Place, rvalue: &Rvalue) {
    let dest = codegen_place(fx, body, place);
    if dest.layout.is_zst() {
        return;
    }

    // Some rvalues need to write directly to the destination place
    match rvalue {
        Rvalue::Aggregate(kind, operands) => {
            codegen_aggregate(fx, body, kind, operands, dest);
            return;
        }
        Rvalue::Ref(_, ref_place) | Rvalue::AddressOf(_, ref_place) => {
            let place = codegen_place(fx, body, ref_place);
            let ptr = place.to_ptr();
            let val = ptr.get_addr(&mut fx.bcx, fx.pointer_type);
            dest.write_cvalue(fx, CValue::by_val(val, dest.layout.clone()));
            return;
        }
        Rvalue::Discriminant(disc_place) => {
            let disc_cplace = codegen_place(fx, body, disc_place);
            let disc_val = codegen_get_discriminant(fx, &disc_cplace, &dest.layout);
            dest.write_cvalue(fx, CValue::by_val(disc_val, dest.layout.clone()));
            return;
        }
        _ => {}
    }

    let val = codegen_rvalue(fx, body, rvalue, &dest.layout);
    dest.write_cvalue(fx, val);
}

fn codegen_rvalue(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    rvalue: &Rvalue,
    result_layout: &LayoutArc,
) -> CValue {
    match rvalue {
        Rvalue::Use(operand) => codegen_operand(fx, body, &operand.kind),
        Rvalue::BinaryOp(op, lhs, rhs) => codegen_binop(fx, body, op, lhs, rhs, result_layout),
        Rvalue::UnaryOp(op, operand) => codegen_unop(fx, body, op, operand, result_layout),
        Rvalue::Cast(kind, operand, target_ty) => {
            codegen_cast(fx, body, kind, operand, target_ty, result_layout)
        }
        Rvalue::Len(place) => {
            // For fixed-size arrays, length is a constant
            let place_ty = place_ty(fx.db, body, place);
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
    body: &MirBody,
    kind: &hir_ty::mir::AggregateKind,
    operands: &[Operand],
    dest: CPlace,
) {
    use hir_ty::mir::AggregateKind;
    match kind {
        AggregateKind::Tuple(_) | AggregateKind::Array(_) | AggregateKind::Closure(_) => {
            // For ScalarPair tuples, construct directly as a pair
            if let BackendRepr::ScalarPair(_, _) = dest.layout.backend_repr {
                assert_eq!(operands.len(), 2, "ScalarPair aggregate expects 2 operands");
                let val0 = codegen_operand(fx, body, &operands[0].kind).load_scalar(fx);
                let val1 = codegen_operand(fx, body, &operands[1].kind).load_scalar(fx);
                dest.write_cvalue(
                    fx,
                    CValue::by_val_pair(val0, val1, dest.layout.clone()),
                );
                return;
            }

            // For single scalar, construct directly
            if let BackendRepr::Scalar(_) = dest.layout.backend_repr {
                assert_eq!(operands.len(), 1, "Scalar aggregate expects 1 operand");
                let val = codegen_operand(fx, body, &operands[0].kind);
                dest.write_cvalue(fx, val);
                return;
            }

            // General case: write each field to the destination place
            for (i, operand) in operands.iter().enumerate() {
                let field_cval = codegen_operand(fx, body, &operand.kind);
                let field_layout = field_cval.layout.clone();
                let field_place = dest.place_field(fx, i, field_layout);
                field_place.write_cvalue(fx, field_cval);
            }
        }
        AggregateKind::Adt(_, _) => {
            todo!("ADT aggregate (coming in M3.3)")
        }
        AggregateKind::Union(_, _) => {
            todo!("Union aggregate")
        }
        AggregateKind::RawPtr(_, _) => {
            todo!("RawPtr aggregate")
        }
    }
}

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
            // Single-variant enum or struct: for types with no explicit
            // discriminant values the variant index IS the discriminant.
            // TODO: enums with explicit discriminant values (e.g. `A = 100`)
            // need db.const_eval_discriminant() lookup here.
            let discr_val = index.as_u32();
            fx.bcx.ins().iconst(dest_clif_ty, i64::from(discr_val))
        }
        Variants::Multiple { tag, tag_field, tag_encoding, .. } => {
            use rustc_abi::TagEncoding;
            // Read the tag field
            let tag_offset = place.layout.fields.offset(tag_field.as_usize());
            let tag_clif_ty = scalar_to_clif_type(fx.dl, tag);
            let tag_ptr = place.to_ptr().offset_i64(
                &mut fx.bcx,
                fx.pointer_type,
                i64::try_from(tag_offset.bytes()).unwrap(),
            );
            let mut flags = MemFlags::new();
            flags.set_notrap();
            let tag_val = tag_ptr.load(&mut fx.bcx, tag_clif_ty, flags);

            match tag_encoding {
                TagEncoding::Direct => {
                    let signed = match tag.primitive() {
                        Primitive::Int(_, signed) => signed,
                        _ => false,
                    };
                    codegen_intcast(fx, tag_val, dest_clif_ty, signed)
                }
                TagEncoding::Niche { .. } => {
                    todo!("niche-encoded discriminant")
                }
            }
        }
        Variants::Empty => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
            // Unreachable, but need a value for cranelift
            fx.bcx.ins().iconst(dest_clif_ty, 0)
        }
    }
}

fn codegen_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    op: &BinOp,
    lhs: &Operand,
    rhs: &Operand,
    result_layout: &LayoutArc,
) -> CValue {
    let lhs_cval = codegen_operand(fx, body, &lhs.kind);
    let rhs_cval = codegen_operand(fx, body, &rhs.kind);
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
                let lhs_ty = operand_ty(fx.db, body, &lhs.kind);
                let pointee_ty = lhs_ty
                    .as_ref()
                    .builtin_deref(true)
                    .expect("Offset lhs must be a pointer/reference");
                let pointee_layout = fx
                    .db
                    .layout_of_ty(pointee_ty.store(), fx.env.clone())
                    .expect("layout error for pointee type");
                let ptr_ty = fx.bcx.func.dfg.value_type(lhs_val);
                let rhs_ty = fx.bcx.func.dfg.value_type(rhs_val);
                let rhs_signed = ty_is_signed_int(operand_ty(fx.db, body, &rhs.kind));
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
    body: &MirBody,
    op: &UnOp,
    operand: &Operand,
    result_layout: &LayoutArc,
) -> CValue {
    let cval = codegen_operand(fx, body, &operand.kind);
    let val = cval.load_scalar(fx);
    let result = match op {
        UnOp::Not => {
            let ty = operand_ty(fx.db, body, &operand.kind);
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
    body: &MirBody,
    kind: &OperandKind,
) -> CValue {
    match kind {
        OperandKind::Constant { konst, ty } => {
            let layout = fx
                .db
                .layout_of_ty(ty.clone(), fx.env.clone())
                .expect("layout error for constant type");
            if layout.is_zst() {
                return CValue::zst(layout);
            }
            if let BackendRepr::Scalar(scalar) = layout.backend_repr {
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
            } else {
                todo!("non-scalar constant")
            }
        }
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            codegen_place(fx, body, place).to_cvalue(fx)
        }
        OperandKind::Static(_) => todo!("static operand"),
    }
}

// ---------------------------------------------------------------------------
// Terminator codegen
// ---------------------------------------------------------------------------

fn codegen_terminator(fx: &mut FunctionCx<'_, impl Module>, body: &MirBody, term: &TerminatorKind) {
    match term {
        TerminatorKind::Return => {
            let ret_place = fx.local_place(hir_ty::mir::return_slot()).clone();
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
                    _ => todo!("return non-scalar/pair type"),
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
            let discr_cval = codegen_operand(fx, body, &discr.kind);
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
            codegen_call(fx, body, func, args, destination, target);
        }
        TerminatorKind::Drop { target, .. } => {
            // For scalar types, drop is a no-op — just jump to target.
            // TODO: call drop glue for types that need it.
            let block = fx.clif_block(*target);
            fx.bcx.ins().jump(block, &[]);
        }
        _ => todo!("terminator: {:?}", term),
    }
}

fn codegen_call(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    func: &Operand,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    // Extract the function type from the func operand
    let fn_ty = match &func.kind {
        OperandKind::Constant { ty, .. } => ty.as_ref(),
        _ => todo!("indirect/fn-pointer calls"),
    };

    match fn_ty.kind() {
        TyKind::FnDef(def, generic_args) => {
            let callable_def: CallableDefId = def.0;
            match callable_def {
                CallableDefId::FunctionId(callee_func_id) => {
                    codegen_direct_call(
                        fx,
                        body,
                        callee_func_id,
                        generic_args,
                        args,
                        destination,
                        target,
                    );
                }
                _ => todo!("struct/enum constructor calls"),
            }
        }
        _ => todo!("non-FnDef call (fn pointers, closures)"),
    }
}

fn codegen_direct_call(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    callee_func_id: hir_def::FunctionId,
    generic_args: GenericArgs<'_>,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) {
    if codegen_intrinsic_call(fx, body, callee_func_id, args, destination, target) {
        return;
    }

    // Check if this is an extern function (no MIR available)
    let is_extern = matches!(
        callee_func_id.loc(fx.db).container,
        ItemContainerId::ExternBlockId(_)
    );

    let (callee_sig, callee_name) = if is_extern {
        // Extern functions: build signature from type info, use raw symbol name
        let sig = build_fn_sig_from_ty(fx.isa, fx.db, fx.dl, &fx.env, callee_func_id)
            .expect("extern fn sig");
        let name = fx.db.function_signature(callee_func_id).name.as_str().to_owned();
        (sig, name)
    } else {
        // Regular functions: build signature from MIR, use v0 mangled name
        let callee_body = fx
            .db
            .monomorphized_mir_body(callee_func_id.into(), generic_args.store(), fx.env.clone())
            .expect("failed to get callee MIR");
        let sig =
            build_fn_sig(fx.isa, fx.db, fx.dl, &fx.env, &callee_body).expect("callee sig");
        let name = symbol_mangling::mangle_function(fx.db, callee_func_id, generic_args);
        (sig, name)
    };

    // Declare callee in module (Import linkage — it may be defined elsewhere or in same module)
    let callee_id = fx
        .module
        .declare_function(&callee_name, Linkage::Import, &callee_sig)
        .expect("declare callee");

    // Import into current function to get a FuncRef
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    // Build argument values (skip ZST args that have no Cranelift representation)
    let mut call_args: Vec<Value> = Vec::new();
    for arg in args {
        let cval = codegen_operand(fx, body, &arg.kind);
        if cval.layout.is_zst() {
            continue;
        }
        match cval.layout.backend_repr {
            BackendRepr::ScalarPair(_, _) => {
                let (a, b) = cval.load_scalar_pair(fx);
                call_args.push(a);
                call_args.push(b);
            }
            _ => {
                call_args.push(cval.load_scalar(fx));
            }
        }
    }

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);

    // Store return value into destination place
    let dest = codegen_place(fx, body, destination);
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
            _ => {
                // Memory return types: not yet handled
                if !results.is_empty() {
                    todo!("memory return types");
                }
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

fn codegen_intrinsic_call(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    callee_func_id: hir_def::FunctionId,
    args: &[Operand],
    destination: &Place,
    target: &Option<BasicBlockId>,
) -> bool {
    if !FunctionSignature::is_intrinsic(fx.db, callee_func_id) {
        return false;
    }

    let sig = fx.db.function_signature(callee_func_id);
    let name = sig.name.as_str();
    let result = match name {
        "offset" | "arith_offset" => {
            assert_eq!(args.len(), 2, "{name} intrinsic expects 2 args");

            let ptr_cval = codegen_operand(fx, body, &args[0].kind);
            let offset_cval = codegen_operand(fx, body, &args[1].kind);
            let ptr = ptr_cval.load_scalar(fx);
            let offset = offset_cval.load_scalar(fx);

            let ptr_ty = operand_ty(fx.db, body, &args[0].kind);
            let pointee_ty = ptr_ty
                .as_ref()
                .builtin_deref(true)
                .expect("offset intrinsic first argument must be a pointer");
            let pointee_layout = fx
                .db
                .layout_of_ty(pointee_ty.store(), fx.env.clone())
                .expect("layout error for offset intrinsic pointee");

            let ptr_clif_ty = fx.bcx.func.dfg.value_type(ptr);
            let offset_clif_ty = fx.bcx.func.dfg.value_type(offset);
            let offset_signed = ty_is_signed_int(operand_ty(fx.db, body, &args[1].kind));
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

            let ptr_cval = codegen_operand(fx, body, &args[0].kind);
            let base_cval = codegen_operand(fx, body, &args[1].kind);
            let ptr = ptr_cval.load_scalar(fx);
            let base = base_cval.load_scalar(fx);

            let ptr_ty = operand_ty(fx.db, body, &args[0].kind);
            let pointee_ty = ptr_ty
                .as_ref()
                .builtin_deref(true)
                .expect("ptr_offset_from first argument must be a pointer");
            let pointee_layout = fx
                .db
                .layout_of_ty(pointee_ty.store(), fx.env.clone())
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
        _ => return false,
    };

    // Write result to destination
    let dest = codegen_place(fx, body, destination);
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
fn build_fn_sig(
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
    match ret_layout.backend_repr {
        BackendRepr::Scalar(scalar) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        }
        BackendRepr::ScalarPair(a, b) => {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &a)));
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &b)));
        }
        _ => {
            // ZST or Memory — no return params
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
                return Err("unsupported param type: non-scalar, non-ZST, non-pair".into());
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
        if let BackendRepr::Scalar(scalar) = ret_layout.backend_repr {
            sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        }
        // ZST returns → no return param (already handled by not pushing)
    }

    // Parameter types
    for &param_ty in fn_sig.inputs_and_output.inputs() {
        let param_layout = db
            .layout_of_ty(param_ty.store(), env.clone())
            .map_err(|e| format!("param layout error: {:?}", e))?;
        if let BackendRepr::Scalar(scalar) = param_layout.backend_repr {
            sig.params.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
        } else if param_layout.is_zst() {
            // ZST params ignored
        } else {
            return Err("unsupported param type: non-scalar, non-ZST".into());
        }
    }

    Ok(sig)
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
        let mut block_map = ArenaMap::new();
        for (bb_id, _bb) in body.basic_blocks.iter() {
            let block = bcx.create_block();
            block_map.insert(bb_id, block);
        }

        // Set up entry block
        let entry_block = block_map[body.start_block];
        bcx.switch_to_block(entry_block);
        bcx.append_block_params_for_function_params(entry_block);

        // Build a temporary FunctionCx for local setup (needed by CPlace constructors)
        let mut local_map = ArenaMap::new();

        // First pass: create CPlace for each local using direct bcx access
        // (We can't use FunctionCx yet because we're still building it)
        for (local_id, local) in body.locals.iter() {
            let local_layout = db
                .layout_of_ty(local.ty.clone(), env.clone())
                .map_err(|e| format!("local layout error: {:?}", e))?;

            let place = match local_layout.backend_repr {
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
            local_map.insert(local_id, place);
        }

        // Wire function parameters to their locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;
        for &param_local in &body.param_locals {
            let place = &local_map[param_local];
            if place.layout.is_zst() {
                continue;
            }
            match place.layout.backend_repr {
                BackendRepr::Scalar(_) => {
                    place.def_var(0, block_params[param_idx], &mut bcx);
                    param_idx += 1;
                }
                BackendRepr::ScalarPair(_, _) => {
                    place.def_var(0, block_params[param_idx], &mut bcx);
                    place.def_var(1, block_params[param_idx + 1], &mut bcx);
                    param_idx += 2;
                }
                _ => {
                    // Stack slot params: not yet handled via ABI
                    // For now, skip non-scalar/pair params
                }
            }
        }

        let mut fx = FunctionCx {
            bcx,
            module,
            isa,
            pointer_type,
            db,
            dl,
            env: env.clone(),
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
                codegen_statement(&mut fx, body, &stmt.kind);
            }

            if let Some(term) = &bb.terminator {
                codegen_terminator(&mut fx, body, &term.kind);
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
) -> Result<Vec<u8>, String> {
    let isa = build_host_isa(true);

    let builder =
        ObjectBuilder::new(isa.clone(), "rac_output", cranelift_module::default_libcall_names())
            .map_err(|e| format!("ObjectBuilder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    compile_fn(&mut module, &*isa, db, dl, env, body, fn_name, Linkage::Export)?;

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

/// Compile `fn main() {}` to an executable: compile → emit entry point → link.
pub fn compile_executable(
    db: &dyn HirDatabase,
    dl: &TargetDataLayout,
    env: &StoredParamEnvAndCrate,
    main_body: &MirBody,
    main_func_id: hir_def::FunctionId,
    generic_args: hir_ty::next_solver::GenericArgs<'_>,
    output_path: &Path,
) -> Result<(), String> {
    let isa = build_host_isa(true);

    let builder =
        ObjectBuilder::new(isa.clone(), "rac_output", cranelift_module::default_libcall_names())
            .map_err(|e| format!("ObjectBuilder: {e}"))?;
    let mut module = ObjectModule::new(builder);

    let fn_name = symbol_mangling::mangle_function(db, main_func_id, generic_args);
    let user_main_id =
        compile_fn(&mut module, &*isa, db, dl, env, main_body, &fn_name, Linkage::Export)?;

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

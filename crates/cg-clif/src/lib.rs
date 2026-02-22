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
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Switch, Variable};
use cranelift_module::{FuncId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use hir_def::CallableDefId;
use hir_def::signatures::FunctionSignature;
use hir_ty::db::HirDatabase;
use hir_ty::mir::{
    BasicBlockId, BinOp, CastKind, LocalId, MirBody, Operand, OperandKind, Place, Rvalue,
    StatementKind,
    TerminatorKind, UnOp,
};
use hir_ty::next_solver::{Const, ConstKind, GenericArgs, IntoKind, StoredTy, TyKind};
use hir_ty::traits::StoredParamEnvAndCrate;
use la_arena::ArenaMap;
use rustc_abi::{BackendRepr, Primitive, Scalar, Size, TargetDataLayout};

pub mod link;
mod pointer;
pub mod symbol_mangling;

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
    /// MIR local → Cranelift variable (for scalar locals) or ZST
    local_map: ArenaMap<LocalId, LocalStorage>,
}

#[derive(Clone)]
enum LocalStorage {
    Var(Variable, Type),
    Zst,
    // Stack(StackSlot),
}

impl<M: Module> FunctionCx<'_, M> {
    fn clif_block(&self, bb: BasicBlockId) -> Block {
        self.block_map[bb]
    }

    fn local_storage(&self, local: LocalId) -> &LocalStorage {
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
fn operand_ty(body: &MirBody, kind: &OperandKind) -> StoredTy {
    match kind {
        OperandKind::Constant { ty, .. } => ty.clone(),
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            assert!(place.projection.is_empty(), "projections not yet supported");
            body.locals[place.local].ty.clone()
        }
        OperandKind::Static(_) => todo!("static operand type"),
    }
}

/// Get the scalar representation of an operand's type.
fn operand_scalar(fx: &FunctionCx<'_, impl Module>, body: &MirBody, kind: &OperandKind) -> Scalar {
    let ty = operand_ty(body, kind);
    let layout = fx.db.layout_of_ty(ty, fx.env.clone()).expect("layout error");
    match layout.backend_repr {
        BackendRepr::Scalar(scalar) => scalar,
        _ => panic!("expected scalar type for operand"),
    }
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
) -> Value {
    let from_ty = operand_ty(body, &operand.kind);
    let from_val = codegen_operand(fx, body, &operand.kind);
    let from_clif_ty = fx.bcx.func.dfg.value_type(from_val);

    let target_layout = fx
        .db
        .layout_of_ty(target_ty.clone(), fx.env.clone())
        .expect("layout error for cast target");
    let BackendRepr::Scalar(target_scalar) = target_layout.backend_repr else {
        todo!("cast target must be scalar")
    };
    let target_clif_ty = scalar_to_clif_type(fx.dl, &target_scalar);

    match kind {
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
            let from_layout = fx
                .db
                .layout_of_ty(from_ty.clone(), fx.env.clone())
                .expect("layout error for transmute source");
            assert_eq!(
                from_layout.size, target_layout.size,
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
    assert!(place.projection.is_empty(), "projections not yet supported");

    match fx.local_storage(place.local) {
        LocalStorage::Zst => return,
        LocalStorage::Var(_, _) => {}
    }

    let val = codegen_rvalue(fx, body, rvalue);

    match fx.local_storage(place.local) {
        LocalStorage::Var(var, _) => {
            fx.bcx.def_var(*var, val);
        }
        LocalStorage::Zst => unreachable!(),
    }
}

fn codegen_rvalue(fx: &mut FunctionCx<'_, impl Module>, body: &MirBody, rvalue: &Rvalue) -> Value {
    match rvalue {
        Rvalue::Use(operand) => codegen_operand(fx, body, &operand.kind),
        Rvalue::BinaryOp(op, lhs, rhs) => codegen_binop(fx, body, op, lhs, rhs),
        Rvalue::UnaryOp(op, operand) => codegen_unop(fx, body, op, operand),
        Rvalue::Cast(kind, operand, target_ty) => codegen_cast(fx, body, kind, operand, target_ty),
        _ => todo!("rvalue: {:?}", rvalue),
    }
}

fn codegen_binop(
    fx: &mut FunctionCx<'_, impl Module>,
    body: &MirBody,
    op: &BinOp,
    lhs: &Operand,
    rhs: &Operand,
) -> Value {
    let lhs_val = codegen_operand(fx, body, &lhs.kind);
    let rhs_val = codegen_operand(fx, body, &rhs.kind);
    let scalar = operand_scalar(fx, body, &lhs.kind);

    match scalar.primitive() {
        Primitive::Int(_, signed) => codegen_int_binop(fx, op, lhs_val, rhs_val, signed),
        Primitive::Float(_) => codegen_float_binop(fx, op, lhs_val, rhs_val),
        Primitive::Pointer(_) => match op {
            BinOp::Offset => {
                let lhs_ty = operand_ty(body, &lhs.kind);
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
                let rhs_signed = ty_is_signed_int(operand_ty(body, &rhs.kind));
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
    }
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
            todo!("overflow binops (need CValue pairs)")
        }
        BinOp::Offset => unreachable!("Offset on integer type"),
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
) -> Value {
    let val = codegen_operand(fx, body, &operand.kind);
    match op {
        UnOp::Not => {
            let ty = operand_ty(body, &operand.kind);
            if ty.as_ref().kind() == TyKind::Bool {
                fx.bcx.ins().icmp_imm(IntCC::Equal, val, 0)
            } else {
                fx.bcx.ins().bnot(val)
            }
        }
        UnOp::Neg => {
            let scalar = operand_scalar(fx, body, &operand.kind);
            match scalar.primitive() {
                Primitive::Int(_, _) => fx.bcx.ins().ineg(val),
                Primitive::Float(_) => fx.bcx.ins().fneg(val),
                Primitive::Pointer(_) => unreachable!("neg on pointer"),
            }
        }
    }
}

fn codegen_operand(fx: &mut FunctionCx<'_, impl Module>, _body: &MirBody, kind: &OperandKind) -> Value {
    match kind {
        OperandKind::Constant { konst, ty } => {
            let layout = fx
                .db
                .layout_of_ty(ty.clone(), fx.env.clone())
                .expect("layout error for constant type");
            if let BackendRepr::Scalar(scalar) = layout.backend_repr {
                let raw = const_to_i64(konst.as_ref(), scalar.size(fx.dl));
                match scalar.primitive() {
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
                }
            } else {
                todo!("non-scalar constant")
            }
        }
        OperandKind::Copy(place) | OperandKind::Move(place) => {
            assert!(place.projection.is_empty(), "projections not yet supported");
            match fx.local_storage(place.local) {
                LocalStorage::Var(var, _) => fx.bcx.use_var(*var),
                LocalStorage::Zst => unreachable!("cannot produce a Value from ZST local"),
            }
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
            let ret_local = hir_ty::mir::return_slot();
            match fx.local_storage(ret_local) {
                LocalStorage::Zst => {
                    fx.bcx.ins().return_(&[]);
                }
                LocalStorage::Var(var, _) => {
                    let val = fx.bcx.use_var(*var);
                    fx.bcx.ins().return_(&[val]);
                }
            }
        }
        TerminatorKind::Goto { target } => {
            let block = fx.clif_block(*target);
            fx.bcx.ins().jump(block, &[]);
        }
        TerminatorKind::Unreachable => {
            fx.bcx.ins().trap(cranelift_codegen::ir::TrapCode::user(0).unwrap());
        }
        TerminatorKind::SwitchInt { discr, targets } => {
            let discr_val = codegen_operand(fx, body, &discr.kind);
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

    // Get callee MIR to determine its signature
    let callee_body = fx
        .db
        .monomorphized_mir_body(callee_func_id.into(), generic_args.store(), fx.env.clone())
        .expect("failed to get callee MIR");

    // Build callee's Cranelift signature
    let callee_sig =
        build_fn_sig(fx.isa, fx.db, fx.dl, &fx.env, &callee_body).expect("callee sig");

    // Get callee name using v0 symbol mangling
    let callee_name = symbol_mangling::mangle_function(fx.db, callee_func_id, generic_args);

    // Declare callee in module (Import linkage — it may be defined elsewhere or in same module)
    let callee_id = fx
        .module
        .declare_function(&callee_name, Linkage::Import, &callee_sig)
        .expect("declare callee");

    // Import into current function to get a FuncRef
    let callee_ref = fx.module.declare_func_in_func(callee_id, fx.bcx.func);

    // Build argument values (skip ZST args that have no Cranelift representation)
    let call_args: Vec<Value> = args
        .iter()
        .filter_map(|arg| {
            let ty = operand_ty(body, &arg.kind);
            let layout = fx
                .db
                .layout_of_ty(ty, fx.env.clone())
                .expect("arg layout");
            if layout.is_zst() {
                None
            } else {
                Some(codegen_operand(fx, body, &arg.kind))
            }
        })
        .collect();

    // Emit the call
    let call = fx.bcx.ins().call(callee_ref, &call_args);

    // Store return value
    let results = fx.bcx.inst_results(call);
    write_call_destination(fx, destination, results.get(0).copied());

    // Jump to continuation block
    if let Some(target) = target {
        let block = fx.clif_block(*target);
        fx.bcx.ins().jump(block, &[]);
    }
}

fn write_call_destination(
    fx: &mut FunctionCx<'_, impl Module>,
    destination: &Place,
    result: Option<Value>,
) {
    assert!(
        destination.projection.is_empty(),
        "call destination projections not yet supported"
    );
    match fx.local_storage(destination.local) {
        LocalStorage::Zst => {}
        LocalStorage::Var(var, _) => {
            if let Some(val) = result {
                fx.bcx.def_var(*var, val);
            }
        }
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

            let ptr = codegen_operand(fx, body, &args[0].kind);
            let offset = codegen_operand(fx, body, &args[1].kind);

            let ptr_ty = operand_ty(body, &args[0].kind);
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
            let offset_signed = ty_is_signed_int(operand_ty(body, &args[1].kind));
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

            let ptr = codegen_operand(fx, body, &args[0].kind);
            let base = codegen_operand(fx, body, &args[1].kind);

            let ptr_ty = operand_ty(body, &args[0].kind);
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

    write_call_destination(fx, destination, result);
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
    if let BackendRepr::Scalar(scalar) = ret_layout.backend_repr {
        sig.returns.push(AbiParam::new(scalar_to_clif_type(dl, &scalar)));
    }

    // Parameter types
    for &param_local in &body.param_locals {
        let param = &body.locals[param_local];
        let param_layout = db
            .layout_of_ty(param.ty.clone(), env.clone())
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

        // Set up locals as Cranelift variables
        let mut local_map = ArenaMap::new();

        for (local_id, local) in body.locals.iter() {
            let local_layout = db
                .layout_of_ty(local.ty.clone(), env.clone())
                .map_err(|e| format!("local layout error: {:?}", e))?;

            if let BackendRepr::Scalar(scalar) = local_layout.backend_repr {
                let clif_ty = scalar_to_clif_type(dl, &scalar);
                let var = bcx.declare_var(clif_ty);
                local_map.insert(local_id, LocalStorage::Var(var, clif_ty));
            } else if local_layout.is_zst() {
                local_map.insert(local_id, LocalStorage::Zst);
            } else {
                // TODO: stack slot for non-scalar types
                return Err(format!("unsupported local type: non-scalar, non-ZST"));
            }
        }

        // Wire function parameters to their locals
        let block_params: Vec<Value> = bcx.block_params(entry_block).to_vec();
        let mut param_idx = 0;
        for &param_local in &body.param_locals {
            match &local_map[param_local] {
                LocalStorage::Var(var, _) => {
                    bcx.def_var(*var, block_params[param_idx]);
                    param_idx += 1;
                }
                LocalStorage::Zst => {}
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

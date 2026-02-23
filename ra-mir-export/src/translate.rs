//! Translation from rustc MIR types to our serializable MIR types.

use rustc_middle::mir;
use rustc_middle::ty::{self, TyCtxt};
use rustc_hir::def_id::DefId;

use ra_mir_types as mir_types;

/// Ensure sufficient stack space for recursive translation.
/// Grows the stack if less than 256KB remains.
fn ensure_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, f)
}

// ---------------------------------------------------------------------------
// DefPathHash extraction
// ---------------------------------------------------------------------------

pub fn def_path_hash(tcx: TyCtxt<'_>, def_id: DefId) -> mir_types::DefPathHash {
    let hash = tcx.def_path_hash(def_id);
    let (crate_hash, local_hash) = hash.0.split();
    (crate_hash.as_u64(), local_hash.as_u64())
}

// ---------------------------------------------------------------------------
// Body
// ---------------------------------------------------------------------------

pub fn translate_body<'tcx>(tcx: TyCtxt<'tcx>, body: &mir::Body<'tcx>) -> mir_types::Body {
    let locals = body
        .local_decls
        .iter()
        .map(|decl| mir_types::Local {
            ty: translate_ty(tcx, decl.ty),
            layout: None,
        })
        .collect();

    let blocks = body
        .basic_blocks
        .iter()
        .map(|bb| translate_basic_block(tcx, bb))
        .collect();

    mir_types::Body {
        locals,
        arg_count: body.arg_count as u32,
        blocks,
    }
}

fn translate_basic_block<'tcx>(
    tcx: TyCtxt<'tcx>,
    bb: &mir::BasicBlockData<'tcx>,
) -> mir_types::BasicBlock {
    let stmts = bb
        .statements
        .iter()
        .filter_map(|stmt| translate_statement(tcx, stmt))
        .collect();

    let terminator = bb
        .terminator
        .as_ref()
        .map(|t| translate_terminator(tcx, t))
        .unwrap_or(mir_types::Terminator::Unreachable);

    mir_types::BasicBlock {
        stmts,
        terminator,
        is_cleanup: bb.is_cleanup,
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

fn translate_statement<'tcx>(
    tcx: TyCtxt<'tcx>,
    stmt: &mir::Statement<'tcx>,
) -> Option<mir_types::Statement> {
    use mir::StatementKind::*;
    Some(match &stmt.kind {
        Assign(box (place, rvalue)) => {
            mir_types::Statement::Assign(
                translate_place(tcx, place),
                translate_rvalue(tcx, rvalue),
            )
        }
        SetDiscriminant { place, variant_index } => {
            mir_types::Statement::SetDiscriminant {
                place: translate_place(tcx, place),
                variant_index: variant_index.as_u32(),
            }
        }
        StorageLive(local) => mir_types::Statement::StorageLive(local.as_u32()),
        StorageDead(local) => mir_types::Statement::StorageDead(local.as_u32()),
        Nop => mir_types::Statement::Nop,
        // Skip statements we don't need in codegen
        FakeRead(_)
        | PlaceMention(_)
        | AscribeUserType(_, _)
        | Coverage(_)
        | ConstEvalCounter
        | Retag(_, _)
        | Intrinsic(_)
        | BackwardIncompatibleDropHint { .. } => return None,
    })
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

fn translate_terminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    term: &mir::Terminator<'tcx>,
) -> mir_types::Terminator {
    use mir::TerminatorKind::*;
    match &term.kind {
        Goto { target } => mir_types::Terminator::Goto(target.as_u32()),
        SwitchInt { discr, targets } => {
            let values: Vec<u128> = targets.iter().map(|(v, _)| v).collect();
            let target_blocks: Vec<u32> = targets
                .all_targets()
                .iter()
                .map(|bb| bb.as_u32())
                .collect();
            mir_types::Terminator::SwitchInt {
                discr: translate_operand(tcx, discr),
                targets: mir_types::SwitchTargets {
                    values,
                    targets: target_blocks,
                },
            }
        }
        Return => mir_types::Terminator::Return,
        Unreachable => mir_types::Terminator::Unreachable,
        UnwindResume => mir_types::Terminator::UnwindResume,
        UnwindTerminate(_) => mir_types::Terminator::Unreachable,
        Drop { place, target, unwind, .. } => {
            mir_types::Terminator::Drop {
                place: translate_place(tcx, place),
                target: target.as_u32(),
                unwind: translate_unwind(unwind),
            }
        }
        Call { func, args, destination, target, unwind, .. } => {
            mir_types::Terminator::Call {
                func: translate_operand(tcx, func),
                args: args.iter().map(|a| translate_operand(tcx, &a.node)).collect(),
                dest: translate_place(tcx, destination),
                target: target.map(|bb| bb.as_u32()),
                unwind: translate_unwind(unwind),
            }
        }
        Assert { cond, expected, target, unwind, .. } => {
            mir_types::Terminator::Assert {
                cond: translate_operand(tcx, cond),
                expected: *expected,
                target: target.as_u32(),
                unwind: translate_unwind(unwind),
            }
        }
        // Constructs we skip/fallback
        TailCall { .. }
        | Yield { .. }
        | CoroutineDrop
        | FalseEdge { .. }
        | FalseUnwind { .. }
        | InlineAsm { .. } => mir_types::Terminator::Unreachable,
    }
}

fn translate_unwind(unwind: &mir::UnwindAction) -> mir_types::UnwindAction {
    match unwind {
        mir::UnwindAction::Continue => mir_types::UnwindAction::Continue,
        mir::UnwindAction::Unreachable => mir_types::UnwindAction::Unreachable,
        mir::UnwindAction::Terminate(_) => mir_types::UnwindAction::Terminate,
        mir::UnwindAction::Cleanup(bb) => mir_types::UnwindAction::Cleanup(bb.as_u32()),
    }
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

fn translate_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    op: &mir::Operand<'tcx>,
) -> mir_types::Operand {
    match op {
        mir::Operand::Copy(place) => {
            mir_types::Operand::Copy(translate_place(tcx, place))
        }
        mir::Operand::Move(place) => {
            mir_types::Operand::Move(translate_place(tcx, place))
        }
        mir::Operand::Constant(box c) => {
            mir_types::Operand::Constant(translate_const_operand(tcx, c))
        }
        // RuntimeChecks operand - treat as a constant true
        _ => {
            mir_types::Operand::Constant(mir_types::ConstOperand {
                ty: mir_types::Ty::Bool,
                kind: mir_types::ConstKind::Scalar(1, 1),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

fn translate_const_operand<'tcx>(
    tcx: TyCtxt<'tcx>,
    c: &mir::ConstOperand<'tcx>,
) -> mir_types::ConstOperand {
    let ty = translate_ty(tcx, c.const_.ty());
    let kind = translate_const(tcx, c.const_);
    mir_types::ConstOperand { ty, kind }
}

fn translate_const<'tcx>(tcx: TyCtxt<'tcx>, c: mir::Const<'tcx>) -> mir_types::ConstKind {
    ensure_stack(|| translate_const_inner(tcx, c))
}

fn translate_const_inner<'tcx>(tcx: TyCtxt<'tcx>, c: mir::Const<'tcx>) -> mir_types::ConstKind {
    match c {
        mir::Const::Val(val, _ty) => translate_const_value(tcx, val),
        mir::Const::Ty(_ty, ct) => {
            // Type-system constant: try to extract a scalar leaf
            if let Some(scalar_int) = ct.try_to_leaf() {
                let data = scalar_int.to_bits(scalar_int.size());
                let size = scalar_int.size().bytes() as u8;
                mir_types::ConstKind::Scalar(data, size)
            } else {
                mir_types::ConstKind::Todo(format!("{ct:?}"))
            }
        }
        mir::Const::Unevaluated(uneval, _ty) => {
            let def_hash = def_path_hash(tcx, uneval.def);
            let args = translate_generic_args(tcx, uneval.args);
            mir_types::ConstKind::Unevaluated(def_hash, args)
        }
    }
}

fn translate_const_value(
    tcx: TyCtxt<'_>,
    val: mir::ConstValue,
) -> mir_types::ConstKind {
    match val {
        mir::ConstValue::Scalar(scalar) => {
            match scalar {
                rustc_middle::mir::interpret::Scalar::Int(scalar_int) => {
                    let data = scalar_int.to_bits(scalar_int.size());
                    let size = scalar_int.size().bytes() as u8;
                    mir_types::ConstKind::Scalar(data, size)
                }
                rustc_middle::mir::interpret::Scalar::Ptr(ptr, _size) => {
                    // Pointer scalar - try to resolve what it points to
                    let alloc_id = ptr.provenance.alloc_id();
                    match tcx.global_alloc(alloc_id) {
                        rustc_middle::mir::interpret::GlobalAlloc::Function { instance, .. } => {
                            let def_hash = def_path_hash(tcx, instance.def_id());
                            let args = translate_generic_args(tcx, instance.args);
                            mir_types::ConstKind::Unevaluated(def_hash, args)
                        }
                        _ => {
                            mir_types::ConstKind::Todo(format!("ptr:{alloc_id:?}"))
                        }
                    }
                }
            }
        }
        mir::ConstValue::ZeroSized => mir_types::ConstKind::ZeroSized,
        mir::ConstValue::Slice { alloc_id, meta } => {
            // Try to read the slice data from the allocation
            match tcx.global_alloc(alloc_id) {
                rustc_middle::mir::interpret::GlobalAlloc::Memory(alloc) => {
                    let alloc = alloc.inner();
                    let len = meta as usize;
                    let bytes: Vec<u8> = (0..len)
                        .filter_map(|i| {
                            alloc.inspect_with_uninit_and_ptr_outside_interpreter(
                                i..i + 1,
                            ).first().copied()
                        })
                        .collect();
                    mir_types::ConstKind::Slice(bytes, meta)
                }
                _ => mir_types::ConstKind::Todo(format!("slice:{alloc_id:?}")),
            }
        }
        mir::ConstValue::Indirect { .. } => {
            mir_types::ConstKind::Todo("indirect_const".to_string())
        }
    }
}

// ---------------------------------------------------------------------------
// Places
// ---------------------------------------------------------------------------

fn translate_place<'tcx>(tcx: TyCtxt<'tcx>, place: &mir::Place<'tcx>) -> mir_types::Place {
    let projections = place
        .projection
        .iter()
        .map(|elem| translate_projection(tcx, elem))
        .collect();

    mir_types::Place {
        local: place.local.as_u32(),
        projections,
    }
}

fn translate_projection<'tcx>(
    tcx: TyCtxt<'tcx>,
    elem: mir::PlaceElem<'tcx>,
) -> mir_types::Projection {
    match elem {
        mir::ProjectionElem::Deref => mir_types::Projection::Deref,
        mir::ProjectionElem::Field(idx, ty) => {
            mir_types::Projection::Field(idx.as_u32(), translate_ty(tcx, ty))
        }
        mir::ProjectionElem::Index(local) => {
            mir_types::Projection::Index(local.as_u32())
        }
        mir::ProjectionElem::ConstantIndex { offset, min_length, from_end } => {
            mir_types::Projection::ConstantIndex { offset, min_length, from_end }
        }
        mir::ProjectionElem::Subslice { from, to, from_end } => {
            mir_types::Projection::Subslice { from, to, from_end }
        }
        mir::ProjectionElem::Downcast(_, variant_idx) => {
            mir_types::Projection::Downcast(variant_idx.as_u32())
        }
        mir::ProjectionElem::OpaqueCast(ty) => {
            mir_types::Projection::OpaqueCast(translate_ty(tcx, ty))
        }
        mir::ProjectionElem::UnwrapUnsafeBinder(ty) => {
            // Treat like opaque cast for now
            mir_types::Projection::OpaqueCast(translate_ty(tcx, ty))
        }
    }
}

// ---------------------------------------------------------------------------
// Rvalues
// ---------------------------------------------------------------------------

fn translate_rvalue<'tcx>(tcx: TyCtxt<'tcx>, rv: &mir::Rvalue<'tcx>) -> mir_types::Rvalue {
    match rv {
        mir::Rvalue::Use(op) => {
            mir_types::Rvalue::Use(translate_operand(tcx, op))
        }
        mir::Rvalue::Repeat(op, ct) => {
            let count = ct.try_to_target_usize(tcx).unwrap_or(0);
            mir_types::Rvalue::Repeat(translate_operand(tcx, op), count)
        }
        mir::Rvalue::Ref(_, borrow_kind, place) => {
            mir_types::Rvalue::Ref(
                translate_borrow_kind(borrow_kind),
                translate_place(tcx, place),
            )
        }
        mir::Rvalue::ThreadLocalRef(def_id) => {
            mir_types::Rvalue::ThreadLocalRef(def_path_hash(tcx, *def_id))
        }
        mir::Rvalue::RawPtr(kind, place) => {
            let mutbl = match kind {
                mir::RawPtrKind::Mut => mir_types::Mutability::Mut,
                mir::RawPtrKind::Const | mir::RawPtrKind::FakeForPtrMetadata => {
                    mir_types::Mutability::Not
                }
            };
            mir_types::Rvalue::RawPtr(mutbl, translate_place(tcx, place))
        }
        mir::Rvalue::Cast(kind, op, ty) => {
            mir_types::Rvalue::Cast(
                translate_cast_kind(kind),
                translate_operand(tcx, op),
                translate_ty(tcx, *ty),
            )
        }
        mir::Rvalue::BinaryOp(bin_op, box (lhs, rhs)) => {
            mir_types::Rvalue::BinaryOp(
                translate_bin_op(*bin_op),
                translate_operand(tcx, lhs),
                translate_operand(tcx, rhs),
            )
        }
        mir::Rvalue::UnaryOp(un_op, op) => {
            mir_types::Rvalue::UnaryOp(
                translate_un_op(*un_op),
                translate_operand(tcx, op),
            )
        }
        mir::Rvalue::Discriminant(place) => {
            mir_types::Rvalue::Discriminant(translate_place(tcx, place))
        }
        mir::Rvalue::Aggregate(box kind, fields) => {
            let agg_kind = translate_aggregate_kind(tcx, kind);
            let ops: Vec<_> = fields.iter().map(|op| translate_operand(tcx, op)).collect();
            mir_types::Rvalue::Aggregate(agg_kind, ops)
        }
        mir::Rvalue::CopyForDeref(place) => {
            mir_types::Rvalue::CopyForDeref(translate_place(tcx, place))
        }
        mir::Rvalue::WrapUnsafeBinder(op, _ty) => {
            // Treat as Use for now
            mir_types::Rvalue::Use(translate_operand(tcx, op))
        }
    }
}

fn translate_aggregate_kind<'tcx>(
    tcx: TyCtxt<'tcx>,
    kind: &mir::AggregateKind<'tcx>,
) -> mir_types::AggregateKind {
    match kind {
        mir::AggregateKind::Array(ty) => {
            mir_types::AggregateKind::Array(translate_ty(tcx, *ty))
        }
        mir::AggregateKind::Tuple => mir_types::AggregateKind::Tuple,
        mir::AggregateKind::Adt(def_id, variant_idx, args, _, _) => {
            mir_types::AggregateKind::Adt(
                def_path_hash(tcx, *def_id),
                variant_idx.as_u32(),
                translate_generic_args(tcx, args),
            )
        }
        mir::AggregateKind::Closure(def_id, args) => {
            mir_types::AggregateKind::Closure(
                def_path_hash(tcx, *def_id),
                translate_generic_args(tcx, args),
            )
        }
        mir::AggregateKind::Coroutine(def_id, args) => {
            // Treat coroutine like closure
            mir_types::AggregateKind::Closure(
                def_path_hash(tcx, *def_id),
                translate_generic_args(tcx, args),
            )
        }
        mir::AggregateKind::CoroutineClosure(def_id, args) => {
            mir_types::AggregateKind::Closure(
                def_path_hash(tcx, *def_id),
                translate_generic_args(tcx, args),
            )
        }
        mir::AggregateKind::RawPtr(ty, mutbl) => {
            mir_types::AggregateKind::RawPtr(
                translate_ty(tcx, *ty),
                translate_mutability(*mutbl),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

fn translate_bin_op(op: mir::BinOp) -> mir_types::BinOp {
    match op {
        mir::BinOp::Add => mir_types::BinOp::Add,
        mir::BinOp::Sub => mir_types::BinOp::Sub,
        mir::BinOp::Mul => mir_types::BinOp::Mul,
        mir::BinOp::Div => mir_types::BinOp::Div,
        mir::BinOp::Rem => mir_types::BinOp::Rem,
        mir::BinOp::BitXor => mir_types::BinOp::BitXor,
        mir::BinOp::BitAnd => mir_types::BinOp::BitAnd,
        mir::BinOp::BitOr => mir_types::BinOp::BitOr,
        mir::BinOp::Shl => mir_types::BinOp::Shl,
        mir::BinOp::Shr => mir_types::BinOp::Shr,
        mir::BinOp::Eq => mir_types::BinOp::Eq,
        mir::BinOp::Lt => mir_types::BinOp::Lt,
        mir::BinOp::Le => mir_types::BinOp::Le,
        mir::BinOp::Ne => mir_types::BinOp::Ne,
        mir::BinOp::Ge => mir_types::BinOp::Ge,
        mir::BinOp::Gt => mir_types::BinOp::Gt,
        mir::BinOp::Cmp => mir_types::BinOp::Cmp,
        mir::BinOp::Offset => mir_types::BinOp::Offset,
        mir::BinOp::AddWithOverflow => mir_types::BinOp::AddWithOverflow,
        mir::BinOp::SubWithOverflow => mir_types::BinOp::SubWithOverflow,
        mir::BinOp::MulWithOverflow => mir_types::BinOp::MulWithOverflow,
        mir::BinOp::AddUnchecked => mir_types::BinOp::AddUnchecked,
        mir::BinOp::SubUnchecked => mir_types::BinOp::SubUnchecked,
        mir::BinOp::MulUnchecked => mir_types::BinOp::MulUnchecked,
        mir::BinOp::ShlUnchecked => mir_types::BinOp::ShlUnchecked,
        mir::BinOp::ShrUnchecked => mir_types::BinOp::ShrUnchecked,
    }
}

fn translate_un_op(op: mir::UnOp) -> mir_types::UnOp {
    match op {
        mir::UnOp::Not => mir_types::UnOp::Not,
        mir::UnOp::Neg => mir_types::UnOp::Neg,
        mir::UnOp::PtrMetadata => mir_types::UnOp::PtrMetadata,
    }
}

fn translate_cast_kind(kind: &mir::CastKind) -> mir_types::CastKind {
    match kind {
        mir::CastKind::IntToInt => mir_types::CastKind::IntToInt,
        mir::CastKind::FloatToInt => mir_types::CastKind::FloatToInt,
        mir::CastKind::IntToFloat => mir_types::CastKind::IntToFloat,
        mir::CastKind::FloatToFloat => mir_types::CastKind::FloatToFloat,
        mir::CastKind::PtrToPtr => mir_types::CastKind::PtrToPtr,
        mir::CastKind::FnPtrToPtr => mir_types::CastKind::FnPtrToPtr,
        mir::CastKind::PointerExposeProvenance => mir_types::CastKind::PointerExposeProvenance,
        mir::CastKind::PointerWithExposedProvenance => {
            mir_types::CastKind::PointerWithExposedProvenance
        }
        mir::CastKind::PointerCoercion(coercion, _source) => {
            mir_types::CastKind::PointerCoercion(translate_pointer_coercion(coercion))
        }
        mir::CastKind::Transmute => mir_types::CastKind::Transmute,
        // Subtype casts are identity at the MIR level
        mir::CastKind::Subtype => mir_types::CastKind::Transmute,
    }
}

fn translate_pointer_coercion(
    coercion: &ty::adjustment::PointerCoercion,
) -> mir_types::PointerCoercion {
    match coercion {
        ty::adjustment::PointerCoercion::ReifyFnPointer(_) => {
            mir_types::PointerCoercion::ReifyFnPointer
        }
        ty::adjustment::PointerCoercion::UnsafeFnPointer => {
            mir_types::PointerCoercion::UnsafeFnPointer
        }
        ty::adjustment::PointerCoercion::ClosureFnPointer(_) => {
            mir_types::PointerCoercion::ClosureFnPointer
        }
        ty::adjustment::PointerCoercion::MutToConstPointer => {
            mir_types::PointerCoercion::MutToConstPointer
        }
        ty::adjustment::PointerCoercion::ArrayToPointer => {
            mir_types::PointerCoercion::ArrayToPointer
        }
        ty::adjustment::PointerCoercion::Unsize => {
            mir_types::PointerCoercion::Unsize
        }
    }
}

fn translate_borrow_kind(kind: &mir::BorrowKind) -> mir_types::BorrowKind {
    match kind {
        mir::BorrowKind::Shared => mir_types::BorrowKind::Shared,
        mir::BorrowKind::Mut { .. } => mir_types::BorrowKind::Mut,
        mir::BorrowKind::Fake(mir::FakeBorrowKind::Shallow) => mir_types::BorrowKind::Shallow,
        mir::BorrowKind::Fake(mir::FakeBorrowKind::Deep) => mir_types::BorrowKind::Shared,
    }
}

fn translate_mutability(m: rustc_hir::Mutability) -> mir_types::Mutability {
    match m {
        rustc_hir::Mutability::Not => mir_types::Mutability::Not,
        rustc_hir::Mutability::Mut => mir_types::Mutability::Mut,
    }
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub fn translate_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: ty::Ty<'tcx>) -> mir_types::Ty {
    ensure_stack(|| translate_ty_inner(tcx, ty))
}

fn translate_ty_inner<'tcx>(tcx: TyCtxt<'tcx>, ty: ty::Ty<'tcx>) -> mir_types::Ty {
    match ty.kind() {
        ty::TyKind::Bool => mir_types::Ty::Bool,
        ty::TyKind::Char => mir_types::Ty::Char,
        ty::TyKind::Str => mir_types::Ty::Str,
        ty::TyKind::Never => mir_types::Ty::Never,
        ty::TyKind::Int(int_ty) => mir_types::Ty::Int(translate_int_ty(*int_ty)),
        ty::TyKind::Uint(uint_ty) => mir_types::Ty::Uint(translate_uint_ty(*uint_ty)),
        ty::TyKind::Float(float_ty) => mir_types::Ty::Float(translate_float_ty(*float_ty)),
        ty::TyKind::Tuple(tys) => {
            mir_types::Ty::Tuple(tys.iter().map(|t| translate_ty(tcx, t)).collect())
        }
        ty::TyKind::Array(elem, len) => {
            let count = len.try_to_target_usize(tcx).unwrap_or(0);
            mir_types::Ty::Array(Box::new(translate_ty(tcx, *elem)), count)
        }
        ty::TyKind::Slice(elem) => {
            mir_types::Ty::Slice(Box::new(translate_ty(tcx, *elem)))
        }
        ty::TyKind::Ref(_, inner, mutbl) => {
            mir_types::Ty::Ref(
                translate_mutability(*mutbl),
                Box::new(translate_ty(tcx, *inner)),
            )
        }
        ty::TyKind::RawPtr(inner, mutbl) => {
            mir_types::Ty::RawPtr(
                translate_mutability(*mutbl),
                Box::new(translate_ty(tcx, *inner)),
            )
        }
        ty::TyKind::Adt(adt_def, args) => {
            let def_id = adt_def.did();
            let hash = def_path_hash(tcx, def_id);
            let name = tcx.def_path_str(def_id);
            let generic_args = translate_generic_args(tcx, args);
            mir_types::Ty::Adt(hash, name, generic_args)
        }
        ty::TyKind::FnDef(def_id, args) => {
            mir_types::Ty::FnDef(
                def_path_hash(tcx, *def_id),
                translate_generic_args(tcx, args),
            )
        }
        ty::TyKind::FnPtr(sig_tys, _header) => {
            let sig = sig_tys.skip_binder();
            let params: Vec<_> = sig.inputs().iter().map(|t| translate_ty(tcx, *t)).collect();
            let ret = Box::new(translate_ty(tcx, sig.output()));
            mir_types::Ty::FnPtr(params, ret)
        }
        ty::TyKind::Param(param_ty) => {
            mir_types::Ty::Param(param_ty.index, param_ty.name.to_string())
        }
        ty::TyKind::Closure(def_id, args) => {
            mir_types::Ty::Closure(
                def_path_hash(tcx, *def_id),
                translate_generic_args(tcx, args),
            )
        }
        ty::TyKind::Dynamic(predicates, _) => {
            let preds: Vec<_> = predicates
                .iter()
                .filter_map(|pred| {
                    match pred.skip_binder() {
                        ty::ExistentialPredicate::Trait(trait_ref) => {
                            let hash = def_path_hash(tcx, trait_ref.def_id);
                            let args = translate_generic_args(tcx, trait_ref.args);
                            Some(mir_types::ExistentialPredicate {
                                trait_ref: Some((hash, args)),
                            })
                        }
                        _ => None,
                    }
                })
                .collect();
            mir_types::Ty::Dynamic(preds)
        }
        ty::TyKind::Foreign(def_id) => {
            mir_types::Ty::Foreign(def_path_hash(tcx, *def_id))
        }
        ty::TyKind::Alias(_, _) => {
            mir_types::Ty::Opaque(format!("{ty:?}"))
        }
        _ => mir_types::Ty::Opaque(format!("{ty:?}")),
    }
}

fn translate_int_ty(ty: ty::IntTy) -> mir_types::IntTy {
    match ty {
        ty::IntTy::Isize => mir_types::IntTy::Isize,
        ty::IntTy::I8 => mir_types::IntTy::I8,
        ty::IntTy::I16 => mir_types::IntTy::I16,
        ty::IntTy::I32 => mir_types::IntTy::I32,
        ty::IntTy::I64 => mir_types::IntTy::I64,
        ty::IntTy::I128 => mir_types::IntTy::I128,
    }
}

fn translate_uint_ty(ty: ty::UintTy) -> mir_types::UintTy {
    match ty {
        ty::UintTy::Usize => mir_types::UintTy::Usize,
        ty::UintTy::U8 => mir_types::UintTy::U8,
        ty::UintTy::U16 => mir_types::UintTy::U16,
        ty::UintTy::U32 => mir_types::UintTy::U32,
        ty::UintTy::U64 => mir_types::UintTy::U64,
        ty::UintTy::U128 => mir_types::UintTy::U128,
    }
}

fn translate_float_ty(ty: ty::FloatTy) -> mir_types::FloatTy {
    match ty {
        ty::FloatTy::F16 => mir_types::FloatTy::F16,
        ty::FloatTy::F32 => mir_types::FloatTy::F32,
        ty::FloatTy::F64 => mir_types::FloatTy::F64,
        ty::FloatTy::F128 => mir_types::FloatTy::F128,
    }
}

// ---------------------------------------------------------------------------
// Generic args
// ---------------------------------------------------------------------------

fn translate_generic_args<'tcx>(
    tcx: TyCtxt<'tcx>,
    args: &[ty::GenericArg<'tcx>],
) -> Vec<mir_types::GenericArg> {
    args.iter()
        .map(|arg| translate_generic_arg(tcx, *arg))
        .collect()
}

fn translate_generic_arg<'tcx>(
    tcx: TyCtxt<'tcx>,
    arg: ty::GenericArg<'tcx>,
) -> mir_types::GenericArg {
    match arg.kind() {
        ty::GenericArgKind::Type(ty) => {
            mir_types::GenericArg::Ty(translate_ty(tcx, ty))
        }
        ty::GenericArgKind::Const(ct) => {
            // Try to extract a scalar leaf from the const
            if let Some(scalar_int) = ct.try_to_leaf() {
                let data = scalar_int.to_bits(scalar_int.size());
                let size = scalar_int.size().bytes() as u8;
                mir_types::GenericArg::Const(mir_types::ConstOperand {
                    ty: mir_types::Ty::Opaque("const_arg".to_string()),
                    kind: mir_types::ConstKind::Scalar(data, size),
                })
            } else {
                mir_types::GenericArg::Const(mir_types::ConstOperand {
                    ty: mir_types::Ty::Opaque("const_arg".to_string()),
                    kind: mir_types::ConstKind::Todo(format!("{ct:?}")),
                })
            }
        }
        ty::GenericArgKind::Lifetime(_) => mir_types::GenericArg::Lifetime,
    }
}

// ---------------------------------------------------------------------------
// Layouts
// ---------------------------------------------------------------------------

pub fn translate_layout_data(
    layout: &rustc_abi::LayoutData<rustc_abi::FieldIdx, rustc_abi::VariantIdx>,
) -> mir_types::LayoutInfo {
    mir_types::LayoutInfo {
        size: layout.size.bytes(),
        align: layout.align.abi.bytes(),
        backend_repr: translate_backend_repr(&layout.backend_repr),
        fields: translate_fields_shape(&layout.fields),
        variants: translate_variants(&layout.variants),
        largest_niche: layout.largest_niche.as_ref().map(translate_niche),
    }
}

fn translate_backend_repr(repr: &rustc_abi::BackendRepr) -> mir_types::ExportedBackendRepr {
    match repr {
        rustc_abi::BackendRepr::Scalar(s) => {
            mir_types::ExportedBackendRepr::Scalar(translate_scalar(s))
        }
        rustc_abi::BackendRepr::ScalarPair(a, b) => {
            mir_types::ExportedBackendRepr::ScalarPair(translate_scalar(a), translate_scalar(b))
        }
        rustc_abi::BackendRepr::Memory { sized } => {
            mir_types::ExportedBackendRepr::Memory { sized: *sized }
        }
        rustc_abi::BackendRepr::SimdVector { .. }
        | rustc_abi::BackendRepr::ScalableVector { .. } => {
            mir_types::ExportedBackendRepr::Memory { sized: true }
        }
    }
}

fn translate_scalar(s: &rustc_abi::Scalar) -> mir_types::ExportedScalar {
    match s {
        rustc_abi::Scalar::Initialized { value, valid_range } => mir_types::ExportedScalar {
            primitive: translate_primitive(value),
            valid_range_start: valid_range.start,
            valid_range_end: valid_range.end,
        },
        rustc_abi::Scalar::Union { value } => mir_types::ExportedScalar {
            primitive: translate_primitive(value),
            valid_range_start: 0,
            valid_range_end: u128::MAX,
        },
    }
}

fn translate_primitive(p: &rustc_abi::Primitive) -> mir_types::ExportedPrimitive {
    match p {
        rustc_abi::Primitive::Int(int, signed) => {
            let size_bytes = match int {
                rustc_abi::Integer::I8 => 1,
                rustc_abi::Integer::I16 => 2,
                rustc_abi::Integer::I32 => 4,
                rustc_abi::Integer::I64 => 8,
                rustc_abi::Integer::I128 => 16,
            };
            mir_types::ExportedPrimitive::Int { size_bytes, signed: *signed }
        }
        rustc_abi::Primitive::Float(float) => {
            let size_bytes = match float {
                rustc_abi::Float::F16 => 2,
                rustc_abi::Float::F32 => 4,
                rustc_abi::Float::F64 => 8,
                rustc_abi::Float::F128 => 16,
            };
            mir_types::ExportedPrimitive::Float { size_bytes }
        }
        rustc_abi::Primitive::Pointer(_) => mir_types::ExportedPrimitive::Pointer,
    }
}

fn translate_fields_shape(
    fields: &rustc_abi::FieldsShape<rustc_abi::FieldIdx>,
) -> mir_types::ExportedFieldsShape {
    match fields {
        rustc_abi::FieldsShape::Primitive => mir_types::ExportedFieldsShape::Primitive,
        rustc_abi::FieldsShape::Union(count) => mir_types::ExportedFieldsShape::Union(count.get()),
        rustc_abi::FieldsShape::Array { stride, count } => {
            mir_types::ExportedFieldsShape::Array {
                stride: stride.bytes(),
                count: *count,
            }
        }
        rustc_abi::FieldsShape::Arbitrary { offsets, .. } => {
            mir_types::ExportedFieldsShape::Arbitrary {
                offsets: offsets.iter().map(|o| o.bytes()).collect(),
            }
        }
    }
}

fn translate_variants(
    variants: &rustc_abi::Variants<rustc_abi::FieldIdx, rustc_abi::VariantIdx>,
) -> mir_types::ExportedVariants {
    match variants {
        rustc_abi::Variants::Empty => mir_types::ExportedVariants::Empty,
        rustc_abi::Variants::Single { index } => {
            mir_types::ExportedVariants::Single { index: index.as_u32() }
        }
        rustc_abi::Variants::Multiple { tag, tag_encoding, tag_field, variants } => {
            mir_types::ExportedVariants::Multiple {
                tag: translate_scalar(tag),
                tag_encoding: translate_tag_encoding(tag_encoding),
                tag_field: tag_field.as_u32(),
                variants: variants.iter().map(|v| translate_layout_data(v)).collect(),
            }
        }
    }
}

fn translate_tag_encoding(
    enc: &rustc_abi::TagEncoding<rustc_abi::VariantIdx>,
) -> mir_types::ExportedTagEncoding {
    match enc {
        rustc_abi::TagEncoding::Direct => mir_types::ExportedTagEncoding::Direct,
        rustc_abi::TagEncoding::Niche { untagged_variant, niche_variants, niche_start } => {
            mir_types::ExportedTagEncoding::Niche {
                untagged_variant: untagged_variant.as_u32(),
                niche_variants_start: niche_variants.start().as_u32(),
                niche_variants_end: niche_variants.end().as_u32(),
                niche_start: *niche_start,
            }
        }
    }
}

fn translate_niche(niche: &rustc_abi::Niche) -> mir_types::ExportedNiche {
    mir_types::ExportedNiche {
        offset: niche.offset.bytes(),
        scalar: mir_types::ExportedScalar {
            primitive: translate_primitive(&niche.value),
            valid_range_start: niche.valid_range.start,
            valid_range_end: niche.valid_range.end,
        },
    }
}

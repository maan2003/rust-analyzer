//! Serializable MIR type definitions for cross-crate MIR export.
//!
//! These types mirror rustc's MIR closely but are self-contained and
//! serializable via serde/postcard. They are produced by `ra-mir-export`
//! (rustc driver) and consumed by `cg-clif` (r-a codegen).

use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// Stable identity for a DefId, using (StableCrateId, local_def_path_hash).
pub type DefPathHash = (u64, u64);

// ---------------------------------------------------------------------------
// Top-level container
// ---------------------------------------------------------------------------

/// Top-level .mirdata payload.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MirData {
    pub crates: Vec<CrateInfo>,
    pub bodies: Vec<FnBody>,
    pub layouts: Vec<TypeLayoutEntry>,
    #[serde(default)]
    pub generic_fn_lookup: Vec<GenericFnLookupEntry>,
}

/// Crate name + StableCrateId.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrateInfo {
    pub name: String,
    pub stable_crate_id: u64,
}

/// A single exported function body.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FnBody {
    /// Stable identity: (StableCrateId, local_def_path_hash)
    pub def_path_hash: DefPathHash,
    /// Human-readable path, e.g. "alloc::vec::Vec::<T>::push"
    pub name: String,
    /// Number of generic type params (0 = non-generic)
    pub num_generic_params: usize,
    pub body: Body,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GenericFnLookupKey {
    pub stable_crate_id: u64,
    pub normalized_path: String,
    pub num_generic_params: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct GenericFnLookupEntry {
    pub key: GenericFnLookupKey,
    pub def_path_hash: DefPathHash,
}

pub fn normalize_def_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut depth = 0u32;

    for ch in path.chars() {
        match ch {
            '<' => {
                if depth == 0 && out.ends_with("::") {
                    out.pop();
                    out.pop();
                }
                depth += 1;
            }
            '>' => {
                depth = depth.saturating_sub(1);
            }
            _ if depth == 0 => {
                if !ch.is_whitespace() {
                    out.push(ch);
                }
            }
            _ => {}
        }
    }

    out
}

// ---------------------------------------------------------------------------
// MIR body structure
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Body {
    /// Locals: index 0 = return place, then args, then temporaries.
    pub locals: Vec<Local>,
    pub arg_count: u32,
    pub blocks: Vec<BasicBlock>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Local {
    pub ty: Ty,
    pub layout: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BasicBlock {
    pub stmts: Vec<Statement>,
    pub terminator: Terminator,
    pub is_cleanup: bool,
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Statement {
    Assign(Place, Rvalue),
    SetDiscriminant { place: Place, variant_index: u32 },
    StorageLive(u32),
    StorageDead(u32),
    Deinit(Place),
    Nop,
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Terminator {
    Goto(u32),
    SwitchInt { discr: Operand, targets: SwitchTargets },
    Return,
    Unreachable,
    UnwindResume,
    Drop { place: Place, target: u32, unwind: UnwindAction },
    Call {
        func: Operand,
        args: Vec<Operand>,
        dest: Place,
        target: Option<u32>,
        unwind: UnwindAction,
    },
    Assert {
        cond: Operand,
        expected: bool,
        target: u32,
        unwind: UnwindAction,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UnwindAction {
    Continue,
    Unreachable,
    Terminate,
    Cleanup(u32),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SwitchTargets {
    pub values: Vec<u128>,
    /// Length = values.len() + 1; last element is the "otherwise" target.
    pub targets: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Operands, places, rvalues
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Operand {
    Copy(Place),
    Move(Place),
    Constant(ConstOperand),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstOperand {
    pub ty: Ty,
    pub kind: ConstKind,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstKind {
    /// Evaluated scalar: raw bits + size in bytes.
    Scalar(u128, u8),
    /// Zero-sized value.
    ZeroSized,
    /// Slice constant (string literal etc): bytes + meta.
    Slice(Vec<u8>, u64),
    /// Unevaluated const (DefPathHash + generic args).
    Unevaluated(DefPathHash, Vec<GenericArg>),
    /// Fallback: human-readable debug string.
    Todo(String),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Place {
    pub local: u32,
    pub projections: Vec<Projection>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Projection {
    Deref,
    Field(u32, Ty),
    Index(u32),
    ConstantIndex { offset: u64, min_length: u64, from_end: bool },
    Subslice { from: u64, to: u64, from_end: bool },
    Downcast(u32),
    OpaqueCast(Ty),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Rvalue {
    Use(Operand),
    Repeat(Operand, u64),
    Ref(BorrowKind, Place),
    RawPtr(Mutability, Place),
    Cast(CastKind, Operand, Ty),
    BinaryOp(BinOp, Operand, Operand),
    UnaryOp(UnOp, Operand),
    Discriminant(Place),
    Aggregate(AggregateKind, Vec<Operand>),
    CopyForDeref(Place),
    ThreadLocalRef(DefPathHash),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum AggregateKind {
    Array(Ty),
    Tuple,
    Adt(DefPathHash, u32, Vec<GenericArg>),
    Closure(DefPathHash, Vec<GenericArg>),
    RawPtr(Ty, Mutability),
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitXor, BitAnd, BitOr, Shl, Shr,
    Eq, Lt, Le, Ne, Ge, Gt, Cmp, Offset,
    AddWithOverflow, SubWithOverflow, MulWithOverflow,
    AddUnchecked, SubUnchecked, MulUnchecked,
    ShlUnchecked, ShrUnchecked,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum UnOp {
    Not,
    Neg,
    PtrMetadata,
}

impl BinOp {
    pub fn run_compare<T: PartialEq + PartialOrd>(&self, l: T, r: T) -> bool {
        match self {
            BinOp::Ge => l >= r,
            BinOp::Gt => l > r,
            BinOp::Le => l <= r,
            BinOp::Lt => l < r,
            BinOp::Eq => l == r,
            BinOp::Ne => l != r,
            x => panic!("`run_compare` called on operator {x:?}"),
        }
    }

    /// Convert an overflowing variant to its wrapping equivalent.
    pub fn overflowing_to_wrapping(self) -> Option<Self> {
        Some(match self {
            BinOp::AddWithOverflow => BinOp::Add,
            BinOp::SubWithOverflow => BinOp::Sub,
            BinOp::MulWithOverflow => BinOp::Mul,
            _ => return None,
        })
    }

    /// Convert a wrapping variant to its overflowing equivalent.
    pub fn wrapping_to_overflowing(self) -> Option<Self> {
        Some(match self {
            BinOp::Add => BinOp::AddWithOverflow,
            BinOp::Sub => BinOp::SubWithOverflow,
            BinOp::Mul => BinOp::MulWithOverflow,
            _ => return None,
        })
    }
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Rem => "%",
            BinOp::BitXor => "^",
            BinOp::BitAnd => "&",
            BinOp::BitOr => "|",
            BinOp::Shl => "<<",
            BinOp::Shr => ">>",
            BinOp::Eq => "==",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Ne => "!=",
            BinOp::Ge => ">=",
            BinOp::Gt => ">",
            BinOp::Offset => "`offset`",
            BinOp::AddWithOverflow => "`add_with_overflow`",
            BinOp::SubWithOverflow => "`sub_with_overflow`",
            BinOp::MulWithOverflow => "`mul_with_overflow`",
            BinOp::AddUnchecked => "`add_unchecked`",
            BinOp::SubUnchecked => "`sub_unchecked`",
            BinOp::MulUnchecked => "`mul_unchecked`",
            BinOp::ShlUnchecked => "`shl_unchecked`",
            BinOp::ShrUnchecked => "`shr_unchecked`",
            BinOp::Cmp => "`cmp`",
        })
    }
}

// ---------------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum CastKind {
    IntToInt,
    FloatToInt,
    IntToFloat,
    FloatToFloat,
    PtrToPtr,
    FnPtrToPtr,
    PointerExposeProvenance,
    PointerWithExposedProvenance,
    PointerCoercion(PointerCoercion),
    Transmute,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PointerCoercion {
    ReifyFnPointer,
    ClosureFnPointer,
    MutToConstPointer,
    ArrayToPointer,
    Unsize,
    UnsafeFnPointer,
}

// ---------------------------------------------------------------------------
// Misc enums
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum BorrowKind {
    Shared,
    Mut,
    Shallow,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mutability {
    Not,
    Mut,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum Ty {
    Bool,
    Char,
    Str,
    Never,
    Int(IntTy),
    Uint(UintTy),
    Float(FloatTy),
    Tuple(Vec<Ty>),
    Array(Box<Ty>, u64),
    Slice(Box<Ty>),
    Ref(Mutability, Box<Ty>),
    RawPtr(Mutability, Box<Ty>),
    Adt(DefPathHash, String, Vec<GenericArg>),
    FnDef(DefPathHash, Vec<GenericArg>),
    FnPtr(Vec<Ty>, Box<Ty>),
    Param(u32, String),
    Closure(DefPathHash, Vec<GenericArg>),
    Dynamic(Vec<ExistentialPredicate>),
    Foreign(DefPathHash),
    /// Fallback for types we don't handle yet.
    Opaque(String),
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntTy {
    Isize, I8, I16, I32, I64, I128,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum UintTy {
    Usize, U8, U16, U32, U64, U128,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum FloatTy {
    F16, F32, F64, F128,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub enum GenericArg {
    Ty(Ty),
    Const(ConstOperand),
    /// Erased lifetime, just a placeholder.
    Lifetime,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExistentialPredicate {
    pub trait_ref: Option<(DefPathHash, Vec<GenericArg>)>,
}

// ---------------------------------------------------------------------------
// Type substitution (monomorphization)
// ---------------------------------------------------------------------------

impl Ty {
    /// Substitute `Param(idx, _)` with the concrete type from `substs[idx]`.
    /// Returns the type unchanged if it contains no Param references.
    pub fn subst(&self, substs: &[GenericArg]) -> Ty {
        match self {
            // Leaf types — no substitution needed
            Ty::Bool | Ty::Char | Ty::Str | Ty::Never
            | Ty::Int(_) | Ty::Uint(_) | Ty::Float(_)
            | Ty::Foreign(_) | Ty::Opaque(_) => self.clone(),

            // The key case: substitute type parameters
            Ty::Param(idx, _name) => {
                match substs.get(*idx as usize) {
                    Some(GenericArg::Ty(concrete)) => concrete.clone(),
                    _ => self.clone(), // keep as-is if no substitution available
                }
            }

            // Recursive cases
            Ty::Tuple(tys) => Ty::Tuple(tys.iter().map(|t| t.subst(substs)).collect()),
            Ty::Array(elem, len) => Ty::Array(Box::new(elem.subst(substs)), *len),
            Ty::Slice(elem) => Ty::Slice(Box::new(elem.subst(substs))),
            Ty::Ref(m, pointee) => Ty::Ref(*m, Box::new(pointee.subst(substs))),
            Ty::RawPtr(m, pointee) => Ty::RawPtr(*m, Box::new(pointee.subst(substs))),
            Ty::Adt(hash, name, args) => {
                Ty::Adt(*hash, name.clone(), subst_generic_args(args, substs))
            }
            Ty::FnDef(hash, args) => {
                Ty::FnDef(*hash, subst_generic_args(args, substs))
            }
            Ty::FnPtr(params, ret) => Ty::FnPtr(
                params.iter().map(|t| t.subst(substs)).collect(),
                Box::new(ret.subst(substs)),
            ),
            Ty::Closure(hash, args) => {
                Ty::Closure(*hash, subst_generic_args(args, substs))
            }
            Ty::Dynamic(preds) => Ty::Dynamic(
                preds.iter().map(|p| ExistentialPredicate {
                    trait_ref: p.trait_ref.as_ref().map(|(hash, args)| {
                        (*hash, subst_generic_args(args, substs))
                    }),
                }).collect(),
            ),
        }
    }

    /// Returns true if this type contains any `Param` references.
    pub fn has_param(&self) -> bool {
        match self {
            Ty::Param(_, _) => true,
            Ty::Bool | Ty::Char | Ty::Str | Ty::Never
            | Ty::Int(_) | Ty::Uint(_) | Ty::Float(_)
            | Ty::Foreign(_) | Ty::Opaque(_) => false,
            Ty::Tuple(tys) => tys.iter().any(|t| t.has_param()),
            Ty::Array(elem, _) | Ty::Slice(elem)
            | Ty::Ref(_, elem) | Ty::RawPtr(_, elem) => elem.has_param(),
            Ty::Adt(_, _, args) | Ty::FnDef(_, args) | Ty::Closure(_, args) => {
                args.iter().any(|a| matches!(a, GenericArg::Ty(t) if t.has_param()))
            }
            Ty::FnPtr(params, ret) => {
                params.iter().any(|t| t.has_param()) || ret.has_param()
            }
            Ty::Dynamic(preds) => preds.iter().any(|p| {
                p.trait_ref.as_ref().is_some_and(|(_, args)| {
                    args.iter().any(|a| matches!(a, GenericArg::Ty(t) if t.has_param()))
                })
            }),
        }
    }
}

/// Substitute generic args, recursing into Ty args.
fn subst_generic_args(args: &[GenericArg], substs: &[GenericArg]) -> Vec<GenericArg> {
    args.iter().map(|arg| match arg {
        GenericArg::Ty(ty) => GenericArg::Ty(ty.subst(substs)),
        other => other.clone(),
    }).collect()
}

impl Body {
    /// Create a monomorphized copy of this body by substituting all `Ty::Param`
    /// references with concrete types from `substs`.
    pub fn subst(&self, substs: &[GenericArg]) -> Body {
        Body {
            locals: self.locals.iter().map(|l| Local {
                ty: l.ty.subst(substs),
                layout: l.layout, // layout indices are preserved; caller must remap
            }).collect(),
            arg_count: self.arg_count,
            blocks: self.blocks.iter().map(|bb| BasicBlock {
                stmts: bb.stmts.iter().map(|s| s.subst(substs)).collect(),
                terminator: bb.terminator.subst(substs),
                is_cleanup: bb.is_cleanup,
            }).collect(),
        }
    }
}

impl Statement {
    fn subst(&self, substs: &[GenericArg]) -> Statement {
        match self {
            Statement::Assign(place, rvalue) => {
                Statement::Assign(place.subst(substs), rvalue.subst(substs))
            }
            Statement::SetDiscriminant { place, variant_index } => {
                Statement::SetDiscriminant { place: place.subst(substs), variant_index: *variant_index }
            }
            Statement::Deinit(place) => Statement::Deinit(place.subst(substs)),
            Statement::StorageLive(l) => Statement::StorageLive(*l),
            Statement::StorageDead(l) => Statement::StorageDead(*l),
            Statement::Nop => Statement::Nop,
        }
    }
}

impl Terminator {
    fn subst(&self, substs: &[GenericArg]) -> Terminator {
        match self {
            Terminator::Call { func, args, dest, target, unwind } => Terminator::Call {
                func: func.subst(substs),
                args: args.iter().map(|a| a.subst(substs)).collect(),
                dest: dest.subst(substs),
                target: *target,
                unwind: unwind.clone(),
            },
            Terminator::SwitchInt { discr, targets } => Terminator::SwitchInt {
                discr: discr.subst(substs),
                targets: targets.clone(),
            },
            Terminator::Drop { place, target, unwind } => Terminator::Drop {
                place: place.subst(substs),
                target: *target,
                unwind: unwind.clone(),
            },
            Terminator::Assert { cond, expected, target, unwind } => Terminator::Assert {
                cond: cond.subst(substs),
                expected: *expected,
                target: *target,
                unwind: unwind.clone(),
            },
            Terminator::Goto(t) => Terminator::Goto(*t),
            Terminator::Return => Terminator::Return,
            Terminator::Unreachable => Terminator::Unreachable,
            Terminator::UnwindResume => Terminator::UnwindResume,
        }
    }
}

impl Operand {
    fn subst(&self, substs: &[GenericArg]) -> Operand {
        match self {
            Operand::Copy(p) => Operand::Copy(p.subst(substs)),
            Operand::Move(p) => Operand::Move(p.subst(substs)),
            Operand::Constant(c) => Operand::Constant(ConstOperand {
                ty: c.ty.subst(substs),
                kind: c.kind.subst(substs),
            }),
        }
    }
}

impl ConstKind {
    fn subst(&self, substs: &[GenericArg]) -> ConstKind {
        match self {
            ConstKind::Unevaluated(hash, args) => {
                ConstKind::Unevaluated(*hash, subst_generic_args(args, substs))
            }
            other => other.clone(),
        }
    }
}

impl Place {
    fn subst(&self, substs: &[GenericArg]) -> Place {
        Place {
            local: self.local,
            projections: self.projections.iter().map(|p| p.subst(substs)).collect(),
        }
    }
}

impl Projection {
    fn subst(&self, substs: &[GenericArg]) -> Projection {
        match self {
            Projection::Field(idx, ty) => Projection::Field(*idx, ty.subst(substs)),
            Projection::OpaqueCast(ty) => Projection::OpaqueCast(ty.subst(substs)),
            other => other.clone(),
        }
    }
}

impl Rvalue {
    fn subst(&self, substs: &[GenericArg]) -> Rvalue {
        match self {
            Rvalue::Use(op) => Rvalue::Use(op.subst(substs)),
            Rvalue::Repeat(op, count) => Rvalue::Repeat(op.subst(substs), *count),
            Rvalue::Ref(bk, place) => Rvalue::Ref(bk.clone(), place.subst(substs)),
            Rvalue::RawPtr(m, place) => Rvalue::RawPtr(*m, place.subst(substs)),
            Rvalue::Cast(kind, op, ty) => {
                Rvalue::Cast(kind.clone(), op.subst(substs), ty.subst(substs))
            }
            Rvalue::BinaryOp(op, lhs, rhs) => {
                Rvalue::BinaryOp(*op, lhs.subst(substs), rhs.subst(substs))
            }
            Rvalue::UnaryOp(op, operand) => Rvalue::UnaryOp(*op, operand.subst(substs)),
            Rvalue::Discriminant(place) => Rvalue::Discriminant(place.subst(substs)),
            Rvalue::Aggregate(kind, ops) => {
                let kind = match kind {
                    AggregateKind::Array(ty) => AggregateKind::Array(ty.subst(substs)),
                    AggregateKind::Adt(hash, variant, args) => {
                        AggregateKind::Adt(*hash, *variant, subst_generic_args(args, substs))
                    }
                    AggregateKind::Closure(hash, args) => {
                        AggregateKind::Closure(*hash, subst_generic_args(args, substs))
                    }
                    AggregateKind::RawPtr(ty, m) => AggregateKind::RawPtr(ty.subst(substs), *m),
                    AggregateKind::Tuple => AggregateKind::Tuple,
                };
                Rvalue::Aggregate(kind, ops.iter().map(|o| o.subst(substs)).collect())
            }
            Rvalue::CopyForDeref(place) => Rvalue::CopyForDeref(place.subst(substs)),
            Rvalue::ThreadLocalRef(hash) => Rvalue::ThreadLocalRef(*hash),
        }
    }
}

// ---------------------------------------------------------------------------
// Layout types
// ---------------------------------------------------------------------------

/// A unique type with its layout, keyed by index into `MirData.layouts`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TypeLayoutEntry {
    pub ty: Ty,
    pub layout: LayoutInfo,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayoutInfo {
    pub size: u64,
    pub align: u64,
    pub backend_repr: ExportedBackendRepr,
    pub fields: ExportedFieldsShape,
    pub variants: ExportedVariants,
    pub largest_niche: Option<ExportedNiche>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExportedBackendRepr {
    Scalar(ExportedScalar),
    ScalarPair(ExportedScalar, ExportedScalar),
    Memory { sized: bool },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExportedScalar {
    pub primitive: ExportedPrimitive,
    pub valid_range_start: u128,
    pub valid_range_end: u128,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExportedPrimitive {
    Int { size_bytes: u8, signed: bool },
    Float { size_bytes: u8 },
    Pointer,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExportedFieldsShape {
    Primitive,
    Union(usize),
    Array { stride: u64, count: u64 },
    Arbitrary { offsets: Vec<u64> },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExportedVariants {
    Empty,
    Single { index: u32 },
    Multiple {
        tag: ExportedScalar,
        tag_encoding: ExportedTagEncoding,
        tag_field: u32,
        variants: Vec<LayoutInfo>,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ExportedTagEncoding {
    Direct,
    Niche {
        untagged_variant: u32,
        niche_variants_start: u32,
        niche_variants_end: u32,
        niche_start: u128,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExportedNiche {
    pub offset: u64,
    pub scalar: ExportedScalar,
}

//! Serializable MIR type definitions for cross-crate MIR export.
//!
//! These types mirror rustc's MIR closely but are self-contained and
//! serializable via serde/postcard. They are produced by `ra-mir-export`
//! (rustc driver) and consumed by `cg-clif` (r-a codegen).

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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ConstOperand {
    pub ty: Ty,
    pub kind: ConstKind,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div, Rem,
    BitXor, BitAnd, BitOr, Shl, Shr,
    Eq, Lt, Le, Ne, Ge, Gt, Cmp, Offset,
    AddWithOverflow, SubWithOverflow, MulWithOverflow,
    AddUnchecked, SubUnchecked, MulUnchecked,
    ShlUnchecked, ShrUnchecked,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UnOp {
    Not,
    Neg,
    PtrMetadata,
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BorrowKind {
    Shared,
    Mut,
    Shallow,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Mutability {
    Not,
    Mut,
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Debug, Clone)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum IntTy {
    Isize, I8, I16, I32, I64, I128,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UintTy {
    Usize, U8, U16, U32, U64, U128,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum FloatTy {
    F16, F32, F64, F128,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GenericArg {
    Ty(Ty),
    Const(ConstOperand),
    /// Erased lifetime, just a placeholder.
    Lifetime,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExistentialPredicate {
    pub trait_ref: Option<(DefPathHash, Vec<GenericArg>)>,
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

//! Shared serializable types used by `ra-mir-export` and `hir-ty`.

use std::fmt::Display;

use serde::{Deserialize, Serialize};

/// Top-level `.mirdata` payload.
///
/// For now we only export crate metadata required for symbol disambiguation.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MirData {
    pub crates: Vec<CrateInfo>,
}

/// Crate name + `StableCrateId`.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct CrateInfo {
    pub name: String,
    pub stable_crate_id: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    BitXor,
    BitAnd,
    BitOr,
    Shl,
    Shr,
    Eq,
    Lt,
    Le,
    Ne,
    Ge,
    Gt,
    Cmp,
    Offset,
    AddWithOverflow,
    SubWithOverflow,
    MulWithOverflow,
    AddUnchecked,
    SubUnchecked,
    MulUnchecked,
    ShlUnchecked,
    ShrUnchecked,
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

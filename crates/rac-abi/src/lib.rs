//! ABI computation for rac codegen.
//!
//! See `CHANGES.md` for provenance and modifications.
#![allow(dead_code)]

pub(crate) mod layout_ty;
mod abi_callconv;
pub mod callconv;
pub mod spec;

pub use layout_ty::{
    FIRST_VARIANT, FieldIdx, Interned, Layout, TyAbiInterface, TyAndLayout, VariantIdx,
};
pub use abi_callconv::{Heterogeneous, HomogeneousAggregate};

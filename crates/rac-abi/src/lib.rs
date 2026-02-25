//! ABI computation for rac codegen.
//!
//! See `CHANGES.md` for provenance and modifications.
#![allow(dead_code)]

mod abi_callconv;
pub mod callconv;
pub(crate) mod layout_ty;
pub mod spec;

pub use abi_callconv::{Heterogeneous, HomogeneousAggregate};
pub use layout_ty::{
    FIRST_VARIANT, FieldIdx, Interned, Layout, TyAbiInterface, TyAndLayout, VariantIdx,
};

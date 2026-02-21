# Changes from upstream rustc

Source revision: `1c00e989ca032d57e815e930fad00b61e65a1826`

This crate contains code copied from `rustc_abi` (nightly-gated parts) and
`rustc_target::callconv`, adapted to compile against the published
`ra-ap-rustc_abi` (v0.143) crate. The `ra-ap-rustc_abi` crate strips the
`nightly` feature, so types like `TyAbiInterface`, `TyAndLayout`, `Layout`,
`FieldIdx`, `VariantIdx`, `Reg`, `RegKind`, and the `callconv` module are not
available from it.

## File origins

| Local path | Upstream path |
|---|---|
| `src/layout_ty.rs` | `rustc_abi/src/layout/ty.rs` |
| `src/abi_callconv.rs` | `rustc_abi/src/callconv.rs` |
| `src/callconv/reg.rs` | `rustc_abi/src/callconv/reg.rs` |
| `src/callconv/mod.rs` | `rustc_target/src/callconv/mod.rs` |
| `src/callconv/<arch>.rs` (25 files) | `rustc_target/src/callconv/<arch>.rs` |
| `src/spec.rs` | *New file* (not from rustc) |

## Changes applied

### All files

- **Import path adjustments**: `TyAbiInterface` and `TyAndLayout` are imported
  from `crate::layout_ty` instead of `rustc_abi`. `Reg` and `RegKind` are
  imported from `crate::callconv` (re-exported from `callconv::reg`) instead of
  `rustc_abi`.

### `layout_ty.rs` (from `rustc_abi/src/layout/ty.rs`)

- **`Interned`**: Defined locally as a trivial newtype (`pub struct
  Interned<'a, T>(pub &'a T)`) with manual `Copy`/`Clone` impls (no bound on
  `T`), replacing `rustc_data_structures::intern::Interned`.
- **`FieldIdx` / `VariantIdx`**: Replaced `newtype_index!` macro invocations
  with manual struct definitions + `rustc_index::Idx` trait implementations.
  The `newtype_index!` macro requires nightly-only attributes
  (`#[rustc_layout_scalar_valid_range_end]`, `#[rustc_pass_by_value]`).
- **Removed `HashStable_Generic` derive** from all types (requires
  `rustc_macros` which is not available).
- **Changed `use crate::` to `use rustc_abi::`** since the base types (`Size`,
  `Align`, `BackendRepr`, etc.) live in the `ra-ap-rustc_abi` dependency, not
  in this crate.

### `abi_callconv.rs` (from `rustc_abi/src/callconv.rs`)

- **Removed `#[cfg(feature = "nightly")]` gates**.
- **Removed `mod reg;` and `pub use reg::{Reg, RegKind};`** — `Reg`/`RegKind`
  are now in `callconv/reg.rs` (under `rustc_target`'s callconv module).
- **Removed `BackendRepr::ScalableVector` match arm** from
  `homogeneous_aggregate` (variant does not exist in `ra-ap-rustc_abi` 0.143).

### `callconv/reg.rs` (from `rustc_abi/src/callconv/reg.rs`)

- **Removed `HashStable_Generic` derives** and `rustc_macros` import.
- **Changed `use crate::` to `use rustc_abi::`**.

### `callconv/mod.rs` (from `rustc_target/src/callconv/mod.rs`)

- **Removed `HashStable_Generic` derives** from `PassMode`, `CastTarget`,
  `ArgAbi`, `FnAbi`, etc.
- **Replaced `rustc_data_structures::external_bitflags_debug!`** with a manual
  `Debug` impl using `bitflags::parser::to_writer`.
- **Removed `static_assert_size!`** calls (the `size_asserts` module at the end
  of the file).
- **Removed `BackendRepr::ScalableVector` match arm**.
- **Added `mod reg;`** declaration.

### `callconv/<arch>.rs` (25 architecture files)

- **Import path adjustments** (mechanical, applied to all 25 files):
  `TyAbiInterface`/`TyAndLayout` moved from `rustc_abi::` to
  `crate::layout_ty::`, `Reg`/`RegKind` moved from `rustc_abi::` to
  `crate::callconv::`.
- **Removed `BackendRepr::ScalableVector` match arms** in: `loongarch.rs`,
  `riscv.rs`, `sparc64.rs`, `x86.rs`, `x86_64.rs`, `x86_win64.rs`.

### `spec.rs` (new, not from rustc)

Provides minimal versions of `rustc_target::spec` types needed by the callconv
code: `Arch`, `Abi`, `Env`, `Os`, `RustcAbi`, `Target`, `HasTargetSpec`,
`X86Abi`, `HasX86AbiOpt`.

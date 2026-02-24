# mirdata JIT test harness — status & next steps

## What works

The seamless mirdata JIT harness (`jit_run_with_std`) lets tests write plain Rust:
```rust
fn foo() -> i32 {
    core::convert::identity(42)
}
```

The harness:
1. Generates mirdata by running `cargo run` on `ra-mir-export` (into a tempdir, cleaned up on drop)
2. Loads real sysroot sources (core, alloc, std) into r-a's TestDB
3. Compiles user code via r-a MIR (`collect_reachable_fns`)
4. Cross-crate **free** generic calls → compiled from mirdata with matching v0 names
5. Cross-crate non-generic calls → resolved via dlopen'd libstd.so

Passing tests:
- `mirdata_jit_identity_i32` — `core::convert::identity(42)`
- `mirdata_jit_ptr_write_read_i32` — `core::ptr::write` / `core::ptr::read`
- `mirdata_jit_mem_replace_i32` — `core::mem::replace`

## Blocking issue: mirdata lookup for associated functions

**The mirdata body lookup uses string path matching**, and the paths don't match
for associated (impl) functions.

mirdata names (from `with_no_visible_paths!(tcx.def_path_str(def_id))` in ra-mir-export):
```
alloc::vec::Vec::<T>::new          (1 generic param)
alloc::vec::Vec::<T, A>::push      (2 generic params)
alloc::raw_vec::RawVec::<T, A>::grow_one
```

r-a's `fn_display_path` (walks module hierarchy from FunctionId):
```
alloc::vec::new
alloc::vec::push
alloc::raw_vec::grow_one
```

**The r-a path is missing the type name** (`Vec`, `RawVec`) because `fn_display_path`
walks modules only — it doesn't know about the impl's self type.

### Current test failure
```
mirdata body not found for cross-crate generic: alloc::vec::new
  (mangled: _RINvMNtCs...5alloc3vecINtNtCs...3vec3VecpNtNtCs...5alloc6GlobalE3newlE)
```

Note: the v0-mangled name IS correct (includes `Vec` in the path). Only the
string-based mirdata lookup is broken.

## Fix options

### Option A: Match by `def_path_hash` (robust, recommended)

mirdata already stores `def_path_hash: (u64, u64)` per function body (from
`tcx.def_path_hash(def_id)` → `(stable_crate_id, local_hash)`).

Need: compute the same `DefPathHash` from r-a's FunctionId. This requires:
- The crate's `StableCrateId` (already available from `ext_crate_disambiguators`)
- The local DefPath hash (same hash algorithm as rustc — based on the
  `DefPath` segments: `CrateRoot / TypeNs("vec") / Impl / ValueNs("push")`)

**Complication**: r-a doesn't compute rustc-compatible `DefPathHash` values.
The hash includes the full DefPath with disambiguators, and r-a's internal IDs
don't map 1:1 to rustc's DefPath segments (especially for impl blocks).

### Option B: Improve `fn_display_path` to include the type name

For associated functions in impl blocks, include the self type name:
```
alloc::vec::Vec::new    (instead of alloc::vec::new)
alloc::vec::Vec::push   (instead of alloc::vec::push)
```

And strip generic params from mirdata names for matching:
```
alloc::vec::Vec::<T>::new  →  alloc::vec::Vec::new
```

**Implementation**: In `fn_display_path`, check if the function's container is
an `ImplId`. If so, get the impl's self type and insert the type name between
the module path and function name.

For mirdata matching: strip `<...>` from the name before lookup.

**Risk**: Type names must match exactly (r-a's ADT name vs rustc's). Should be
fine for standard library types but could diverge for complex generics.

### Option C: Match by v0-mangled symbol (without generic substitution)

The v0 mangling already encodes the full path correctly (including impl paths).
Could store a "base symbol" (v0-mangled path without generic instantiation) in
mirdata and compute the same from r-a.

**Complication**: v0 mangling is designed for concrete instantiations. A
"generic base symbol" isn't standard. Would need a custom scheme.

## Key files

- `crates/cg-clif/src/tests.rs` — `jit_run_with_std`, `fn_display_path`, `generate_mirdata`
- `ra-mir-export/src/main.rs` — `try_export_fn`, naming via `with_no_visible_paths!`
- `crates/cg-clif/src/symbol_mangling.rs` — v0 mangling (works correctly for impl fns)
- `crates/cg-clif/src/lib.rs` — `collect_reachable_fns`, `build_fn_sig_from_ty`

## Other notes

- `collect_reachable_fns` returns a 4th value: `Vec<(FunctionId, StoredGenericArgs)>`
  for cross-crate callees. These are functions where `is_cross_crate == true`.
- `build_fn_sig_from_ty` takes `GenericArgs` and uses `EarlyBinder::instantiate()`
  to substitute generic args into the function signature (fixes HasPlaceholder errors).
- `ext_crate_disambiguators` maps crate name → StableCrateId from mirdata, used
  for correct v0 mangling of cross-crate symbols.

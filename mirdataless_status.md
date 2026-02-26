# Mirdataless Harness Status

## Current Status

The `cg-clif` test harness now has a revived **mirdataless** std-JIT path in
`crates/cg-clif/src/tests.rs`.

What is in place:

- `jit_run_with_std` is restored with the new architecture (no mirdata body translation).
- Real sysroot sources (`core`, `alloc`, `std`) are loaded into a `TestDB` for r-a MIR/type queries.
- Reachable local code and cross-crate generic instantiations are compiled using the normal pipeline:
  - `collect_reachable_fns`
  - `compile_fn`
  - `compile_drop_in_place`
- Cross-crate monomorphic calls use a hybrid strategy:
  - compile from MIR when the mangled symbol is not exported from `libstd.so`
    (covers many `#[inline]` std/core helpers such as `core::str::len`)
  - keep as imports when exported in `libstd.so`
  - if cross-crate MIR lowering fails (e.g. unresolved `libc::...` in sys shims),
    fall back to import linkage instead of failing test compilation
- External calls are resolved through existing mangling/link strategy:
  - crate disambiguators from `RA_MIRDATA` metadata (`extract_crate_disambiguators()`)
  - dynamic loading of `libstd.so` (`dlopen(..., RTLD_GLOBAL)`)
- `core::intrinsics::ptr_metadata` is now lowered in codegen intrinsic handling.
- `.mirdata` remains metadata-only (crate names + stable crate ids); no old body/layout schema was revived.
- rust-analyzer builtin macro coverage now includes `pattern_type!` (erased to base type in fallback expander).
  - this unblocks `core::num::niche_types::UsizeNoHighBit` type resolution in `alloc::raw_vec`/`Vec` layout paths.
  - `std` JIT flow no longer fails at `local layout error: HasErrorType` for this path.
- Codegen constant lowering now resolves `ConstKind::Unevaluated` via `const_eval`/`const_eval_static`
  before extracting bytes (Scalar, ScalarPair, memory constants, array/repeat lengths).
- `codegen_cast` now handles ptr-like wide-pointer casts explicitly:
  - wide (`ScalarPair`) -> wide keeps data+metadata
  - wide (`ScalarPair`) -> thin drops metadata
  - this removed the `load_scalar on ByValPair` panic seen in `Vec::new` paths.
- `core::intrinsics::{add,sub,mul}_with_overflow` now lower directly in intrinsic codegen
  (instead of becoming unresolved external symbols).
- Lang-item `core::ptr::drop_in_place` calls are now lowered to generated drop glue,
  avoiding execution of the intentionally-recursive shim body from core.
- Unsized-place pointer projection handling has been extended in `cg-clif`:
  - `CPlace::to_ptr_unsized()` exists and is used in `codegen_place` for
    `ProjectionElem::Index` and `ProjectionElem::ConstantIndex` on slice/str-like DST places.
  - `ProjectionElem::Subslice` lowering is implemented (array/slice/str cases),
    carrying updated metadata for DST subslices.
- `core::intrinsics::cold_path` is lowered as a no-op intrinsic hint.
- MIR lowering now supports const blocks in the mirdataless path:
  - `Expr::Const` no longer returns `NotSupported("const block")`.
  - the inner const-block expression is lowered under const scope in `crates/hir-ty/src/mir/lower.rs`.
- ADT aggregate ScalarPair lowering now materializes ABI lanes from field representations
  (Scalar / ScalarPair) instead of assuming "2 non-ZST fields".
  - this fixes the `ScalarPair ADT expects 2 non-ZST fields` panic in `codegen_adt_fields`
    and aligns behavior with upstream `cg_clif`'s field-wise aggregate writes.
- `PointerCoercion::Unsize` now handles slice metadata in addition to dyn-trait metadata:
  - array-to-slice / str-like unsizing now builds `(data_ptr, len)` fat pointers.
  - fixes the panic `Unsize target pointee must be dyn Trait` on array->slice deref paths.
- MIR lowering for overloaded unary deref no longer injects an extra borrow before
  `Deref::deref` calls.
  - avoids lowering `*v` (where `v: Vec<T>`) as an effective `deref(&&Vec<T>)` call.
  - fixes incorrect fat-pointer results on `&*v` paths used by `Vec` indexing.

## Tests Revived

- Passing std-JIT smoke tests (current):
  - `std_jit_process_id_nonzero`
  - `std_jit_generic_identity_i32`
  - `std_jit_str_len_smoke`
  - `std_jit_array_index_smoke`
  - `std_jit_option_match_smoke`
  - `std_jit_option_unwrap_smoke`
  - `std_jit_option_expect_smoke`
  - `std_jit_box_new_i32_smoke`
  - `std_jit_vec_new_smoke` (now unignored and passing)
  - `std_jit_vec_push_smoke` (now unignored and passing)
  - `std_jit_vec_growth_sum_smoke`
  - `std_jit_refcell_borrow_mut_smoke`
  - `std_jit_array_deref_to_slice_smoke`
  - `std_jit_env_var_smoke`
  - `std_jit_str_parse_i32_smoke` (now unignored and passing)
- `std_jit_process_id_is_stable_across_calls` is currently non-ignored in `tests.rs`
  (historically flaky; keep watching for intermittent regressions)
- Additional probes present and currently ignored:
  - `std_jit_string_from_smoke`
    - currently aborts at runtime with allocator double-free (`free(): double free detected in tcache 2`)
  - `std_jit_env_set_var_smoke`
    - fails while compiling `std::io::error::repr_bitpacked::Repr::drop`
      with `local layout error: HasErrorType`
  - `std_jit_env_var_roundtrip`
    - fails with `non-value const in ScalarPair constant`
  - `std_jit_mutex_lock_smoke`
    - fails with `GenericArgNotProvided` in `std::sync::poison::mutex::MutexGuard::drop`

Validation recently run:

- `just test-clif -E 'test(std_jit_process_id_nonzero)' --no-capture` passes
- `just test-clif -E 'test(std_jit_str_len_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_box_new_i32_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_new_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_push_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_growth_sum_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_array_deref_to_slice_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_env_var_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_str_parse_i32_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_refcell_borrow_mut_smoke)' --no-capture` passes
- `just test-clif std_jit_string_from_smoke --run-ignored only --no-capture` aborts with
  `free(): double free detected in tcache 2`
- `just test-clif -E 'test(std_jit_env_set_var_smoke)' --no-capture` fails with
  `local layout error: HasErrorType` in `std::io::error::repr_bitpacked::Repr::drop`
- `just test-clif -E 'test(std_jit_mutex_lock_smoke)' --no-capture` fails with
  `GenericArgNotProvided` in `std::sync::poison::mutex::MutexGuard::drop`

## What Is Still Missing

- Coverage improved, but many real std paths are still gated by a few backend gaps.
- `const_eval_select` runtime-arm signature collisions are no longer the blocker for Vec paths.
- `String::from` still hits a runtime ownership/deallocation bug (double free).
- ScalarPair constant materialization is still incomplete for non-value const forms.
  - currently blocks `std_jit_env_var_roundtrip`.
- Some std paths still hit `HasErrorType` layout failures in deeper std internals.
  - currently blocks `std_jit_env_set_var_smoke` (`std::io::error::repr_bitpacked::Repr::drop`).
- Runtime symbol resolution relies on process-global `dlopen` behavior; robustness improvements are possible.

## Recommended Fix Order

1. Fix the `String::from` runtime double-free.
   - Repro target: `std_jit_string_from_smoke`.
   - Focus: ownership/drop paths across std string construction and drop glue.

2. Complete ScalarPair constant lowering for non-value const forms.
   - Repro target: `std_jit_env_var_roundtrip`.
   - Goal: support the const shapes used by `std::env` result/error paths.

3. Address `HasErrorType` layout holes in std IO error paths.
   - Repro target: `std_jit_env_set_var_smoke`.
   - Focus: why `repr_bitpacked::Repr::drop` local layout still unresolved in mirdataless flow.

4. Re-run and unignore probes incrementally as each blocker is fixed.
   - Current state: `str_parse` and `vec_push` are unignored; `string_from` remains blocked.

5. Expand coverage with more deterministic std probes once blockers are cleared.
   - Candidate families: `std::thread::current`, `std::time`, and light `std::sync` probes.

6. Improve sysroot loading ergonomics/perf for tests.
   - Cache file discovery and/or loaded roots across tests when feasible.
   - Keep wall-time reasonable as std-smoke coverage grows.

7. Introduce a focused std-JIT test group in `just test-clif` docs/comments.
   - Make it easy to run only mirdataless std-JIT smoke tests locally.

## Non-Goals (Still True)

- Do not reintroduce mirdata body translation paths (`mirdata_codegen.rs`, old layout bridging, large mirdata schema).
- Do not use `.mirdata` for anything beyond crate disambiguator metadata in this flow.

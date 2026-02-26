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
- `core::intrinsics::size_of_val` now computes DST runtime sizes from pointer metadata
  (slice/str lengths and dyn-trait vtable size slot), instead of using static layout size.
- `core::intrinsics::align_of_val` now shares the same DST-aware runtime size/align helper path.
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
- `Drop::drop` impl monomorphization now strips region generic args before
  symbol/mir queries in drop-call paths.
  - fixes the previous `GenericArgNotProvided` failure in
    `std::sync::poison::mutex::MutexGuard::drop`.
- Vtable impl lookup now uses shared cross-crate search (`find_trait_impl_for_simplified_ty`)
  and includes closure-trait fallback wiring in `get_or_create_vtable`.
  - resolves prior merge-conflict drift around dyn-trait/closure vtable construction.
- `ReifyFnPointer` trait-method resolution now routes static trait calls through
  `lookup_impl_method` in both codegen and reachability scanning.
  - this removed an earlier unresolved-symbol failure (`core::fmt::Debug::fmt` in
    the `env_set_var` probe) and moved the failing edge deeper to
    `core::fmt::num::impl::fmt` with `local layout error: HasErrorConst`.
- Enum discriminant lowering now uses `const_eval_discriminant` for
  `SetDiscriminant` direct-tag writes and `Rvalue::Discriminant` reads.
  - `TagEncoding::Niche` reads now remap decoded variant indices to actual
    discriminant values before producing the MIR discriminant result.
  - this fixes explicit-discriminant enum casts that previously returned
    variant indices.
- Type inference indexing now has a builtin array/slice fast path for
  `usize` indices in `lookup_indexing`.
  - this avoids `{type error}` propagation in cast+index expressions,
    which previously caused monomorphization failures in memory-intrinsic probes.

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
  - `std_jit_cell_set_get_smoke`
  - `std_jit_vec_pop_smoke`
  - `std_jit_vec_with_capacity_smoke`
  - `std_jit_result_unwrap_or_smoke`
  - `std_jit_option_take_smoke`
  - `std_jit_slice_split_at_smoke`
  - `std_jit_cmp_max_min_smoke`
  - `std_jit_array_deref_to_slice_smoke`
  - `std_jit_env_var_smoke`
  - `std_jit_str_parse_i32_smoke` (now unignored and passing)
  - `std_jit_option_map_closure_probe` (newly unignored and passing)
  - `std_jit_string_from_smoke` (passes under `--run-ignored only`; ignore annotation is stale)
  - `std_jit_string_push_str_smoke` (passes under `--run-ignored only`; currently kept as probe)
- Passing non-std frontier probe (current):
  - `jit_size_of_val_slice_unsized_probe` (newly unignored and passing)
  - `jit_size_of_val_dyn_trait_probe` (newly unignored and passing)
  - `jit_explicit_enum_discriminant_probe` (newly unignored and passing)
  - `jit_float_scalar_pair_const_probe` (newly unignored and passing)
  - `jit_copy_nonoverlapping_intrinsic_probe` (newly unignored and passing)
  - `jit_write_bytes_intrinsic_probe` (newly unignored and passing)
- `std_jit_process_id_is_stable_across_calls` is currently non-ignored in `tests.rs`
  (historically flaky; keep watching for intermittent regressions)
- Additional probes present and currently ignored:
  - `std_jit_env_set_var_smoke`
    - currently fails while compiling `core::fmt::num::impl::fmt`
      with `local layout error: HasErrorConst`
  - `std_jit_env_var_roundtrip`
    - currently fails at the same point as `env_set_var`:
      `core::fmt::num::impl::fmt` with `local layout error: HasErrorConst`
  - `std_jit_mutex_lock_smoke`
    - now fails deeper in atomic internals:
      `core::sync::atomic::atomic_compare_exchange` with
      `NotSupported("monomorphization resulted in errors")`
  - `std_jit_mutex_try_lock_smoke`
    - now fails deeper in atomic internals:
      `core::sync::atomic::atomic_compare_exchange` with
      `NotSupported("monomorphization resulted in errors")`
  - `std_jit_once_call_once_smoke`
    - currently fails with `GenericArgNotProvided` on
      `std::sys::sync::once::futex::CompletionGuard::drop`
      (`LifetimeParamId ...`, empty generic args)
  - `std_jit_iter_repeat_take_collect_smoke`
    - last known failure was `no impl found for vtable` (not yet revalidated after
      vtable-lookup merge)
  - `std_jit_refcell_replace_smoke`
    - last known behavior: aborts with `SIGSEGV` (not revalidated in latest batch)

Validation recently run:

- `just test-clif -E 'test(std_jit_process_id_nonzero)' --no-capture` passes
- `just test-clif -E 'test(std_jit_str_len_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_box_new_i32_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_new_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_push_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_growth_sum_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_cell_set_get_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_pop_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_with_capacity_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_result_unwrap_or_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_option_take_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_slice_split_at_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_cmp_max_min_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_array_deref_to_slice_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_env_var_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_str_parse_i32_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_refcell_borrow_mut_smoke)' --no-capture` passes
- `just test-clif std_jit_string_from_smoke --run-ignored only --no-capture` passes
- `just test-clif std_jit_string_push_str_smoke --run-ignored only --no-capture` passes
- `just test-clif -E 'test(std_jit_env_set_var_smoke)' --no-capture` fails with
  `local layout error: HasErrorConst` in `core::fmt::num::impl::fmt`
- `just test-clif -E 'test(std_jit_mutex_lock_smoke)' --no-capture` fails with
  `NotSupported("monomorphization resulted in errors")` in
  `core::sync::atomic::atomic_compare_exchange`
- `just test-clif -j 24 -E 'test(std_jit_cell_set_get_smoke) or ... or test(std_jit_mutex_try_lock_smoke)' --run-ignored all --no-fail-fast`
  runs 12 tests concurrently in ~14s total: 8 passed, 4 failed
  (`std_jit_mutex_try_lock_smoke`, `std_jit_once_call_once_smoke`,
  `std_jit_iter_repeat_take_collect_smoke`, `std_jit_refcell_replace_smoke`)
- `just test-clif std_jit_env_var_roundtrip --run-ignored only --no-capture` fails with
  `local layout error: HasErrorConst` in `core::fmt::num::impl::fmt`
- `just test-clif -j 24 -E 'test(jit_dyn_dispatch) or test(jit_dyn_dispatch_multiple_methods) or test(jit_closure_basic) or test(jit_drop_basic) or test(jit_drop_side_effect) or test(jit_drop_no_drop_impl) or test(jit_drop_field_recursive) or test(jit_drop_generic) or test(std_jit_env_var_smoke)' --no-fail-fast`
  runs 9 targeted tests: all pass
- `just test-clif -j 24 -E 'test(std_jit_env_set_var_smoke) or test(std_jit_env_var_roundtrip) or test(std_jit_mutex_lock_smoke) or test(std_jit_mutex_try_lock_smoke) or test(std_jit_once_call_once_smoke)' --run-ignored all --no-fail-fast`
  runs 5 ignored probes: all fail with current blockers listed above
- `just test-clif -j 24 -E 'test(jit_size_of_val_slice_unsized_probe) or test(jit_size_of_val_dyn_trait_probe) or test(jit_explicit_enum_discriminant_probe) or test(jit_float_scalar_pair_const_probe) or test(jit_copy_nonoverlapping_intrinsic_probe) or test(jit_write_bytes_intrinsic_probe) or test(std_jit_option_map_closure_probe)' --no-fail-fast`
  runs active members of the frontier set: 7 passed
  (`jit_size_of_val_slice_unsized_probe`, `jit_size_of_val_dyn_trait_probe`,
  `jit_explicit_enum_discriminant_probe`, `jit_float_scalar_pair_const_probe`,
  `jit_copy_nonoverlapping_intrinsic_probe`, `jit_write_bytes_intrinsic_probe`,
  `std_jit_option_map_closure_probe`).
- `just test-clif -j 24 -E 'test(jit_size_of_val_slice_unsized_probe) or test(jit_size_of_val_dyn_trait_probe) or test(jit_explicit_enum_discriminant_probe) or test(jit_float_scalar_pair_const_probe) or test(jit_copy_nonoverlapping_intrinsic_probe) or test(jit_write_bytes_intrinsic_probe) or test(std_jit_option_map_closure_probe)' --run-ignored all --no-fail-fast`
  runs full frontier set: 7 run, 7 passed.
- `just test-clif -E 'test(jit_size_of_val_dyn_trait_probe)' --no-fail-fast` passes
- `just test-clif -j 24 -E 'test(jit_size_of_val_slice_unsized_probe)' --run-ignored all --no-fail-fast` passes
- `just test-clif -j 24 -E 'test(jit_size_of_val_slice_unsized_probe)' --no-fail-fast` passes

## What Is Still Missing

- Coverage improved, but many real std paths are still gated by a few backend gaps.
- `const_eval_select` runtime-arm signature collisions are no longer the blocker for Vec paths.
- Primary remaining blockers now are:
  - const/layout holes (`HasErrorConst`) in fmt/env-var paths
    (`std_jit_env_set_var_smoke`, `std_jit_env_var_roundtrip`)
  - monomorphization-with-error consts in atomic compare-exchange paths
    (`std_jit_mutex_lock_smoke`, `std_jit_mutex_try_lock_smoke`)
  - lifetime/generic-arg propagation for some monomorphic drop impl lookups
    (`std_jit_once_call_once_smoke` -> `CompletionGuard::drop`)
- Runtime symbol resolution relies on process-global `dlopen` behavior; robustness improvements are possible.

## Recommended Fix Order

1. Address `HasErrorConst` layout holes in fmt/env-var paths.
   - Primary repro target: `std_jit_env_set_var_smoke`.
   - Secondary repro target: `std_jit_env_var_roundtrip`.
   - Focus: why `core::fmt::num::impl::fmt` locals still carry unresolved consts in
     mirdataless flow.

2. Handle atomic compare-exchange monomorphization failures.
   - Repro targets: `std_jit_mutex_lock_smoke`, `std_jit_mutex_try_lock_smoke`.
   - Focus: unresolved `{const error}` in intrinsic-heavy atomic bodies.

3. Fix monomorphic drop call generic/lifetime propagation for std once internals.
   - Repro target: `std_jit_once_call_once_smoke`.

4. Unignore probes that now pass to keep regressions visible.
   - Candidate unignores: `std_jit_string_from_smoke`, `std_jit_string_push_str_smoke`.

5. Re-run and unignore probes incrementally as each blocker is fixed.
   - Current state: `str_parse` and `vec_push` are unignored; `string_from` and
     `string_push_str` now pass but are still marked ignored.
   - New state: `std_jit_option_map_closure_probe`,
     `jit_float_scalar_pair_const_probe`, `jit_size_of_val_dyn_trait_probe`,
     `jit_explicit_enum_discriminant_probe`, `jit_copy_nonoverlapping_intrinsic_probe`, and
     `jit_write_bytes_intrinsic_probe` are unignored and passing.

6. Expand coverage with more deterministic std probes once blockers are cleared.
   - Candidate families: `std::thread::current`, `std::time`, and light `std::sync` probes.

7. Improve sysroot loading ergonomics/perf for tests.
   - Cache file discovery and/or loaded roots across tests when feasible.
   - Keep wall-time reasonable as std-smoke coverage grows.

8. Introduce a focused std-JIT test group in `just test-clif` docs/comments.
   - Make it easy to run only mirdataless std-JIT smoke tests locally.

## Next Candidate To Debug

- `std_jit_env_set_var_smoke`
  - reason: still failing with `local layout error: HasErrorConst` in
    `core::fmt::num::impl::fmt`, and blocks environment-variable probe expansion.

## Non-Goals (Still True)

- Do not reintroduce mirdata body translation paths (`mirdata_codegen.rs`, old layout bridging, large mirdata schema).
- Do not use `.mirdata` for anything beyond crate disambiguator metadata in this flow.

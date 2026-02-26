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
    the `env_set_var` probe); current env-var probe failures are now in a later
    codegen stage (`Index on non-array/slice type`).
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
- `OperandKind::Static` now lowers in codegen:
  - static operand type discovery now builds `&'static T` from static body inference.
  - codegen now imports extern + external-crate statics and defines local const-eval-backed statics,
    removing the `not yet implemented: static operand` panic path.
- Trait-method impl resolution in `cg-clif` now consumes
  `db.lookup_impl_method`'s `Either` result directly in call lowering and
  reachability scanning.
  - builtin-derive `Clone::clone` pseudo methods now lower as direct `*self`
    copies for `Self: Copy`, without falling back to unresolved trait-item symbols.
  - this fixes Arc allocator clone paths that previously imported an unresolved
    `core::clone::Clone::clone::<alloc::alloc::Global>` symbol.

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
  - `std_jit_atomic_u32_smoke`
  - `std_jit_cmp_max_min_smoke`
  - `std_jit_array_deref_to_slice_smoke`
  - `std_jit_env_var_smoke`
  - `std_jit_str_parse_i32_smoke` (now unignored and passing)
  - `std_jit_refcell_replace_smoke`
  - `std_jit_mutex_lock_smoke` (newly unignored and passing)
  - `std_jit_mutex_try_lock_smoke` (newly unignored and passing)
  - `std_jit_arc_mutex_probe` (newly unignored and passing)
  - `std_jit_once_call_once_smoke` (passes under `--run-ignored all`; ignore annotation is stale)
  - `std_jit_option_map_closure_probe` (newly unignored and passing)
  - `std_jit_string_from_smoke` (passes under `--run-ignored all`; ignore annotation is stale)
  - `std_jit_string_push_str_smoke` (passes under `--run-ignored all`; ignore annotation is stale)
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
  - Passing under `--run-ignored all` (ignore annotation is stale):
    - `std_jit_once_call_once_smoke`
    - `std_jit_string_from_smoke`
    - `std_jit_string_push_str_smoke`
  - Failing under `--run-ignored all`:
    - `std_jit_env_set_var_smoke`
      - fails in codegen with `Index on non-array/slice type`
    - `std_jit_env_var_roundtrip`
      - fails in codegen with `Index on non-array/slice type`
    - `std_jit_iter_repeat_take_collect_smoke`
      - fails with `GenericArgNotProvided` while resolving impl method MIR for vtable
    - `std_jit_format_macro_probe`
      - fails with `NotSupported("monomorphization resulted in errors")` while compiling
        `core::fmt::Arguments::estimated_capacity`
    - `std_jit_vec_sort_probe`
      - fails in codegen with `Index on non-array/slice type`
    - `std_jit_hashmap_insert_get_probe`
      - fails with `NotSupported("monomorphization resulted in errors")`
    - `std_jit_btreemap_range_probe`
      - fails with `NotSupported("monomorphization resulted in errors")` in alloc btree
        node-descent path (`alloc::collections::btree::node::Handle<...>::descend`)

Validation recently run:

- `just test-clif --run-ignored all --no-fail-fast`
  - latest full-suite rerun after sync/static-operand and builtin-derive impl-resolution fixes
  - runs one full `cg-clif` nextest sweep in one invocation (covers all `tests.rs` cases,
    plus `link::tests::find_libstd_so_in_sysroot`)
  - result: 129 tests run, 122 passed, 7 failed
  - failures:
    - `std_jit_hashmap_insert_get_probe`
    - `std_jit_format_macro_probe`
    - `std_jit_iter_repeat_take_collect_smoke`
    - `std_jit_env_set_var_smoke`
    - `std_jit_env_var_roundtrip`
    - `std_jit_btreemap_range_probe`
    - `std_jit_vec_sort_probe`

- `just test-clif -j 24 -E 'test(std_jit_mutex_lock_smoke) or test(std_jit_mutex_try_lock_smoke) or test(std_jit_arc_mutex_probe)' --run-ignored all --no-fail-fast`
  - targeted sync triage after builtin-derive `Clone` lowering fix
  - result: 3 tests run, 3 passed
  - passing: `std_jit_mutex_lock_smoke`, `std_jit_mutex_try_lock_smoke`, `std_jit_arc_mutex_probe`

## What Is Still Missing

- Coverage improved, but many real std paths are still gated by a few backend gaps.
- `const_eval_select` runtime-arm signature collisions are no longer the blocker for Vec paths.
- Primary remaining blockers now are:
  - indexing/codegen gap (`Index on non-array/slice type`) in std paths
    (`std_jit_env_set_var_smoke`, `std_jit_env_var_roundtrip`, `std_jit_vec_sort_probe`)
  - monomorphization errors in formatting/collection-heavy generic code
    (`std_jit_format_macro_probe`, `std_jit_hashmap_insert_get_probe`, `std_jit_btreemap_range_probe`)
  - trait-impl/vtable generic arg propagation hole
    (`std_jit_iter_repeat_take_collect_smoke` -> `GenericArgNotProvided`)
  - builtin-derive pseudo methods in call lowering are only implemented for
    `Clone::clone`; other builtin-derive methods currently fail fast if reached.
- Additional ambitious-blocker signals:
  - formatting-heavy paths still hit monomorphization errors in `core::fmt`
  - algorithm/collection-heavy paths (Vec sort / HashMap / BTreeMap) still surface
    indexing + monomorphization gaps in deeper generic code
  - sync-heavy probes (`mutex_lock`, `mutex_try_lock`, `arc_mutex`) now pass in latest full-suite runs;
    remaining blockers are in formatting/collections and indexing paths
- Runtime symbol resolution relies on process-global `dlopen` behavior; robustness improvements are possible.

## Recommended Fix Order

1. Implement static-operand codegen support in sync-heavy std paths.
   - Status: done (including `std_jit_arc_mutex_probe` after builtin-derive `Clone` lowering).

2. Fix `Index on non-array/slice type` codegen failures in std paths.
   - Repro targets: `std_jit_env_set_var_smoke`, `std_jit_env_var_roundtrip`,
     `std_jit_vec_sort_probe`.

3. Handle remaining trait-impl/vtable generic-arg propagation failures.
   - Repro target: `std_jit_iter_repeat_take_collect_smoke`.

4. Fix formatting/collection frontier blockers from ambitious probes.
   - Repro targets: `std_jit_format_macro_probe`, `std_jit_vec_sort_probe`,
     `std_jit_hashmap_insert_get_probe`, `std_jit_btreemap_range_probe`.

5. Unignore probes that now pass to keep regressions visible.
   - Candidate unignores: `std_jit_once_call_once_smoke`,
     `std_jit_string_from_smoke`, `std_jit_string_push_str_smoke`.

6. Re-run and unignore probes incrementally as each blocker is fixed.
   - Current state: `str_parse`, `vec_push`, `mutex_lock`, `mutex_try_lock`, and `arc_mutex`
     are unignored; `string_from` and `string_push_str` now pass but are still marked ignored.
   - New state: `std_jit_option_map_closure_probe`,
     `jit_float_scalar_pair_const_probe`, `jit_size_of_val_dyn_trait_probe`,
     `jit_explicit_enum_discriminant_probe`, `jit_copy_nonoverlapping_intrinsic_probe`, and
     `jit_write_bytes_intrinsic_probe` are unignored and passing.

7. Expand coverage with more deterministic std probes once blockers are cleared.
   - Candidate families: `std::thread::current`, `std::time`, and light `std::sync` probes.

8. Improve sysroot loading ergonomics/perf for tests.
   - Cache file discovery and/or loaded roots across tests when feasible.
   - Keep wall-time reasonable as std-smoke coverage grows.

9. Introduce a focused std-JIT test group in `just test-clif` docs/comments.
   - Make it easy to run only mirdataless std-JIT smoke tests locally.

## Next Candidate To Debug

- `std_jit_env_set_var_smoke`
  - reason: representative repro for the current indexing/codegen blocker
    (`Index on non-array/slice type`) that also affects other std probes.

## Non-Goals (Still True)

- Do not reintroduce mirdata body translation paths (`mirdata_codegen.rs`, old layout bridging, large mirdata schema).
- Do not use `.mirdata` for anything beyond crate disambiguator metadata in this flow.

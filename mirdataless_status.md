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

## Tests Revived

- Passing smoke test:
  - `std_jit_process_id_nonzero`
  - `std_jit_generic_identity_i32`
  - `std_jit_str_len_smoke`
- Additional probe test present but ignored:
  - `std_jit_process_id_is_stable_across_calls`
  - currently `#[ignore]` due to observed JIT flakiness where repeated calls can diverge.
  - `std_jit_vec_new_smoke`
  - currently `#[ignore]` due to unresolved symbol at JIT finalize:
    `alloc::raw_vec::RawVec::<T, A>::new_in`.
    The harness attempts to compile it, but MIR lowering currently returns
    `NotSupported("const block")`, so it falls back to import linkage; this symbol is not
    exported from `libstd.so` and cannot be resolved.
  - `std_jit_env_var_roundtrip`
  - currently `#[ignore]` due to `non-value const in ScalarPair constant` during codegen.

Validation recently run:

- `cargo check -p cg-clif` passes
- `just test-clif -E 'test(std_jit_process_id_nonzero)' --no-capture` passes
- `just test-clif -E 'test(std_jit_str_len_smoke)' --no-capture` passes
- `just test-clif -E 'test(std_jit_vec_new_smoke)' --no-capture --run-ignored all`
  now gets past previous codegen panics and fails at JIT finalize with unresolved symbol
  `_R...alloc7raw_vec...RawVec...new_in...`; traced cause is
  `fallback=import reason=mir_error error=NotSupported("const block")`.
- `just test-clif -E 'test(std_jit_env_var_roundtrip)' --no-capture --run-ignored all`
  now reaches codegen and fails with `non-value const in ScalarPair constant` (previous `HasErrorType` gone).

## What Is Still Missing

- Coverage is still narrow: only a small `std` smoke path is live.
- We do not yet have a broader stable suite of std-JIT end-to-end tests (collections, env/thread/time, etc.).
- The ignored repeated-call process-id test indicates a correctness/ABI/call-lowering issue that should be diagnosed.
- `Vec::new` path is now blocked at cross-crate MIR lowering for `const { ... }`.
  - `alloc::raw_vec::RawVec::<T, A>::new_in` lowering reports `NotSupported("const block")`.
  - import fallback cannot save this case because the symbol is not exported from `libstd.so`.
- `std::env::var` path is now blocked later in codegen by ScalarPair-constant lowering (`non-value const in ScalarPair constant`).
- Runtime symbol resolution relies on process-global `dlopen` behavior; robustness improvements are possible.

## Good Next Steps

1. Expand passing std smoke coverage beyond the current process/id + str len checks.
   - Candidate families: `std::env`, `std::thread::current`, `std::time`.
   - Keep tests small and deterministic.

2. Investigate and fix the repeated-call instability.
   - Repro target: `std_jit_process_id_is_stable_across_calls`.
   - Check call signature lowering, return handling, and import resolution consistency.

3. Add MIR lowering support for const blocks in the mirdataless path.
   - Repro target: `std_jit_vec_new_smoke` (`RawVec::new_in`).
   - Goal: avoid fallback-to-import for non-exported cross-crate std helpers using `const { ... }`.

4. Fix ScalarPair constant lowering for std flows (current blocker for `std_jit_env_var_roundtrip`).
   - Repro target: `std_jit_env_var_roundtrip`.
   - Handle non-value const forms used by `std::env::var` path when materializing ScalarPair constants.

5. Add a harness-level debug mode for unresolved/late-bound symbols.
   - Emit resolved mangled symbol + address during JIT setup when an env flag is set.
   - Helps quickly distinguish mangling/disambiguator bugs from call-lowering bugs.

6. Improve sysroot loading ergonomics/perf for tests.
   - Cache file discovery and/or loaded roots across tests when feasible.
   - Keep wall-time reasonable as std-smoke coverage grows.

7. Introduce a focused std-JIT test group in `just test-clif` docs/comments.
   - Make it easy to run only mirdataless std-JIT smoke tests locally.

## Non-Goals (Still True)

- Do not reintroduce mirdata body translation paths (`mirdata_codegen.rs`, old layout bridging, large mirdata schema).
- Do not use `.mirdata` for anything beyond crate disambiguator metadata in this flow.

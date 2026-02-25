# Mirdataless Harness Status

## Current Status

The `cg-clif` test harness now has a revived **mirdataless** std-JIT path in
`crates/cg-clif/src/tests.rs`.

What is in place:

- `jit_run_with_std` is restored with the new architecture (no mirdata body translation).
- Real sysroot sources (`core`, `alloc`, `std`) are loaded into a `TestDB` for r-a MIR/type queries.
- Reachable local code is compiled using the normal pipeline:
  - `collect_reachable_fns`
  - `compile_fn`
  - `compile_drop_in_place`
- External monomorphic std calls are resolved through existing mangling/link strategy:
  - crate disambiguators from `RA_MIRDATA` metadata (`extract_crate_disambiguators()`)
  - dynamic loading of `libstd.so` (`dlopen(..., RTLD_GLOBAL)`)
- `.mirdata` remains metadata-only (crate names + stable crate ids); no old body/layout schema was revived.

## Tests Revived

- Passing smoke test:
  - `std_jit_process_id_nonzero`
- Additional probe test present but ignored:
  - `std_jit_process_id_is_stable_across_calls`
  - currently `#[ignore]` due to observed JIT flakiness where repeated calls can diverge.

Validation recently run:

- `cargo check -p cg-clif` passes
- `just test-clif -E 'test(std_jit_process_id_nonzero)' --no-capture` passes

## What Is Still Missing

- Coverage is still narrow: only a small `std` smoke path is live.
- We do not yet have a broader stable suite of std-JIT end-to-end tests (collections, env/thread/time, etc.).
- The ignored repeated-call process-id test indicates a correctness/ABI/call-lowering issue that should be diagnosed.
- Runtime symbol resolution relies on process-global `dlopen` behavior; robustness improvements are possible.

## Good Next Steps

1. Expand passing std smoke coverage with monomorphic, exported std APIs.
   - Candidate families: `std::env`, `std::thread::current`, `std::time` where symbol export is stable.
   - Keep tests small and deterministic.

2. Investigate and fix the repeated-call instability.
   - Repro target: `std_jit_process_id_is_stable_across_calls`.
   - Check call signature lowering, return handling, and import resolution consistency.

3. Add a harness-level debug mode for unresolved/late-bound symbols.
   - Emit resolved mangled symbol + address during JIT setup when an env flag is set.
   - Helps quickly distinguish mangling/disambiguator bugs from call-lowering bugs.

4. Improve sysroot loading ergonomics/perf for tests.
   - Cache file discovery and/or loaded roots across tests when feasible.
   - Keep wall-time reasonable as std-smoke coverage grows.

5. Introduce a focused std-JIT test group in `just test-clif` docs/comments.
   - Make it easy to run only mirdataless std-JIT smoke tests locally.

## Non-Goals (Still True)

- Do not reintroduce mirdata body translation paths (`mirdata_codegen.rs`, old layout bridging, large mirdata schema).
- Do not use `.mirdata` for anything beyond crate disambiguator metadata in this flow.

read goal.md for high level goal

plan.md contains the high level plan
milestones.md contains the current status

cg_clif/ is the original rustc cranelift backend (git subtree). Read-only reference.
Always reference the upstream cg_clif/ when working on our codegen.

crates/cg-clif/ is our MIR->Cranelift codegen crate using r-a types. This is where active development happens. See crates/cg-clif/NOTES.md for architecture notes and porting status.
For debugging workflow, see crates/cg-clif/debugging.md (use two-commit `fix` + `wip-debug` flow, and limit noisy command output with `... | tee /tmp/clif.log | tail -n 120`).

rustc/ is the rust tree, but not a subtree. it is read only.

cargo .. -p .. is slow, avoid it.

Use `jj commit -m "<short commit message>"` to create commits.

AVOID HACKS, just say it to the user

You are in a "vibe coding" setting, user doesn't review the code, so you have to be very clear when
you are doing a hack or something that will bite us in future.

This is a new project, backward compatibility is never needed.

Common commands:
```bash
# refresh target/sysroot.mirdata used by clif tests
just update-mirdata

# run crates/cg-clif tests
just test-clif

# run a specific cg-clif test (good for debugging)
just test-clif -E 'test(mirdata_jit_iter_sum)' --no-capture

# run multiple selected tests concurrently (preferred for batch triage)
just test-clif -j 24 -E 'test(std_jit_vec_pop_smoke) or test(std_jit_option_take_smoke)' --no-fail-fast

# include ignored probes while still running the batch concurrently
just test-clif -j 24 -E 'test(std_jit_mutex_try_lock_smoke) or test(std_jit_once_call_once_smoke)' --run-ignored all --no-fail-fast

# note: in this setup, `--no-capture` disables nextest parallelism.
# Use `--no-capture` only for focused single-test debugging output.
```

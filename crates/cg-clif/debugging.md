# cg-clif debugging notes

Use a two-commit flow from the start: one commit for real fixes and one for
debug probes/logging.

Exact `jj` sequence:

```bash
# 1) Create fix commit, then debug child commit
jj new -m "fix: <topic>"
jj new -m "wip-debug: <topic>"

# 2) Get short change ids (copy both)
jj log -n 6 -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"'

# 3) Work in debug commit, then jump to fix commit to apply real changes
jj edit <fix_change_id>
jj edit <wip_debug_change_id>

# 4) Finalize: keep only real fix in main line
jj edit <fix_change_id>
jj new                        # empty working copy on top of fix
jj abandon <wip_debug_change_id>
```

Notes:

- `change_id` is stable across rebases and rewrites; once copied, keep using it.
- Do not repeatedly re-run `jj log` to "re-confirm" ids after each command.
- If a `jj` command exits successfully, trust it and continue.

Optimal probing strategy:

- Reproduce one failing test first; never debug multiple failing probes at once.
- Add a tiny differential probe (known-good path vs suspected-bad path).
- If they diverge, add one structural check only (`len`, pointer equality, or repr lane).
- Escalate to internal tracing only after probe divergence is confirmed.
- Keep tracing scoped to one symbol/path, then remove probe hooks in the real `fix` commit.

Keep command output short while preserving a log:

```bash
just test-clif std_jit_vec_push_smoke --run-ignored only --no-capture \
  2>&1 | tee /tmp/clif.log | tail -n 120
```

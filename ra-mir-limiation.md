# r-a MIR limitation: ownership and drop semantics around `Move`/`Copy`

## Summary

Our `crates/cg-clif/` backend cannot yet rely on rustc-style MIR ownership and
drop invariants for all paths. In particular, `OperandKind::Move` in r-a MIR can
appear on projections whose effective type is `Copy` (for example, raw-pointer
fields used by std sort internals).

If we treat every `Move` as ownership-consuming unconditionally, we can
incorrectly clear per-local drop flags and skip required destructor/writeback
effects (for example, `CopyOnDrop`-style guard behavior).

## Why this differs from upstream `cg_clif`

Upstream `cg_clif/` (rustc backend) relies on rustc MIR drop elaboration and
move semantics:

- it does not maintain backend-local drop flags,
- it marks call argument ownership from syntax (`Move` vs `Copy`),
- and it relies on MIR `Drop` terminators + drop glue resolution.

That model is principled for rustc MIR, but is not always valid for current
r-a MIR in this project.

## Current backend policy in `crates/cg-clif/`

To keep behavior correct for std code paths today, ownership classification is
type-based and conservative:

- `Move` and `Copy` both go through the same classifier,
- a place is ownership-consuming iff its resolved type is non-`Copy`,
- if place type resolution is ambiguous/non-structural, we conservatively treat
  it as ownership-consuming.

This same classification is used consistently for:

- drop-flag updates, and
- ABI argument lowering (`is_owned` decisions for indirect passing).

## Exit criterion for removing this workaround

We can delete this special handling once r-a MIR guarantees rustc-equivalent
ownership/drop invariants for these paths (or once backend-local drop flags are
eliminated in favor of fully elaborated MIR semantics).

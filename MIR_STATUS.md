# MIR Alignment Status

Tracks how r-a's MIR aligns with rustc's MIR, focused on codegen via cg_clif.

## Rvalue

| Variant | Status | Notes |
|---------|--------|-------|
| `Use` | aligned | |
| `Repeat` | aligned | |
| `Ref` | aligned | |
| `ThreadLocalRef(StaticId)` | type-aligned | Not emitted by lowering yet (r-a doesn't track `#[thread_local]` attribute) |
| `AddressOf(Mutability, Place)` | aligned | Emitted for raw pointer creation (`&raw`, `addr_of!`) |
| `Len` | aligned | |
| `Cast` | aligned | |
| `BinaryOp` | aligned | Includes overflow/unchecked variants (`AddWithOverflow`, etc.) |
| `UnaryOp` | aligned | |
| `Discriminant` | aligned | |
| `Aggregate` | aligned | |
| `ShallowInitBox` | aligned | Exists but not emitted (rustc removed it too) |
| `ShallowInitBoxWithAlloc` | non-standard | r-a invention for `#[rustc_box]` / `Expr::Box`. Handles heap allocation + box init. Codegen layer should decompose to alloc call. Rustc removed both `ShallowInitBox` and `NullaryOp`; modern `vec![]` uses `Box::new_uninit()` + intrinsics instead. |
| `CopyForDeref` | aligned | |

### Removed

- `NullaryOp(Infallible)` -- rustc removed `NullaryOp` entirely. r-a had it stubbed with `Infallible`. Removed.

## StatementKind

| Variant | Status | Notes |
|---------|--------|-------|
| `Assign` | aligned | |
| `FakeRead` | aligned | Stripped before codegen in rustc |
| `SetDiscriminant` | type-aligned | Not emitted by lowering; lowering uses `Aggregate` for enum construction |
| `Deinit` | aligned | |
| `StorageLive` | aligned | |
| `StorageDead` | aligned | |
| `Nop` | aligned | |

### Not implemented

- `Intrinsic(NonDivergingIntrinsic)` -- `assume`, `copy_nonoverlapping`. Commented out. Low priority for initial codegen.
- `Retag` -- Miri/Stacked Borrows only. Not needed for codegen.
- `AscribeUserType` -- type checking only. Not needed for codegen.

## TerminatorKind

| Variant | Status | Notes |
|---------|--------|-------|
| `Goto` | aligned | |
| `SwitchInt` | aligned | |
| `UnwindResume` | aligned | |
| `Abort` | aligned | |
| `Return` | aligned | |
| `Unreachable` | aligned | |
| `Drop` | aligned | |
| `Call` | aligned | |
| `Assert` | aligned | |
| `Yield` | aligned | |
| `CoroutineDrop` | aligned | |
| `FalseEdge` | aligned | Pre-drop-elaboration only |
| `FalseUnwind` | aligned | Pre-drop-elaboration only |

### Removed

- `DropAndReplace` -- rustc removed this. Was defined but never emitted by r-a's lowering. Deleted.

## ProjectionElem

| Variant | Status | Notes |
|---------|--------|-------|
| `Deref` | aligned | |
| `Field` | aligned | |
| `TupleOrClosureField` | non-standard | r-a uses this instead of `Field` for tuples/closures |
| `ConstantIndex` | aligned | |
| `Subslice` | aligned | |
| `OpaqueCast` | aligned | |
| `Index` | aligned | |
| `Downcast(VariantIdx)` | aligned | Emitted before field access on enum variants |

## OperandKind

| Variant | Status | Notes |
|---------|--------|-------|
| `Copy` | aligned | |
| `Move` | aligned | |
| `Constant` | aligned | |
| `Static(StaticId)` | non-standard | r-a uses a separate variant for statics. rustc uses `Constant`. Works for codegen but diverges from cg_clif expectations. |

## Shortcuts taken

- **Eval not updated for new constructs**: `ThreadLocalRef`, `SetDiscriminant` return `not_supported!` in const eval. These constructs are type-correct but eval can't execute them.
- **Borrowck minimally updated**: New rvalue/statement variants added to match arms but with no-op handling. Sufficient since codegen doesn't depend on borrowck.

## Non-standard constructs (kept intentionally)

- **`ShallowInitBoxWithAlloc`**: r-a invention for `#[rustc_box]` backward compat (toolchains < 1.86). Rustc removed both `ShallowInitBox` and `NullaryOp`, so there's no standard equivalent to align with. Codegen handles it directly.
- **`OperandKind::Static`**: r-a uses a separate variant. Codegen handles it specially.
- **`TupleOrClosureField`**: r-a uses this instead of `Field` for tuples/closures. Codegen handles it specially.

## No Infallible stubs remain

All `Infallible` stubs have been removed from MIR types.

## Remaining work (by priority for codegen)

1. `Intrinsic` statement -- needed when MIR uses `assume`/`copy_nonoverlapping`.
2. Coroutine support (`AggregateKind::Coroutine`, `Yield`/`CoroutineDrop`) -- needed for async/await.

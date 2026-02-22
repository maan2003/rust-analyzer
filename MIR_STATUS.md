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
| `Cast` | aligned | All cast kinds including `Transmute` |
| `BinaryOp` | aligned | Includes overflow/unchecked variants (`AddWithOverflow`, etc.) |
| `UnaryOp` | aligned | |
| `Discriminant` | aligned | |
| `Aggregate` | aligned | Includes `RawPtr(Ty, Mutability)` for fat pointer construction |
| `ShallowInitBox` | aligned | Exists but not emitted (rustc removed it too) |
| `ShallowInitBoxWithAlloc` | non-standard | r-a invention for `#[rustc_box]` / `Expr::Box` backward compat. |
| `CopyForDeref` | aligned | |

### CastKind variants

| Variant | Status | Notes |
|---------|--------|-------|
| `PointerExposeProvenance` | aligned | Renamed from `PointerExposeAddress` |
| `PointerWithExposedProvenance` | aligned | Renamed from `PointerFromExposedAddress` |
| `PtrToPtr` | aligned | |
| `PointerCoercion` | aligned | |
| `DynStar` | aligned | Not emitted, `not_supported!` in eval |
| `IntToInt` | aligned | |
| `FloatToInt` | aligned | |
| `FloatToFloat` | aligned | |
| `IntToFloat` | aligned | |
| `FnPtrToPtr` | aligned | |
| `Transmute` | aligned | Lowered from `transmute`/`transmute_unchecked` intrinsic calls |

### AggregateKind variants

| Variant | Status | Notes |
|---------|--------|-------|
| `Array` | aligned | |
| `Tuple` | aligned | |
| `Adt` | aligned | |
| `Union` | aligned | |
| `Closure` | aligned | |
| `RawPtr(Ty, Mutability)` | type-aligned | Not emitted by lowering yet |
| `Coroutine` | missing | Needed for async/await |
| `CoroutineClosure` | missing | Needed for async closures |

### Removed

- `NullaryOp(Infallible)` -- rustc removed `NullaryOp` entirely. Removed.

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

- `Intrinsic(NonDivergingIntrinsic)` -- `assume`, `copy_nonoverlapping`. Low priority for initial codegen.
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

- `DropAndReplace` -- rustc removed this. Deleted.

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
| `Static(StaticId)` | non-standard | r-a uses a separate variant for statics. rustc uses `Constant`. |

## Shortcuts taken

- **Eval not updated for new constructs**: `ThreadLocalRef`, `SetDiscriminant`, `RawPtr` aggregate return `not_supported!` in const eval.
- **Borrowck minimally updated**: New rvalue/statement variants added to match arms with no-op handling.
- **Transmute**: Lowering emits `Cast(Transmute)` instead of intrinsic call. Eval handles it (copies bytes).

## Non-standard constructs (kept intentionally)

- **`ShallowInitBoxWithAlloc`**: r-a invention for `#[rustc_box]` backward compat (toolchains < 1.86). Rustc removed both `ShallowInitBox` and `NullaryOp`.
- **`OperandKind::Static`**: r-a uses a separate variant.
- **`TupleOrClosureField`**: r-a uses this instead of `Field` for tuples/closures.

## No Infallible stubs remain

All `Infallible` stubs have been removed from MIR types.

## Lowering correctness notes

- **Arithmetic**: Lowering emits wrapping `BinOp::Add/Sub/Mul` for `+`/`-`/`*`. This matches rustc — overflow checks are inserted by a later MIR pass (`Assert` terminators using `AddWithOverflow`), not during initial lowering. The `WithOverflow`/`Unchecked` variants are defined for use by MIR passes and intrinsic lowering.
- **Intrinsics as calls**: Most intrinsics (`unchecked_add`, `saturating_add`, `copy_nonoverlapping`, `offset`, etc.) are lowered as regular `Call` terminators and handled by the eval shim. Only `transmute`/`transmute_unchecked` are specially lowered to `Cast(Transmute)`. Codegen will need to recognize these intrinsic calls.
- **Cast lowering**: `cast_kind()` in `lower.rs` covers Ptr↔Int, Int↔Int, Float↔Int, Float↔Float, Ptr↔Ptr, FnPtr→Ptr. Returns `not_supported!` for unknown combinations (conservative).
- **Cast classification is approximate without THIR**: `Expr::Cast` currently special-cases `source_ty.as_reference()` to `PointerCoercion(ArrayToPointer)` before `cast_kind()`. This is intentionally conservative but can differ from rustc's finer-grained cast classification.
- **Downcast**: Correctly emitted before field access on enum variants in pattern matching.
- **AddressOf**: Correctly emitted for `&raw const/mut` and `addr_of!` coercions.
- **LogicOp**: `&&`/`||` lowered as `BitAnd`/`BitOr` (not short-circuit). Has a FIXME.
- **Source-level raw-pointer `offset` in fixtures**: Bare fixtures can fail with `UnresolvedMethod(\"offset\")` because they do not model full core pointer inherent impls. For test scenarios, `minicore` now exposes a `ptr_offset` slice so method resolution can work when explicitly enabled.

## Expressions not lowered (`not_supported!`)

- async/await, async blocks, yield
- inline assembly (`builtin#asm`)
- const blocks
- `builtin#offset_of`
- tail calls (`become`)
- box patterns

These are intentional gaps — uncommon or unstable features.

## Remaining work (by priority for codegen)

1. Coroutine/CoroutineClosure aggregates -- needed for async/await.
2. `Intrinsic` statement -- `assume`, `copy_nonoverlapping`.
3. `UnwindAction` enum -- rustc uses this instead of `Option<BasicBlockId>` for unwind. Only matters for panic=unwind.
4. Lower more intrinsics to MIR constructs (e.g. `offset` → `BinOp::Offset`) as needed by codegen.
5. Plumb `minicore: ptr_offset` into integration-style test harnesses (like `crates/cg-clif` helpers) when source-level `p.offset(n)` coverage is needed.

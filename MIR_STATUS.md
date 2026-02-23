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
| `Coroutine` | type-aligned | Aggregate emitted; no capture analysis or body lowering yet |
| `CoroutineClosure` | type-aligned | Aggregate emitted; no capture analysis or body lowering yet |

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
- **Source-level raw-pointer pointer methods in fixtures**: Bare fixtures can fail with `UnresolvedMethod(\"offset\")` / `UnresolvedMethod(\"offset_from\")` because they do not model full core pointer inherent impls. For test scenarios, `minicore` exposes a `ptr_offset` slice (`offset`, `offset_from`, `offset_from_unsigned`) so method resolution can work when explicitly enabled.
- **cg-clif JIT test harness limitation**: `jit_run` compiles only explicitly listed root functions. Source-level pointer methods may still fail at runtime symbol resolution unless callees are also compiled (or tests use direct intrinsics / object-only checks).

## Expressions not lowered (`not_supported!`)

- await (needs desugar to poll/yield loop)
- async block/coroutine bodies (aggregate constructed, but body not lowered separately yet)
- inline assembly (`builtin#asm`)
- const blocks
- `builtin#offset_of`
- tail calls (`become`)
- box patterns

These are intentional gaps — uncommon or unstable features.

## Remaining work (by priority for codegen)

1. Coroutine capture analysis and body lowering -- aggregates exist but have no captures; bodies not lowered separately.
2. `Intrinsic` statement -- `assume`, `copy_nonoverlapping`.
3. `UnwindAction` enum -- rustc uses this instead of `Option<BasicBlockId>` for unwind. Only matters for panic=unwind.
4. Lower more intrinsics to MIR constructs (e.g. `offset` → `BinOp::Offset`) as needed by codegen.
5. Keep using explicit `minicore: ptr_offset` in fixtures that need source-level raw-pointer methods; bare fixtures intentionally stay minimal.

## Sysroot MIR export quality (ra-mir-export)

`ra-mir-export` exports optimized MIR from sysroot crates to `.mirdata` files.
8111 function bodies from 20 crates, 0 translation failures.

### Construct coverage (from /tmp/sysroot.mirdata analysis)

**Statements (79217 total):**

| Kind | Count | Notes |
|------|-------|-------|
| StorageDead | 29465 | |
| StorageLive | 26279 | |
| Assign | 23473 | |

Skipped (not needed for codegen): `FakeRead`, `PlaceMention`, `AscribeUserType`,
`Coverage`, `ConstEvalCounter`, `Retag`, `Intrinsic`, `BackwardIncompatibleDropHint`.

**Terminators (39224 total):**

| Kind | Count | Notes |
|------|-------|-------|
| Call | 22957 | |
| Return | 7803 | |
| Goto | 4129 | |
| SwitchInt | 2711 | |
| Unreachable | 651 | Includes fallback from TailCall/Yield/CoroutineDrop/InlineAsm |
| Drop | 565 | |
| Assert | 226 | |
| UnwindResume | 182 | |

**Rvalues (23473 assigns):**

| Kind | Count | Notes |
|------|-------|-------|
| Cast | 6857 | |
| Use | 6274 | |
| BinaryOp | 3979 | |
| Aggregate | 3318 | |
| Ref | 1058 | |
| Discriminant | 1028 | |
| UnaryOp | 471 | |
| RawPtr | 461 | |
| Repeat | 22 | |
| ThreadLocalRef | 5 | |

### Constant quality (35392 total)

| ConstKind | Count | % | Notes |
|-----------|-------|---|-------|
| ZeroSized | 23169 | 65.5% | FnDef types, unit values |
| Scalar | 5407 | 15.3% | Integer/float/bool literals |
| Unevaluated | 4746 | 13.4% | DefPathHash + generic args preserved |
| Todo | 1831 | 5.2% | Fallback (see below) |
| Slice | 239 | 0.7% | String literals, byte slices |

**ConstKind::Todo breakdown (1831):**

| Category | Count | Notes |
|----------|-------|-------|
| SIMD const generics (`IMM8`, `ROUNDING`, `SAE`, `SCALE`, etc.) | ~1500 | From `core::arch` / `compiler_builtins`. Not needed for core/alloc/std. |
| `indirect_const` | 204 | Constants stored in allocations (large arrays, struct literals). |
| `ptr:allocNN` | ~20 | Pointer constants to static data. |
| Other const params | ~100 | Various const generic params without scalar leaves. |

### Type quality (172536 total)

| Category | Count | % |
|----------|-------|---|
| Concrete types | 168453 | 97.6% |
| `Ty::Opaque` fallback | 4083 | 2.4% |

**All 4083 opaque types are `Alias(Projection, ...)`** -- associated type projections
like `<F as FnOnce>::Output`, `<Self as Iterator>::Item`. Concentrated in:
- `compiler_builtins` (~1900) -- SIMD trait impls
- `gimli` / `object` (~900) -- debug info/unwinding libraries
- `core` (~250) -- generic trait impls

These are expected for generic function bodies and resolve during monomorphization.

### Assessment for M11 (Vec, heap allocation)

The export quality is sufficient for consuming MIR bodies for core/alloc/std:
- Todo/Opaque fallbacks are concentrated in SIMD intrinsics (`compiler_builtins`,
  `core::arch`) and debug info libraries (`gimli`, `object`), not in
  core/alloc/std functions needed for Vec and heap allocation.
- 94.8% of constants are fully resolved (ZeroSized + Scalar + Unevaluated + Slice).
- 97.6% of types are fully resolved.
- Associated type projections (`Alias(Projection, ...)`) in generic bodies
  resolve to concrete types during monomorphization.

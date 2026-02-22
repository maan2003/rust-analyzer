# cg-clif: r-a's MIR -> Cranelift codegen

## Current porting state

What works:
- Scalar-only functions (`fn foo() -> i32 { 42 }` compiles to a valid object file)
- `Rvalue::Use` (copy/move/constant operands)
- Scalar constants (i8..i128, floats, pointers)
- `TerminatorKind::Return`, `Goto`, `Unreachable`
- Scalar locals as Cranelift variables, ZST locals
- Scalar params and return types
- `pointer.rs` ported (Pointer abstraction over addr/stack/dangling)
- `Rvalue::BinaryOp` — all integer arithmetic (add/sub/mul/div/rem),
  bitwise ops, shifts, comparisons, three-way `Cmp`. Float arithmetic
  (add/sub/mul/div) and comparisons. Pointer offset and comparisons.
  Unchecked variants handled. Overflow variants (`AddWithOverflow` etc.)
  stubbed — need CValue pairs.
- `Rvalue::UnaryOp` — `Neg` (ineg/fneg), `Not` (bnot for ints, icmp_imm for bool)
- `TerminatorKind::SwitchInt` — multi-way branching via `cranelift_frontend::Switch`
- `TerminatorKind::Call` — direct function calls (FnDef with scalar
  args/returns). Resolves callee via TyKind::FnDef → CallableDefId,
  gets callee MIR, builds Cranelift sig, declares in module, emits call.
  ZST args filtered. Tested with call chains and calls combined with branches.
- `TerminatorKind::Drop` — no-op jump (scalar types only, no drop glue yet)

Known bugs (divergence from upstream cg_clif):
- **Pointer `Offset` missing size scaling** — our `BinOp::Offset` does
  `iadd(ptr, offset)` but upstream does `iadd(ptr, imul_imm(offset, pointee_size))`.
  `ptr.offset(2)` should advance by `2 * sizeof(T)` bytes, not 2 bytes.
- **Constants only handle small scalars** — `const_to_i64` extracts raw bytes
  into i64. Missing: pointer constants (references to allocations/statics),
  slice constants (fat pointers), i128 (upstream uses `iconcat(lsb, msb)`),
  indirect constants (stored in allocations). String literals, `const &[T]`,
  etc. won't work.
- **No `PassMode` / ABI handling in calls** — we assume all args are by-val
  scalars or ZST. Upstream has `PassMode::Indirect` (pass-by-pointer for
  large structs), `PassMode::Pair` (scalar pairs like slices),
  `PassMode::Uniform`. Struct args/returns will produce wrong ABI.

What's missing:
- `CValue`/`CPlace` abstractions (currently using raw Cranelift `Value`)
- Overflow binops: `AddWithOverflow`/`SubWithOverflow`/`MulWithOverflow`
  (return tuple, need CValue pairs)
- Float `Rem` (needs libcall to fmod/fmodf)
- Indirect calls (fn pointers, closures)
- Struct/enum constructor calls (`CallableDefId::StructId`/`EnumVariantId`)
- Aggregates: non-scalar locals (need stack slots), projections
- Casts, discriminants, intrinsics, drop glue, etc.

Next steps (easiest to port):
1. Fix pointer `Offset` — one-line fix, multiply offset by pointee size
2. `CValue`/`CPlace` — value-with-layout abstraction (enables aggregates,
   overflow ops, projections)
3. Casts (`Rvalue::Cast`)
4. Aggregate support (struct/enum locals, field projections)

## Original cg_clif architecture

## Where the core logic lives

All codegen logic is in cg_clif itself. `rustc_codegen_ssa` is just a thin
trait (`CodegenBackend`) for plugging into rustc's driver/linker — zero code
generation logic. We don't need it.

## Key files in original cg_clif (at repo root `cg_clif/`)

- **`base.rs`** — Main codegen driver. `codegen_fn()` orchestrates per-function
  codegen. `codegen_fn_body()` walks MIR basic blocks, dispatches on
  `StatementKind` and `TerminatorKind`. `codegen_stmt()` handles rvalues,
  assignments, discriminants.

- **`value_and_place.rs`** — `CValue`/`CPlace` abstractions. CValue = read-only
  value (ByRef | ByVal | ByValPair). CPlace = mutable location (Var | VarPair |
  Addr). All operands flow through CValue, all assignment targets through
  CPlace.

- **`common.rs`** — `FunctionCx` struct (per-function state: builder, module,
  tcx, mir, local_map, block_map). Type mapping helpers
  (`scalar_to_clif_type`, `clif_type_from_ty`). Value manipulation utilities.

- **`abi/`** — Call/return codegen, calling convention handling.

- **`cast.rs`, `num.rs`** — Cranelift IR patterns for casts and arithmetic.

- **`discriminant.rs`** — Enum discriminant get/set.

- **`constant.rs`** — Const evaluation to Cranelift constants.

- **`intrinsics/`** — Intrinsic function implementations.

- **`pointer.rs`** — Pointer abstraction over addr/stack/dangling. Pure
  Cranelift, no rustc types. Already ported.

## Rustc dependencies

Every core file is parameterized over `rustc_middle`'s `Ty<'tcx>`,
`Body<'tcx>`, `Instance<'tcx>`. These are fundamentally different from r-a's
`Ty<'db>`, `MirBody`, etc. Files can't be reused verbatim but the
algorithms/Cranelift IR patterns are directly applicable.

### What's available as ra-ap

| rustc crate | ra-ap? | Notes |
|---|---|---|
| `rustc_abi` | Yes (`ra-ap-rustc_abi`) | Already using |
| `rustc_target` | Partial (`rac-abi`) | Target specs, calling conventions |
| `rustc_index` | Yes (`ra-ap-rustc_index`) | `Idx` trait, `IndexVec` |
| `rustc_middle` | No | The blocker: `Ty`, MIR types, `TyCtxt` |
| `rustc_codegen_ssa` | No | Not needed (just driver glue) |
| `rustc_data_structures` | No | Only `FxHashMap` (trivial to replace) |
| `rustc_hir` | No | `DefId` (r-a has its own) |
| `rustc_session` | No | Not needed |
| `rustc_span` | No | `Symbol` (r-a has its own) |
| `rustc_const_eval` | No | r-a has its own const eval |
| `rustc_symbol_mangling` | No | Need our own |

## Files portable as-is (no rustc types)

- `pointer.rs` — Done

## Files portable with type substitution (algorithms reusable)

- `cast.rs`, `num.rs`, `codegen_i128.rs` — Cranelift IR lowering patterns
- `discriminant.rs` — Enum discriminant logic
- `common.rs` — Type mapping (already partially ported as `scalar_to_clif_type`)

## Files not portable (deep rustc integration)

- `driver/` — rustc driver integration
- `compiler_builtins.rs` — rustc-specific
- `concurrency_limiter.rs`, `config.rs` — rustc session
- `global_asm.rs`, `inline_asm.rs` — rustc AST types
- `linkage.rs` — rustc linker integration

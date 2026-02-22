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

What's missing:
- Arithmetic: `BinaryOp`, `CheckedBinaryOp`, `UnaryOp` rvalues
- Control flow: `SwitchInt` terminator (needed for if/match)
- Function calls: `Call` terminator
- Aggregates: non-scalar locals (need stack slots), projections
- `CValue`/`CPlace` abstractions (currently using raw Cranelift `Value`)
- Casts, discriminants, intrinsics, drops, etc.

Next steps (easiest to port):
1. `BinaryOp`/`CheckedBinaryOp` — port `num.rs` patterns for arithmetic
2. `SwitchInt` terminator — Cranelift block jumps
3. `Call` terminator — function calls

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

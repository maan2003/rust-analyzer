# rac: Fast Rust Compilation via rust-analyzer + Cranelift

## The Problem

Rust compilation is slow. Even with rustc's incremental compilation and
Cranelift as a backend:

- **Coarse invalidation.** The unit of reuse is a CGU, not a function.
  Changing one function body can invalidate a CGU containing hundreds of
  functions, all of which get re-compiled.
- **Per-invocation overhead.** Each `rustc` invocation pays startup cost,
  loads crate metadata, and reconstructs state from the incremental cache,
  even when the actual work to redo is small.
- **Limited parallelism.** Within a single crate, rustc's frontend is largely
  sequential. Parallelism exists at the CGU level for codegen, but analysis
  (type checking, trait solving, MIR lowering) is not parallel.
- **Crate = compilation unit.** Each crate is a separate query graph with its
  own invocation of rustc. Cross-crate information flows only through
  serialized metadata, not shared in-memory state. A workspace with many
  small crates pays the per-invocation overhead for each one.

## The Insight

rust-analyzer already has most of a Rust compiler frontend running incrementally
on Salsa: parsing, name resolution, type inference, MIR lowering — all cached
per-function. When you're editing in your IDE, r-a is continuously doing the
analysis work that rustc would redo from scratch. This work can be reused for
compilation.

## The Approach

Build an AOT compiler backend that takes r-a's MIR and produces native code via
Cranelift.

### Architecture

```
 ┌──────────────────────────────────────────────────────┐
 │  rust-analyzer (existing)                            │
 │  - Parses source, resolves names, infers types       │
 │  - Lowers to MIR (per-function, Salsa-cached)        │
 │  - Monomorphizes generic MIR on demand               │
 │  - Computes layouts via rustc_abi (identical to rustc)│
 │  - Solves traits via rustc_next_trait_solver          │
 └──────────────┬───────────────────────────────────────┘
                │ MirBody per function
                ▼
 ┌──────────────────────────────────────────────────────┐
 │  ra-codegen (new)                                    │
 │  - Translates r-a MIR → Cranelift IR                 │
 │  - Computes FnAbi via rustc_target (shared crate)    │
 │  - Emits object files via cranelift-object           │
 │  - Generates drop glue, vtables, intrinsics          │
 └──────────────┬───────────────────────────────────────┘
                │ .o files
                ▼
 ┌──────────────────────────────────────────────────────┐
 │  Linker                                              │
 │  - Links against rustc-compiled dependency dylibs    │
 │  - Produces final executable                         │
 └──────────────────────────────────────────────────────┘
```

### Dependency Handling: ra-mir-export

r-a's MIR lowering doesn't cover all Rust constructs (async/await, inline asm,
etc.), and dependencies may use any of them. To handle this:

A companion **nightly** rustc driver (`ra-mir-export`) compiles dependencies
with `-Z always-encode-mir`, reads the resulting MIR via rustc queries, and
translates it to r-a's MIR format. This serves as:

- **Fallback** for constructs r-a can't lower
- **Pre-compilation** for the dependency tree (run once, cache)
- **Gradual migration path** — start with the converter for everything, move
  more to r-a's native lowering over time

## Why This Works

### Shared foundations with rustc

r-a isn't a separate implementation — it shares critical rustc crates:

| Crate | What it provides |
|-------|-----------------|
| `rustc_type_ir` | Type representation (`TyKind`, `GenericArgs`) |
| `rustc_abi` | Layout computation (`LayoutCalculator`, `TargetDataLayout`) |
| `rustc_next_trait_solver` | Trait resolution |
| `rustc_target` (to add) | Calling conventions, `FnAbi`, `PassMode` |

Layouts are identical to rustc's **when built against the same rustc revision
and target data layout** — same `LayoutCalculator`, same
`TargetDataLayout`. Under that toolchain lock, linking against
rustc-compiled code is safe: struct sizes, field offsets, enum discriminants,
and niche optimizations all match.

### cg_clif as a blueprint

`rustc_codegen_cranelift` is a mature, working Cranelift backend for rustc. The
MIR → Cranelift IR translation is ~3000 lines of mechanical match arms. The
same patterns apply to r-a's MIR — the Cranelift API calls are identical, only
the input types differ.

## Future: JIT + Incremental Daemon

Once AOT compilation works, the same codegen can power a JIT mode:

- Use `cranelift-jit` instead of `cranelift-object`
- Salsa tells you which functions changed on each edit
- Recompile only those functions, swap function pointers
- Hot reload on save, sub-10ms edit-to-running-code

This is a separate phase. Get AOT right first.


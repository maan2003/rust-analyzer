# Plan: rac

**Take everything here with a grain of salt.** This plan is based on reading
the code, not building it. Actual effort will vary — some things will be easier
than expected, others will surface problems we haven't anticipated.

## Phase 0: Preparation

- Add `ra-ap-rustc_target` as a dependency of r-a
- Implement `TyAbiInterface` for r-a's `Ty` — this unlocks `rustc_target`'s
  calling convention machinery
- Implement `HasTargetSpec` and `HasX86AbiOpt` (or equivalent per target)
  for the codegen context — required by `adjust_for_rust_abi` /
  `adjust_for_foreign_abi`
- Pin a nightly toolchain revision for `ra-mir-export` and keep `ra-ap-*`
  crates aligned with that same rustc revision

## Phase 1: ra-mir-export

Build a **nightly-only** rustc driver that compiles a crate (with
`-Z always-encode-mir`), reads the resulting MIR via rustc queries, and
translates it to our serialized MIR format.

This gives us MIR for any Rust code — including things r-a can't lower yet
(async, inline asm, etc.) and all dependencies. It's the foundation that lets
us make progress on codegen without being blocked by r-a's MIR coverage.

### Serialization format

The format must capture:
- Monomorphized MIR bodies (rustc stores generic MIR; we either export
  monomorphized instances or monomorphize on import)
- Full type information sufficient to compute layouts on the r-a side
- Static/const initializer bodies
- Mangled symbol names matching rustc's output (v0 mangling)

Design the format and write a round-trip test before building the converter.

### Scope

Run the converter on std + the dependency tree, cache the output.

## Phase 2: MIR → Cranelift Codegen

Build `ra-codegen` — a crate that takes MIR (from either source) and produces
Cranelift IR.

- Fork/adapt cg_clif's translation patterns for our MIR representation
- `CValue`/`CPlace` abstraction for Cranelift values with layout info
- Statement and terminator dispatch (match arms over MIR constructs)
- Arithmetic, casts, enum discriminants — mostly Cranelift builder calls
- Note: r-a routes all binary ops through `CheckedBinaryOp` (returns
  `(result, overflow_flag)` tuple), `BinaryOp` is stubbed as `Infallible`.
  Codegen must handle the tuple result correctly.
- ABI: use `rustc_target::callconv` to compute `PassMode` per argument,
  translate to Cranelift `AbiParam`
- Drop glue generation
- Vtable layout and trait object dispatch
- Intrinsics — start with the essential ones, add as needed

### Mono-item collection

Before codegen, walk MIR starting from `main` to discover all reachable
monomorphized functions (similar to rustc's `MonoItem` collection). For each
call site, resolve the callee, determine its generic args, monomorphize, and
recurse. This produces the full set of functions to compile.

### Symbol mangling

Implement rustc's v0 mangling scheme. Symbols must match what rustc produces
so the linker can resolve calls into rustc-compiled rlibs/dylibs. This
requires mapping r-a's `DefPath` representation to v0 mangling inputs
(crate disambiguators, `DefPathData`, generic substitutions).

### Simplifications

- **Skip borrow checking.** Lifetimes are erased before codegen — borrow
  checking doesn't affect generated code. Run real rustc in the background
  for diagnostics.
- **Skip MIR optimizations.** No inlining, const propagation, or dead code
  elimination passes. Feed r-a's unoptimized MIR straight to Cranelift and
  let Cranelift handle optimization at the IR level. Slower output, but far
  less code to write.
- **`panic=abort` only.** No unwinding support. Panics call `abort()`, no
  cleanup blocks, no exception tables. This simplifies drop glue (no unwind
  paths) and eliminates `UnwindAction` complexity entirely.
- **No debug info.** Skip DWARF generation. Lose debugger support and
  symbolized stack traces initially. Removes a whole subsystem from scope.
- **Parallel codegen per function.** Each function's MIR → Cranelift
  translation is independent. Cranelift's `Context` is per-function.
  Compile N functions on N threads, collect the object code.

## Phase 3: AOT Driver

Wire up the codegen into an end-to-end compiler. This phase exists to
**validate correctness**, not to be fast. All speed investment goes into the
JIT phase.

- Use `cranelift-object` to emit `.o` files
- Shell out to a linker (or `rustc` as a linker driver) to produce the
  final binary. Don't build custom linker invocation logic — let
  rustc/cc/mold handle library search paths, native deps, and platform
  quirks.
- Handle entry point glue: emit a symbol that `std::rt::lang_start` can
  call (the real chain is: linker entry → `lang_start` in std → user
  `main`). The emitted symbol must have the right mangled name and
  signature.
- Use a fast linker (mold, lld) to reduce link overhead, but don't
  optimize linking further — the JIT path eliminates it entirely.

**First milestone**: `fn main() {}` compiles and runs.

Immediate follow-up: `fn main() { println!("hello"); }` compiles and runs.

## Phase 4: Fill r-a's MIR Gaps

With the converter as fallback, incrementally improve r-a's native MIR
lowering so fewer things need the rustc roundtrip:

- Uncomment and wire `Downcast`, `SetDiscriminant`
- Implement `BinaryOp` as a real Rvalue
- async/await, inline asm, const blocks — as needed

Each improvement means that function's MIR comes from r-a (Salsa-cached,
incremental) instead of the converter (batch, from-scratch).

## Phase 5: JIT + Incremental Daemon

Once AOT works, switch `cranelift-object` for `cranelift-jit`. This is
where the speed payoff happens — no linker, no disk I/O, no object files.

- Dependencies (std, crates) loaded once as dylibs via `libloading`
  (one-time `cargo build` with appropriate flags to produce dylibs, then
  cached). cg_clif's JIT driver does this in ~50 lines.
- `cranelift-jit` compiles functions into executable memory
- Salsa-driven invalidation: recompile only changed functions
- Function pointer swaps for hot reload
- Lazy compilation: compile functions on first call
- Sub-10ms edit-to-running-code

### Linking is eliminated, not optimized

The JIT path has no linker invocation. Changed functions are recompiled
in-memory and their pointers swapped. The only "linking" is the initial
dylib load for dependencies, which happens once at startup.

## Proc Macros

Proc macros (including `#[derive(...)]`, `serde`, `tokio::main`, etc.) must
run during compilation. r-a already handles proc macro expansion via a
separate proc macro server process. The rac compilation path should reuse
r-a's existing proc macro infrastructure rather than building a separate
solution.

## Open Questions

- How to handle the serialization format for MIR interchange — needs to be
  fast to read, stable enough across versions, and capture the full
  information listed in Phase 1
- How much of std should come from the converter vs r-a's native lowering
- When to add debug info (DWARF) support
- When to add unwinding support (`panic=unwind`)

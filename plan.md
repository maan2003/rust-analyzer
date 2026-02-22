# Plan: rac

**Guiding principle: unknown unknowns first.** Tackle the risky, novel parts
early to learn what actually works. Push well-understood mechanical work
(MIR format serialization, completeness) to later phases where it can't
block discovery.

## Phase 0: Preparation ✅

- Vendored `rustc_target` ABI code into `rac-abi` crate (ra-ap-rustc_target
  doesn't exist as a published crate)
- Replaced hir-ty's hand-rolled index types (`RustcFieldIdx`,
  `RustcEnumVariantIdx`) with rac-abi's `FieldIdx`/`VariantIdx`
- Implemented `TyAbiInterface` for r-a's `Ty` (via `AbiTy` newtype for
  Display requirement)
- Implemented `HasDataLayout`, `HasTargetSpec`, `HasX86AbiOpt` on `CodegenCx`
- `compute_fn_abi` wired to rac-abi's real `adjust_for_rust_abi`

### Known limitations from Phase 0
- `is_transparent` returns `false` (trait signature has no `cx` param to
  query db for repr attrs). Affects ABI on riscv/loongarch/sparc64 for
  transparent types.
- `ty_and_layout_pointee_info_at` stubbed to `None`.
- `TargetSpec` population from target triple not yet implemented.

## Phase 1: Single-function codegen spike

**Goal: translate one MIR body → Cranelift IR → object file.** No linking,
no std, no mangling. Just prove the MIR→CLIF translation works.

- Create `ra-codegen` crate with cranelift dependencies
- Implement `CValue`/`CPlace` (value-with-layout abstraction, adapt from
  cg_clif)
- Translate a trivial function: `fn foo() -> i32 { 42 }`
  - Function signature from `FnAbi` → Cranelift `AbiParam`s
  - MIR statement/terminator dispatch (start with Return, Assign, Const)
  - Emit via `cranelift-object`, verify with `objdump`
- Incrementally add MIR constructs: arithmetic (`CheckedBinaryOp`), locals,
  control flow (if/match → MIR blocks), function calls
- Write a test harness that compiles a function and calls it via `dlopen`
  or similar, to validate output without a full driver

### What this surfaces early
- Whether r-a's MIR representation maps cleanly to Cranelift
- Layout/ABI integration issues in practice (not just type-checking)
- What's missing from r-a's MIR lowering for even simple code

## Phase 2: End-to-end `fn main() {}`

**Goal: produce a running binary.** This forces solving the integration
problems: entry points, symbol names, linking against std.

### Symbol mangling

Implement rustc's v0 mangling scheme. Symbols must match what rustc produces
so the linker can resolve calls into rustc-compiled rlibs/dylibs. This
requires mapping r-a's `DefPath` representation to v0 mangling inputs
(crate disambiguators, `DefPathData`, generic substitutions).

This is well-specified (the v0 scheme is documented) but fiddly. Needs to
happen before linking works.

### Target spec population

Build `rac_abi::spec::Target` from the actual target triple. Needed for
correct ABI computation on non-x86_64 targets, but x86_64-unknown-linux-gnu
can be hardcoded initially.

### Linking

- Use `cranelift-object` to emit `.o` files
- Shell out to `rustc` as linker driver (handles library search paths,
  native deps, platform quirks)
- Entry point glue: emit symbol that `std::rt::lang_start` calls
  (linker entry → `lang_start` in std → user `main`)

### Mono-item collection

Walk MIR starting from `main` to discover reachable monomorphized functions.
For `fn main() {}` this is trivial (just main itself), but the infrastructure
is needed for anything larger.

**Milestone: `fn main() {}` compiles and runs.**

Follow-up: `fn main() { println!("hello"); }` — this pulls in a huge
transitive closure through std and will likely be the point where we need
either ra-mir-export or a way to link against pre-compiled std objects.

## Phase 3: Codegen breadth

Expand the set of Rust constructs the codegen handles. Order by what's
needed to compile progressively more complex programs.

- Struct/enum layout, field access, discriminant read/write
- References, raw pointers, pointer arithmetic
- Trait object dispatch (vtable layout, virtual calls)
- Drop glue generation
- Closures (capture layout, `Fn`/`FnMut`/`FnOnce` dispatch)
- Essential intrinsics (add as needed — `size_of`, `copy`, `transmute`,
  `abort`, etc.)
- Static/const initializers

### Simplifications (maintained throughout)

- **`panic=abort` only.** No unwinding, no cleanup blocks, no exception
  tables. Panics call `abort()`.
- **No debug info.** Skip DWARF generation entirely.
- **No MIR optimizations.** Feed r-a's unoptimized MIR straight to
  Cranelift.
- **Skip borrow checking.** Doesn't affect codegen. Run rustc in background
  for diagnostics.

## Phase 4: ra-mir-export (dependency MIR)

**Deferred until we actually need it.** Build the converter when codegen is
mature enough that the bottleneck is missing MIR for std/dependencies, not
missing codegen capabilities.

Build a **nightly-only** rustc driver that compiles a crate (with
`-Z always-encode-mir`), reads MIR via rustc queries, and translates to
r-a's MIR format.

### Serialization format

The format must capture:
- Monomorphized MIR bodies
- Full type information sufficient to compute layouts on the r-a side
- Static/const initializer bodies
- Mangled symbol names matching rustc's output

### Scope

Run the converter on std + the dependency tree, cache the output.

### Alternative: link against rustc-compiled objects directly

For many cases, we don't need the MIR at all — we just need the symbol to
exist in a `.rlib`/`.so`. Only functions we want to re-compile from source
need their MIR. This may let us defer the converter even further.

## Phase 5: Fill r-a's MIR gaps

Incrementally improve r-a's native MIR lowering so fewer things need the
rustc roundtrip.

### Done

- `BinaryOp` as a real Rvalue (renamed from `CheckedBinaryOp`)
- `NullaryOp` stub removed (rustc replaced with lang items)
- `AddressOf(Mutability, Place)` for raw pointer creation
- `Downcast(VariantIdx)` projection for enum variant field access
- `SetDiscriminant` statement
- Overflow/unchecked `BinOp` variants (`AddWithOverflow`, `SubWithOverflow`,
  `MulWithOverflow`, `AddUnchecked`, `SubUnchecked`, `MulUnchecked`,
  `ShlUnchecked`, `ShrUnchecked`, `Cmp`)
- `DropAndReplace` removed (rustc removed it, r-a never emitted it)
- `ThreadLocalRef(StaticId)` — real variant (no more `Infallible` stub).
  Not emitted by lowering yet (r-a doesn't track `#[thread_local]`).
- Zero `Infallible` stubs remain in MIR types

### Remaining (by priority for codegen)

- `OperandKind::Static` — non-standard, rustc uses `Constant` for statics.
  Works for codegen but diverges from cg_clif's expectations.
- `Intrinsic` statement (`NonDivergingIntrinsic`: `assume`,
  `copy_nonoverlapping`) — needed when lowering uses these intrinsics
- Coroutine support (`AggregateKind::Coroutine`, `Yield`/`CoroutineDrop`
  terminators) — needed for async/await
- async/await, inline asm, const blocks, tail calls — as needed

### Skippable (not needed for codegen)

- `Retag` — only for Stacked Borrows / Miri, not emitted in codegen
- `AscribeUserType` — only for type checking, stripped before codegen

Each improvement means that function's MIR comes from r-a (Salsa-cached,
incremental) instead of the converter (batch, from-scratch).

## Phase 6: JIT + Incremental Daemon

Once AOT works, switch `cranelift-object` for `cranelift-jit`. This is
where the speed payoff happens — no linker, no disk I/O, no object files.

- Dependencies loaded once as dylibs via `libloading`
- `cranelift-jit` compiles functions into executable memory
- Salsa-driven invalidation: recompile only changed functions
- Function pointer swaps for hot reload
- Lazy compilation: compile functions on first call

### Linking is eliminated, not optimized

The JIT path has no linker invocation. Changed functions are recompiled
in-memory and their pointers swapped. The only "linking" is the initial
dylib load for dependencies, which happens once at startup.

## Proc Macros

r-a already handles proc macro expansion via a separate proc macro server
process. The rac compilation path reuses that infrastructure.

## Open Questions

- Can we link against rustc-compiled `.rlib` objects without the full
  ra-mir-export converter? (Just need symbol resolution, not MIR.)
- How to populate `TargetSpec` correctly — parse rustc's target JSON?
  Hardcode common targets? Query `rustc --print target-spec-json`?
- `is_transparent` in `TyAbiInterface` — the trait provides no `cx`.
  Options: add a thread-local, change rac-abi's trait, or accept the
  limitation.
- When to add debug info (DWARF) support
- When to add unwinding support (`panic=unwind`)
- look into TDPE LLVM as well

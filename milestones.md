# Milestones

Each milestone produces something testable. If a milestone's output is wrong,
you know exactly where the bug is before more complexity piles on.

Ordered to surface unknown unknowns early. Well-understood mechanical work
(serialization, completeness) comes later.

---

## Phase 0: Preparation ✅

### M0: TyAbiInterface compiles ✅

Vendored `rustc_target` ABI code into `rac-abi`. Implemented `TyAbiInterface`
for r-a's `Ty` (via `AbiTy` newtype), `HasDataLayout`, `HasTargetSpec`,
`HasX86AbiOpt` on `CodegenCx`. `compute_fn_abi` calls rac-abi's real
`adjust_for_rust_abi`.

### M0.5: MIR aligned with rustc for codegen ✅

Aligned r-a's MIR data structures with rustc so cg_clif can translate them:
`BinaryOp` with overflow/unchecked variants, `AddressOf`, `Downcast`
projection, `SetDiscriminant` statement. Removed dead stubs (`NullaryOp`,
old `BinaryOp(Infallible)`). Later: removed `DropAndReplace`, added
`CastKind::Transmute`, `AggregateKind::RawPtr`, renamed provenance casts,
`ThreadLocalRef(StaticId)`. See `MIR_STATUS.md` for full alignment status.

---

## Phase 1: Single-function codegen spike ✅

### M1: One function → object file ✅

Created `cg-clif` crate. Translates r-a MIR to Cranelift IR, emits via
`cranelift-object`. 14 object-file tests verify compilation of scalars,
arithmetic, control flow, and function calls.

### M2: Call the compiled function ✅

Used `cranelift-jit` instead of dlopen — compiles and executes functions
in-memory. 10 JIT tests assert correct return values.

### M3: Arithmetic and locals ✅

Full scalar codegen working:
- `BinaryOp`: all integer arithmetic (add/sub/mul/div/rem), bitwise ops,
  shifts, comparisons, three-way `Cmp`. Float arithmetic and comparisons.
  Pointer offset and comparisons. Unchecked variants handled.
- `UnaryOp`: `Neg` (int/float), `Not` (bnot for ints, icmp_imm for bool).
- Scalar locals as Cranelift variables, ZST locals, scalar params/returns.
- `SwitchInt` multi-way branching, direct function calls (call chains,
  calls combined with branches), `Drop` as no-op jump.

Known bugs: pointer `Offset` missing size scaling, constants limited to
small scalars, no `PassMode`/ABI handling. See `crates/cg-clif/NOTES.md`.

---

## Phase 1.5: Codegen completeness (current)

Before linking against std, fill out codegen for the MIR constructs we'll
encounter. These can all be tested via JIT without a linker.

### M3.1: CValue/CPlace abstraction

Port `value_and_place.rs` pattern: CValue (ByRef | ByVal | ByValPair),
CPlace (Var | VarPair | Addr). Enables non-scalar locals (stack slots),
overflow binops (return tuples), and field projections.

### M3.2: Casts (`Rvalue::Cast`)

IntToInt, FloatToInt, IntToFloat, FloatToFloat, PtrToPtr,
PointerExposeProvenance, PointerWithExposedProvenance, Transmute.

### M3.3: Aggregates

Struct/enum/tuple locals on stack slots, `Aggregate` rvalue construction,
`Field`/`TupleOrClosureField` projections, `Downcast` for enum variants,
`Discriminant` rvalue, `SetDiscriminant` statement.

### M3.4: Remaining scalar ops

- Overflow binops (`AddWithOverflow` etc.) — need CValue pairs
- Float `Rem` (libcall to fmod/fmodf)
- Fix pointer `Offset` size scaling

---

## Phase 2: End-to-end `fn main() {}`

### M4: Symbol mangling

Implement v0 mangling for a function. Test by mangling `main` in a known
crate and comparing against `rustc`'s output.

Proves: symbols will resolve at link time.

### M5: Empty main runs

Compile `fn main() {}` → `.o` → link against std → run.

Proves: `cranelift-object` emits valid object, linker invocation works,
entry point glue (`lang_start` → `main`) is correct, v0 mangling produces
the right symbol name.

### M6: Exit code

```rust
fn main() -> ! {
    std::process::exit(42);
}
```

Proves: calling an extern function with arguments, linking against std
symbols. Verify by checking exit code.

---

## Phase 3: Codegen breadth

### M7: Control flow + function calls (partially done)

`SwitchInt`, branching, and intra-crate direct calls already work for
scalar types (tested via JIT in M3). What remains: linking compiled
functions together in an object file, calling `std::process::exit`
(extern/ABI), and running as a real executable.

```rust
fn add(a: i32, b: i32) -> i32 {
    if a > 0 { a + b } else { b }
}

fn main() -> ! {
    std::process::exit(add(19, 23));
}
```

### M8: Structs and enums

```rust
struct Point { x: i32, y: i32 }
enum Dir { Left, Right }

fn main() -> ! {
    let p = Point { x: 3, y: 4 };
    let code = match Dir::Right {
        Dir::Left => p.x,
        Dir::Right => p.y,
    };
    std::process::exit(code);
}
```

Proves: `Aggregate` construction, field projection, enum discriminant,
layout correctness (wrong offsets produce wrong values silently).

### M9: Generics and monomorphization

```rust
fn pick<T: Copy>(a: T, b: T, first: bool) -> T {
    if first { a } else { b }
}

fn main() -> ! {
    std::process::exit(pick(7, 3, true));
}
```

Proves: mono-item collection, type substitution in MIR, compiled code
uses concrete types.

### M10: Trait objects and dynamic dispatch

```rust
trait Animal {
    fn legs(&self) -> i32;
}
struct Dog;
impl Animal for Dog {
    fn legs(&self) -> i32 { 4 }
}

fn count(a: &dyn Animal) -> i32 { a.legs() }

fn main() -> ! {
    std::process::exit(count(&Dog));
}
```

Proves: vtable generation, dynamic dispatch, unsizing coercion.

### M11: Drop and heap allocation

```rust
fn main() -> ! {
    let v = vec![10, 20, 30];
    let code = v[1];
    drop(v);
    std::process::exit(code);
}
```

Proves: heap allocation, indexing, drop glue for `Vec`. Likely the point
where we need dependency MIR or linking against pre-compiled std objects.

### M12: println

```rust
fn main() {
    println!("hello, world!");
}
```

Proves: `format_args!` → `fmt::Write` → stdout chain. Huge std surface area.
When this works, most "normal" Rust code is within reach.

---

## Phase 4: ra-mir-export (when needed)

### M13: Round-trip a trivial function

Converter reads `optimized_mir` for a single function from a rustc-compiled
rlib, writes in our format, deserializes back, checks basic blocks/terminators
survived.

### M14: Export std

Run converter on `core`, `alloc`, `std`. Spot-check well-known functions.
Doesn't need to be perfect — just structurally valid.

---

## Phase 5: Expanding coverage

### M15: A real crate

Compile a small real-world crate (CLI tool, parser). Surfaces every edge
case the targeted milestones missed.

### M16: Correctness suite

Run a subset of rustc's run-pass tests. Track pass rate over time.

---

## Phase 6 (future): JIT

### M17: JIT hello world

Same as M12 but via `cranelift-jit` — compile in-memory, run without
producing an object file.

### M18: Incremental recompilation

Change a function body, recompile only that function, re-run. Salsa
identifies invalidated MIR, Cranelift recompiles, function pointer swapped.

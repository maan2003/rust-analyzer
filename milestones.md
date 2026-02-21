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

---

## Phase 1: Single-function codegen spike

### M1: One function → object file

Create `ra-codegen` crate. Take `fn foo() -> i32 { 42 }`, get its MIR from
r-a, translate to Cranelift IR, emit via `cranelift-object`. Verify with
`objdump` that the function exists and has sane instructions.

Proves: the MIR→CLIF translation path works at all. Surfaces type
mismatches, missing MIR constructs, layout issues early.

### M2: Call the compiled function

Build a test harness that compiles a function to a shared object, loads it
with `dlopen`/`dlsym`, and calls it. Assert `foo() == 42`.

Proves: the generated code actually runs and produces correct values. ABI
is right (return value in correct register).

### M3: Arithmetic and locals

```rust
fn bar(a: i32, b: i32) -> i32 {
    let x = a + b;
    x * 2
}
```

Compile, dlopen, call with args, check result.

Proves: `CheckedBinaryOp` translation, local variables, argument passing
through the ABI layer. This is where `FnAbi`/`PassMode` gets exercised
for real.

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

### M7: Control flow + function calls

```rust
fn add(a: i32, b: i32) -> i32 {
    if a > 0 { a + b } else { b }
}

fn main() -> ! {
    std::process::exit(add(19, 23));
}
```

Proves: `SwitchInt`, branching, multi-function compilation, intra-crate calls.

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

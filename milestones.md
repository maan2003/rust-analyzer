# Milestones

Each milestone produces something testable. If a milestone's output is wrong,
you know exactly where the bug is before more complexity piles on.

---

## Phase 0: Preparation

### M0: TyAbiInterface compiles

Implement `TyAbiInterface` for r-a's `Ty`, add `ra-ap-rustc_target` dep.
Write a test that computes `FnAbi` for a simple function signature
(`fn(i32, i32) -> i32`) and asserts the `PassMode` for each argument.

Proves: the bridge between r-a's type system and rustc_target's calling
convention machinery works. Everything downstream depends on this.

---

## Phase 1: ra-mir-export

### M1: Round-trip a trivial function

The converter reads `optimized_mir` for a single function from a rustc-compiled
rlib and writes it in our serialized format. A test deserializes it back and
checks that the basic blocks, terminators, and types survived the round-trip.

Proves: the serialization format works, rustc MIR can be read and translated.

### M2: Export std

Run the converter on `core`, `alloc`, `std`. Deserialize and spot-check a few
well-known functions (`Vec::push`, `Option::unwrap`). Doesn't need to be
perfect — just needs to not crash and produce structurally valid MIR.

Proves: the converter handles real-world code at scale.

---

## Phase 2 + 3: Codegen + AOT Driver

### M3: Empty main runs

Compile and run `fn main() {}`.

Proves: `cranelift-object` emits a valid object file, the linker invocation
works, entry point glue (`lang_start` → `main`) is correct.

### M4: Integer arithmetic

```rust
fn main() -> ! {
    let x = 1 + 2;
    let y = x * 10;
    std::process::exit(y as i32);
}
```

Verify by checking exit code = 30.

Proves: `Assign`, integer arithmetic (`BinaryOp`), integer casts, calling an
extern function with arguments, linking against std.

### M5: Control flow + function calls

```rust
fn add(a: i32, b: i32) -> i32 {
    if a > 0 { a + b } else { b }
}

fn main() -> ! {
    std::process::exit(add(19, 23));
}
```

Proves: `SwitchInt`, branching, multi-function compilation, intra-crate
calls, argument passing through the ABI layer.

### M6: Structs and enums

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
pattern matching. Layout correctness matters here — wrong field offsets
produce wrong values silently.

### M7: Generics and monomorphization

```rust
fn pick<T: Copy>(a: T, b: T, first: bool) -> T {
    if first { a } else { b }
}

fn main() -> ! {
    std::process::exit(pick(7, 3, true));
}
```

Proves: generic functions get monomorphized, type substitution in MIR works,
the compiled code uses concrete types.

### M8: Drop and heap allocation

```rust
fn main() -> ! {
    let v = vec![10, 20, 30];
    let code = v[1];
    drop(v);
    std::process::exit(code);
}
```

Proves: calling std generic functions from exported MIR, heap allocation,
indexing, and explicit drop glue invocation for `Vec` (`drop(v)`). First
milestone that heavily exercises the `ra-mir-export` output.

### M9: Trait objects and dynamic dispatch

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

Proves: vtable generation, dynamic dispatch, unsizing coercion, vtable layout
matches rustc's.

### M10: println

```rust
fn main() {
    println!("hello, world!");
}
```

Proves: the full `println!` → `format_args!` → `fmt::Write` → stdout chain.
This exercises a huge surface area of std. When this works, most "normal"
Rust code is within reach.

---

## Phase 4: Expanding Coverage

### M11: A real crate

Pick a small real-world crate (a CLI tool, a parser, something with a few
deps) and compile it. This surfaces every edge case the targeted milestones
missed.

### M12: Correctness suite

Run a subset of rustc's run-pass tests through the new backend. Track pass
rate. This is ongoing — the number goes up over time as gaps are filled.

---

## Phase 5 (future): JIT

### M13: JIT hello world

Same as M10 but via `cranelift-jit` — compile in-memory, run without
producing an object file.

### M14: Incremental recompilation

Change a function body, recompile only that function, re-run. Salsa
identifies the invalidated MIR, Cranelift recompiles it, function pointer
gets swapped.

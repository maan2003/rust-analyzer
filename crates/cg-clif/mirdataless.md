# The Mirdataless Architecture

This document outlines the architectural pivot away from the `.mirdata` offline-export approach toward a unified compilation strategy for `rust-analyzer`'s Cranelift backend (`cg-clif`).

## The Fundamental Flaw of `.mirdata`

The `.mirdata` architecture attempted to solve the problem of compiling the standard library (and complex crates) by using `rustc` to export its optimized MIR offline, which `cg-clif` would then consume.

This approach fails at the boundary of **monomorphization and trait resolution**:

1. **The Generics Trap:** When a user calls a generic standard library function (e.g., `Vec::<MyLocalStruct>::push`), the pre-exported `.mirdata` does not contain the necessary layout or trait resolution information for `MyLocalStruct`.
2. **State Disconnect:** The `rustc` oracle that generated `.mirdata` is blind to the user's workspace. It cannot synthesize the required drop glue, vtables, or generic trait method bodies (`<MyLocalStruct as Clone>::clone`) needed to instantiate the generic template.
3. **Double Backend Maintenance:** Executing `rustc`'s MIR required a parallel Cranelift translation layer (`mirdata_codegen.rs`), duplicating the effort of lowering `rust-analyzer`'s own internal MIR (`hir_ty::mir::MirBody`).

A compiler must own the type layout and trait resolution for the entire call stack simultaneously. Splitting the frontend (user code) and middle-end (std MIR) across a process boundary is an architectural dead end.

## The Three-Tier Strategy

Instead of trying to emulate `rustc`'s MIR or building complex IPC oracles, we partition the compilation pipeline into three tiers.

### Tier 1: User Code & Simple Generics (Compile from r-a MIR)

For any function defined in the user's workspace, or any generic function where r-a's MIR lowering succeeds:

- Lower from source using r-a's MIR lowerer
- Monomorphize with the user's concrete types using r-a's trait solver and layout engine
- Compile via the single Cranelift backend

This maintains an unbroken connection to the type system. If a generic template calls `T::clone()`, the trait solver resolves to the user's impl and compiles it from MIR.

This already works for user-defined generics, closures, and simple std functions.

### Tier 2: Monomorphic External Functions (dlsym / Linker)

For fully concrete, non-generic functions from `std`, `core`, `alloc`, or external dependencies (e.g., `std::fs::File::open`, `std::process::exit`):

- Do not lower them to MIR
- In AOT mode: emit an extern symbol reference, let the linker resolve it against sysroot rlibs (already working via M7.5)
- In JIT mode: look up the pre-compiled machine code via `dlsym` from `libstd.so` or `target/debug/deps/*.so`, emit `call_indirect`

This bypasses lowering failures for complex OS-specific code, has zero compile-time cost, and uses the real battle-tested implementations.

For `#[inline]` monomorphic functions that don't appear in the symbol table, fall back to Tier 1 (compile from source).

### Tier 3: Complex External Generics (Shims)

For external generic functions that r-a's MIR lowering can't handle (complex std internals, heavy macro usage, inline asm, async machinery):

- Provide **shim crates** with simplified implementations that r-a *can* lower
- Shims bridge the gap using two techniques:

#### Technique A: Thin Generic Wrappers

Reimplement the function in simple Rust that r-a handles, delegating heavy lifting to monomorphic helpers.

Example for `Vec::<T>::push`:
```rust
// Thin generic shell — r-a can lower this
fn push<T>(vec: &mut Vec<T>, value: T) {
    if vec.len == vec.cap {
        // monomorphic: just bytes, size, align
        grow_raw(&mut vec.buf, vec.len, vec.cap, size_of::<T>(), align_of::<T>());
    }
    unsafe {
        ptr::write(vec.as_mut_ptr().add(vec.len), value);
    }
    vec.len += 1;
}
```

`grow_raw` is monomorphic (Tier 2 — dlsym'd or compiled separately). The generic part is trivial pointer ops that r-a already handles.

#### Technique B: Trait Erasure

For functions with trait bounds, erase the trait to a vtable-style struct.

A function `foo<T: Clone>(x: T)` becomes `foo<DynClone>(x: DynClone)` where:
```rust
struct DynClone {
    ptr: *mut u8,
    size: usize,
    align: usize,
    clone_fn: fn(*const u8) -> *mut u8,
    drop_fn: fn(*mut u8),
}

impl Clone for DynClone {
    fn clone(&self) -> Self {
        // call through clone_fn
    }
}
```

`foo<DynClone>` is monomorphic — pre-compiled by rustc into a helper `.so`. At the call site, r-a generates the thin wrapper that packs the user's concrete type into `DynClone` and unpacks the result.

**Limits of trait erasure:** When the erased type appears in containers (`Vec<T>`, `[T; N]`) or return types, the layout changes. Each such case needs a per-function shim that handles wrapping/unwrapping at the boundary. This is not an automatic universal transform — it's a per-function escape hatch.

## Coverage Target: Normal Rust Code

The goal is not to handle every possible generic instantiation. It's to cover normal Rust code: `Vec`, `String`, `HashMap`, `Option`/`Result`, iterators, `println!`, `Box`, basic I/O.

Most of this surface is straightforward generic code that r-a should lower from source — `Vec::push` is pointer writes and length updates, `Option::map` is a match and a call. The complexity lives at the bottom of the call stack, not the top.

### In Practice: Very Few Shims

The std abstractions share a common core. `Vec`, `String`, `VecDeque`, `HashMap` all ultimately go through `RawVec`/`alloc::alloc`. Shim the allocator growth path once and you've covered all growable collections. The actual shim list is likely:

- **Allocator growth** (`RawVec::grow` / the underlying `alloc::realloc` path) — covers all growable collections
- **`fmt` machinery** — the `write!`/`format_args!` internal plumbing
- **`HashMap` hashing internals** — maybe

Everything above that (`Vec::push`, `String::from`, `HashMap::insert`) is generic code r-a compiles from source.

## Shim Ecosystem

Shims are not just for std. Popular crates have stable public APIs that change infrequently:

- **std shims** ship with the project (small — a handful of deep internal functions)
- **Ecosystem shims** for popular crates (tokio, serde, etc.) can be maintained separately
- **Crate authors** may eventually contribute their own shims

As r-a's MIR lowering improves over time, shims become unnecessary and get deleted. The shim layer is a bridge, not a permanent fixture.

## User Escape Hatch

Users are willing to restructure code for sub-second hot reload. When something doesn't compile through r-a:

1. The system provides a clear error pointing at the failing function
2. The user moves the problematic code behind a monomorphic function boundary in a separate crate
3. `rustc` compiles that crate normally, and r-a calls into it via Tier 2

This is the same model as `wasm-bindgen` or FFI boundaries — you constrain the interface for mechanical reasons, and developers accept it because the payoff (hot reload) is worth it.

"rac-compatible" becomes a property of code: stick to normal generics, keep exotic trait machinery behind crate boundaries.

## Symbol Resolution

To call into pre-compiled `.so`/`.rlib` artifacts, we need the mangled symbol name.

**For AOT mode:** v0 symbol mangling is already implemented and working (M4, M7.5). The linker resolves symbols against sysroot rlibs.

**For JIT mode:**
1. Parse symbol tables of `.rlib`/`.so` artifacts in the sysroot and `target/debug/deps/` (using `object` or `goblin`)
2. Map demangled paths to mangled strings
3. Resolve addresses via `dlsym`, emit `call_indirect`

## Call Resolution Pipeline

When `cg-clif` encounters a `TerminatorKind::Call`:

1. **Is it local?** Compile from r-a MIR.
2. **Is it a known shim?** Use the shim implementation (compiled from r-a MIR).
3. **Is it monomorphic external?** Emit extern symbol / dlsym.
4. **Is it generic external?**
   - Try lowering from source via r-a MIR (Tier 1).
   - If lowering fails and a shim exists, use the shim (Tier 3).
   - If no shim: report a clear error identifying the failing function.
5. **User fixes it** by moving code behind a monomorphic boundary.

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

See `crates/cg-clif/NOTES.md` for known bugs and upstream divergences.

---

## Phase 1.5: Codegen completeness

Before linking against std, fill out codegen for the MIR constructs we'll
encounter. These can all be tested via JIT without a linker.

### M3.1: CValue/CPlace abstraction ✅

Ported `value_and_place.rs`: CValue (ByRef | ByVal | ByValPair),
CPlace (Var | VarPair | Addr). Non-scalar locals use stack slots.
ScalarPair locals (tuples like `(i32, i32)`) stored as VarPair,
wired for params and returns. Memory-to-memory copy via
`emit_small_memory_copy`.

### M3.2: Casts (`Rvalue::Cast`) ✅

IntToInt, FloatToInt, IntToFloat, FloatToFloat, PtrToPtr,
PointerExposeProvenance, PointerWithExposedProvenance, Transmute
all working for scalar types.

### M3.3: Aggregates ✅

Tuple and ADT aggregates working (Scalar, ScalarPair, and memory-repr
fast paths). Field projections via `CPlace::place_field`. Deref
projection, Downcast projection (Direct tag encoding), Discriminant
rvalue (Direct tag), Ref/AddressOf, Len for fixed-size arrays.

ADT support: struct/enum aggregate construction, field type resolution
with generic substitution (`db.field_types` + `instantiate`),
`SetDiscriminant` statement (Direct tag encoding), and enum variant
constructor calls (`CallableDefId::StructId`/`EnumVariantId`).
Multi-variant enums on register places (Var/VarPair) spill to memory
for correct field offset handling.

Niche-encoded discriminants (read + write) ported from upstream.
Still missing: closure field types, explicit discriminant values (`A = 100`).

### M3.4: Remaining scalar ops ✅

- Overflow binops (`AddWithOverflow`/`SubWithOverflow`/`MulWithOverflow`)
  implemented via `codegen_checked_int_binop` returning ScalarPair.
- Float `Rem` via fmod/fmodf libcalls.
- Pointer `Offset` size scaling fixed.

---

## Phase 2: End-to-end `fn main() {}`

### M4: Symbol mangling ✅

Implemented v0 symbol mangling (RFC 2603) in `crates/cg-clif/src/symbol_mangling.rs`.
Ported encoding primitives from `rustc_symbol_mangling/src/v0.rs`. Handles
crate paths, module paths, impl paths, trait paths, extern blocks, and
type encoding (all scalar types, refs, raw ptrs, slices, arrays, tuples, ADTs).
Generic instantiations wrapped with `I...E`. All codegen now emits mangled
names; 4 mangling tests + all existing JIT/object tests pass with v0 names.

Deferred: backref caching, const generic encoding (placeholder `p`), dyn Trait,
fn pointers, punycode, trait impl paths (`X`), closures. Crate disambiguator
uses FileId index for local crates; for external (sysroot) crates, real
disambiguators are extracted from rlib archive symbol tables (see M7.5).

Proves: symbols will resolve at link time.

### M5: Empty main runs ✅

Compiled `fn main() {}` → `.o` → linked against std → runs as executable.
Added `emit_entry_point()` (C-ABI `main` wrapper that calls user's Rust main
and returns 0, skipping `lang_start`), `link.rs` (shells out to `cc` with all
sysroot rlibs from `rustc --print target-libdir`), and `compile_executable()`
orchestrating the full pipeline. Integration test `compile_and_run_empty_main`
verifies the binary exits with code 0.

Proves: cranelift-object emits valid object code, linker invocation works,
entry point glue is correct, v0 mangling produces the right symbol name.

### M6: Exit code ✅

```rust
extern "C" {
    fn exit(code: i32) -> !;
}
fn main() -> ! {
    unsafe { exit(42) }
}
```

Calls libc `exit` directly via `extern "C"` (no std). Added
`build_fn_sig_from_ty()` to build Cranelift signatures from type info
(`callable_item_signature`) for extern functions (no MIR available).
`codegen_direct_call` detects extern functions via `ExternBlockId`
container, uses raw symbol names (no v0 mangling), and inserts a trap
after diverging calls (`-> !`). Integration test
`compile_and_run_exit_code` verifies exit code 42.

Proves: extern function detection, type-based signature building,
raw symbol names, never type handling, linking against libc.

---

## Phase 3: Codegen breadth (current)

### M7: Multi-function executables ✅

`SwitchInt`, branching, intra-crate direct calls, and extern `"C"` calls
all work for scalar types (tested via JIT in M3, extern calls in M6).
Added `collect_reachable_fns()` BFS to discover all callees from main,
compile them all into one object module. Integration test
`compile_and_run_multi_fn` verifies multi-function binary.

### M7.5: Cross-crate calls into std ✅

```rust
fn main() -> ! {
    std::process::exit(42)
}
```

Calls `std::process::exit` through v0-mangled symbol resolved from the
real std rlib at link time. Required:

- **Crate disambiguator extraction** (`link.rs`): loads `StableCrateId`
  values from `.mirdata` files produced by `ra-mir-export`, a rustc
  driver that extracts `StableCrateId` from sysroot crates. Maps crate
  names to disambiguator values for v0 symbol mangling.
- **Cross-crate call detection** (`lib.rs`): `codegen_direct_call` checks
  `func_id.krate(db) != local_crate`; cross-crate calls use type-based
  signatures (`build_fn_sig_from_ty`) and v0 mangled names with real
  disambiguators. `collect_reachable_fns` skips external functions.
- **Allocator shim** (`lib.rs`): emits `__rust_alloc`, `__rust_dealloc`,
  `__rust_realloc`, `__rust_alloc_zeroed`, and
  `__rust_no_alloc_shim_is_unstable_v2` as wrappers around libc
  malloc/free/realloc/calloc. Uses `__rustc` crate disambiguator
  extracted from std's symbol table. Will be removed when switching to
  `rustc` as linker driver.
- **Linker hardening**: `--start-group`/`--end-group` for circular rlib
  deps, `--gc-sections` to trim unused std code, filter out `rustc-dev`
  symlinks (NixOS).
- **ScalarPair support** in `build_fn_sig_from_ty` for functions that
  take/return pairs (slices, wide pointers).

Proves: v0 mangling produces symbols that match real rlibs, cross-crate
function detection works, type-based signatures are correct.

### M8: Structs and enums ✅

```rust
struct Point { x: i32, y: i32 }
enum Dir { Left, Right }

fn main() -> ! {
    let p = Point { x: 3, y: 4 };
    let code = match Dir::Right {
        Dir::Left => p.x,
        Dir::Right => p.y,
    };
    std::process::exit(code)
}
```

ADT codegen (M3.3) already handled struct construction, field projection,
enum discriminant read/write, and layout. This milestone confirms it all
works end-to-end as a compiled+linked executable calling into std.
Integration test `compile_and_run_structs_and_enums` exits with code 4.

### M9: Generics and monomorphization ✅

```rust
fn pick<T>(a: T, b: T, first: bool) -> T {
    if first { a } else { b }
}

fn main() -> ! {
    std::process::exit(pick(7, 3, true))
}
```

`collect_reachable_fns()` now tracks `(FunctionId, StoredGenericArgs)` pairs
instead of bare `FunctionId`, so each monomorphized instance is discovered
and compiled separately. `compile_executable()` passes the correct generic
args to `monomorphized_mir_body()` and `mangle_function()`. The codegen
call path (`codegen_direct_call`) already handled generic args correctly.
Integration test `compile_and_run_generics` exits with code 7.

### M10: Trait objects and dynamic dispatch ✅

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

Implemented vtable generation, dynamic dispatch via indirect calls, and
unsizing coercion (`&T → &dyn Trait`). Key components:

- **Vtable construction** (`get_or_create_vtable`): builds vtable data
  constants with the standard layout (drop_in_place null, size, align,
  method fn ptrs). Uses `TraitImpls::for_crate` + `simplify_type` to
  find impl methods by name. Vtable data is declared as immutable,
  methods emitted as function address relocations.
- **Unsizing coercion** (`codegen_unsize_coercion`): handles
  `CastKind::PointerCoercion(Unsize)` by producing a fat pointer
  `(data_ptr, vtable_ptr)` as `CValue::by_val_pair`.
- **Virtual dispatch** (`codegen_virtual_call`): detects dyn calls via
  `is_dyn_method`, loads fn ptr from vtable at computed offset,
  emits `call_indirect` with thin self pointer.
- **Fat pointer deref**: `ProjectionElem::Deref` now handles ScalarPair
  pointers by extracting the data pointer. `Rvalue::Ref` re-borrows
  recover metadata via `extract_place_metadata`.
- **Reachability** (`collect_reachable_fns`): scans for unsizing casts
  to discover vtable impl methods; skips abstract trait method defs
  and virtual call targets.

JIT tests (`jit_dyn_dispatch`, `jit_dyn_dispatch_multiple_methods`) and
end-to-end test (`compile_and_run_dyn_dispatch`) all pass. 60 total tests.

### M11a: PassMode::Indirect ✅

Memory-repr types (structs with 3+ fields, etc.) now passed and returned
by pointer, matching upstream's `PassMode::Indirect` pattern:

- `build_fn_sig` / `build_fn_sig_from_ty`: Memory-repr returns add
  `AbiParam::special(StructReturn)` as first param (no return values);
  Memory-repr params add `AbiParam::new(pointer_ty)`.
- `compile_fn`: detects sret, overrides return slot CPlace to point at
  sret block param; indirect params get CPlace pointing at incoming ptr.
- `codegen_direct_call` / `codegen_virtual_call`: Memory-repr args use
  `force_stack()` + pass pointer; sret returns allocate a stack slot,
  pass pointer as first arg, read result back after call.
- Return terminator: Memory-repr returns emit `return_(&[])` since the
  value is already written to the sret pointer.

3 JIT tests: `jit_pass_and_return_big_struct`, `jit_pass_big_struct_and_modify`,
`jit_big_struct_through_call_chain`. 55 total tests pass.

### M11b: Array indexing ✅

`ProjectionElem::Index` with dynamic offset via `Pointer::offset_value`.
Arrays spill to stack when in registers. `place_ty` updated for Index.
JIT test: `arr[1usize]` returns 20.

### M11c: Essential intrinsics ✅

Implemented 30+ intrinsics in `codegen_intrinsic_call`:
- Size/alignment: `size_of`, `min_align_of`/`pref_align_of`, `size_of_val`,
  `min_align_of_val`, `needs_drop`
- Memory: `copy_nonoverlapping` (memcpy), `copy` (memmove), `write_bytes`
  (memset), `volatile_load`, `volatile_store`
- Hints: `assume`, `likely`/`unlikely`, `black_box`, `assert_inhabited`/
  `assert_zero_valid`/`assert_mem_uninitialized_valid` (no-ops)
- Bit manipulation: `ctlz`/`cttz`/`ctpop`, `bswap`, `bitreverse`,
  `rotate_left`/`rotate_right`
- Arithmetic: `wrapping_add`/`sub`/`mul`, `unchecked_add`/`sub`/`mul`/
  `shl`/`shr`/`div`/`rem`, `exact_div`
- Other: `transmute` (force-to-stack + reinterpret), `abort` (trap),
  `ptr_mask`, atomic fences

8 JIT tests verify `size_of`, `min_align_of`, `bswap`, `wrapping_add`,
`transmute`, `ctlz`, `rotate_left`, `exact_div`. 67 total tests pass.

### M11d: Non-scalar constants ✅

ScalarPair constants extract two scalar values at correct offsets.
Memory-repr constants (arrays, large structs) stored as anonymous data
sections via `DataDescription` with raw bytes. `MemoryMap::Empty` only
(no embedded pointer relocations). JIT test: array constant with index.

### M11e: Fn pointers and indirect calls ✅

`ReifyFnPointer` cast converts `FnDef` to fn pointer via `func_addr`.
`TyKind::FnPtr` calls use `import_signature` + `call_indirect`.
`codegen_call` now uses `operand_ty` to handle any fn operand (not just
constants). `collect_reachable_fns` scans for `ReifyFnPointer` casts.
JIT tests: `jit_fn_pointer_call`, `jit_fn_pointer_higher_order`.

### M11f: Drop glue

```rust
struct Guard { val: i32 }
impl Drop for Guard {
    fn drop(&mut self) { /* side effect */ }
}
fn foo() -> i32 {
    let g = Guard { val: 42 };
    g.val
}  // drop called on scope exit
```

Needs: `TerminatorKind::Drop` to call drop functions instead of being a
no-op jump. Requires generating drop glue shims or resolving `Drop::drop`
impl methods.

### M11g: Closures ✅

Closure construction already handled via `AggregateKind::Closure` (same
path as tuples). New pieces:

- **`closure_field_type`**: resolves capture types via
  `InferenceResult::for_body` + `closure_info` + `CapturedItem::ty`.
- **Closure call dispatch**: `codegen_direct_call` detects when
  `Fn::call`/`FnMut::call_mut`/`FnOnce::call_once` has a concrete
  closure self type, redirects to `codegen_closure_call` which gets
  the closure's MIR body via `monomorphized_mir_body_for_closure`.
- **Closure discovery**: `collect_reachable_fns` scans for
  `AggregateKind::Closure` in statements, compiles closure bodies
  alongside regular functions. Recursively scans closure bodies for
  nested callees/closures.
- **Symbol mangling**: `mangle_closure` uses simple scheme
  `_Rclosure_{crate}_{disamb}_{id}`.
- **`place_ref` fix**: register-stored places (Var/VarPair) now spill
  to stack before taking address — needed for `&self` on closures.
- **`jit_run_reachable`**: new test helper using `collect_reachable_fns`
  for automatic function/closure discovery.

No hacks — closure body params `[closure_self, arg0, arg1, ...]` match
the Call terminator args directly, so no argument restructuring needed.

JIT test: `jit_closure_basic` — `apply(|x| x + offset, 32)` returns 42.
89 total tests pass.

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

## Phase 4.5

Optimize for salsa, rework reachability analysis, maybe use a backchannel during mir lowering to tell us about reachable functions. this is somewhat similar to how salsa handles diagnostics for example. to be explored

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

### M19: Hot-reload JIT

Live-patching running programs. Editing a function body recompiles exactly
that one function and patches it into the running process.

#### Salsa caching strategy

Key queries for incremental codegen:

- `compiled_fn_object(fn_id, generic_args, env) -> Vec<u8>` — per-function
  compiled machine code. The main cache: avoids re-running Cranelift
  lowering/regalloc when MIR hasn't changed.
- `direct_callees(fn_id) -> Vec<FunctionId>` — extracted from MIR call
  terminators. Splitting this from reachability avoids O(N) re-walks when
  one function's MIR changes but its call targets didn't.
- `reachable_functions(root) -> Vec<FunctionId>` — transitive closure of
  `direct_callees`. Backdated when the call graph is unchanged.
- `fn_abi_cranelift(fn_id, env) -> Signature` — Cranelift calling convention
  for a function. Separating this from `compiled_fn_object` means callers
  don't recompile when a callee's body changes (only when its signature does).

#### Firewalls in the dependency graph

Salsa "backdates" queries whose output didn't change despite re-execution,
preventing unnecessary downstream invalidation:

- `item_tree` — structural (declarations, not bodies). Body-only edits
  re-execute but produce the same result, firewalling all other functions
  in the same file.
- `crate_def_map` / `resolve_path` — name resolution. Adding a file
  re-executes the def map, but existing resolutions resolve to the same
  IDs and get backdated.
- `direct_callees` — if a function's MIR changed but it still calls the
  same set of functions, reachability is backdated.

#### Propagation for common edit patterns

- **Body-only edit**: O(1) recompile. `item_tree` backdated, only the
  edited function's `mir_body` and `compiled_fn_object` re-execute.
- **Signature change**: O(callers) recompile. `fn_abi_cranelift` changes,
  all callers' MIR re-lowers (new arg count), their `compiled_fn_object`
  re-executes. Cross-crate callers included.
- **New file**: O(new functions) compile. Existing name resolutions
  backdated, no existing code recompiled.
- **Type layout change**: O(users of that type) recompile. Unavoidable —
  machine code genuinely changes (different stack offsets, field accesses).

#### Auto-boxing user types (dev-mode optimization)

All structs defined in user code (not rlibs) are heap-allocated and passed
as pointers between user functions. This means:

- Function ABIs for user code are always pointer-based — changing a struct's
  fields never changes any caller's compiled code (pointer is still 8 bytes).
- Only functions that directly access fields of a changed struct recompile.
- Layout changes are fully firewalled except at rlib call boundaries.
- At user→rlib boundaries (e.g. `Vec::push(my_struct)`), thin shims
  unbox/box values to match the rlib's expected ABI. These shims recompile
  on layout change, but callers of the shims don't.
- Small scalar types (integers, floats, scalar-pairs) can be exempted —
  they fit in registers and boxing them wastes a heap allocation for no
  incremental benefit.
- Runtime cost (heap allocs + indirection) is acceptable for dev-mode;
  release builds use normal by-value layouts.

#### Indirect call table for hot patching

All intra-user-code calls go through an indirection table (like a PLT/GOT):
`call [fn_table + fn_index]`. Recompiling a function only requires updating
one pointer in the table. Callers don't recompile even though the machine
code address changed. Combined with salsa, editing one function body =
recompile one function + write one pointer.

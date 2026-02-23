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
  (add/sub/mul/div/rem via `fmod`/`fmodf` libcalls) and comparisons.
  Pointer offset and comparisons (offset now scales by pointee size).
  Unchecked variants handled. Overflow variants (`AddWithOverflow` etc.)
  implemented via `codegen_checked_int_binop` returning ScalarPair.
- `Rvalue::UnaryOp` — `Neg` (ineg/fneg), `Not` (bnot for ints, icmp_imm for bool)
- `Rvalue::Cast` for scalar casts: int<->int, int<->float, float<->float,
  ptr<->ptr/int (`PtrToPtr`, provenance casts, `FnPtrToPtr`), scalar transmute
- `TerminatorKind::SwitchInt` — multi-way branching via `cranelift_frontend::Switch`
- `TerminatorKind::Call` — direct function calls (FnDef with scalar and
  ScalarPair args/returns). Resolves callee via TyKind::FnDef → CallableDefId,
  gets callee MIR, builds Cranelift sig, declares in module, emits call.
  ZST args filtered. ScalarPair args passed as two values.
  Tested with call chains and calls combined with branches.
  Includes direct lowering of pointer intrinsics used by raw-pointer methods:
  `offset`, `arith_offset`, `ptr_offset_from`, `ptr_offset_from_unsigned`.
  Essential intrinsics: `size_of`, `min_align_of`/`pref_align_of`,
  `copy_nonoverlapping`/`copy`/`write_bytes` (memcpy/memmove/memset),
  `assume`/`likely`/`unlikely` (no-ops/passthrough), `needs_drop`,
  `black_box`, `transmute`, `ctlz`/`cttz`/`ctpop`, `bswap`/`bitreverse`,
  `rotate_left`/`rotate_right`, `exact_div`, `wrapping_add`/`sub`/`mul`,
  `unchecked_add`/`sub`/`mul`/`shl`/`shr`/`div`/`rem`, `abort`, `ptr_mask`,
  `volatile_load`/`volatile_store`, atomic fences.
  Extern `"C"` function calls supported: detects `ExternBlockId` container,
  builds signature from type info (`callable_item_signature`) instead of MIR,
  uses raw (unmangled) symbol name. Diverging calls (`-> !`) terminate the
  block with a trap instruction.
- `TerminatorKind::Drop` — no-op jump (scalar types only, no drop glue yet)
- **CValue/CPlace abstractions** — `value_and_place.rs` ported with
  CValue (ByRef | ByVal | ByValPair) and CPlace (Var | VarPair |
  Addr(Pointer, Option<Value>)). The `Option<Value>` in Addr carries
  unsized metadata (e.g. vtable pointer for `dyn Trait`), matching
  upstream. `place_ref()` produces fat or thin pointer CValues
  depending on metadata presence. All locals use CPlace, operands
  flow through CValue.
- **ScalarPair locals** — tuples like `(i32, i32)` stored as VarPair,
  properly wired for function params and returns.
- **Non-scalar locals** — Memory-repr types allocated as stack slots.
- **PassMode::Indirect** — Memory-repr types (3+ field structs, etc.) passed
  and returned by pointer. Return uses `AbiParam::special(StructReturn)` as
  first param (sret); params use `AbiParam::new(pointer_ty)`. Callee's return
  slot CPlace points at the sret pointer; callee writes result there and
  returns void. Caller allocates stack slot, passes pointer, reads result
  back after call. `force_stack()` used for indirect args. Works for direct
  calls, virtual calls, and cross-function chains.
- **Tuple aggregates** — `Rvalue::Aggregate(Tuple)` constructs tuples,
  with fast paths for Scalar and ScalarPair representations.
- **ADT aggregates** — `Rvalue::Aggregate(Adt)` constructs structs and enums.
  Scalar/ScalarPair fast paths for single-variant ADTs. Multi-variant enums
  spill to memory temp for correct field offsets, then read back. ADT field
  type resolution uses `db.field_types` + `instantiate` for generic substitution.
- **ADT constructor calls** — `CallableDefId::StructId`/`EnumVariantId` handled
  as inline aggregate construction (not real function calls).
- **SetDiscriminant** — writes enum tag via `place_field(tag_field)` (unified
  for Var/VarPair/Addr). Supports Direct and Niche tag encodings.
- **Field projections** — `ProjectionElem::Field` on tuples and ADTs via
  `CPlace::place_field`. VarPair splits into individual Vars for
  ScalarPair fields; Addr offsets by field layout offset.
- **Rvalue::Ref/AddressOf** — takes address of a place.
- **Rvalue::Discriminant** — reads enum discriminant. Supports Direct and
  Niche tag encodings. Handles Scalar, ScalarPair, and memory-repr places.
- **Rvalue::Len** — returns fixed-size array length as a constant.
- **Trait objects / dynamic dispatch** — `PointerCoercion(Unsize)` cast
  produces fat pointer `(data_ptr, vtable_ptr)`. Vtable construction
  builds data constants with standard layout (drop null, size, align,
  method fn ptrs) using `TraitImpls::for_crate` + `simplify_type` from
  `rustc_type_ir` to find impl methods. Virtual calls detected via
  `is_dyn_method`, dispatched through `call_indirect` with fn ptr loaded
  from vtable. Fat pointer deref via `CPlace::for_ptr_with_extra`
  carries metadata (vtable ptr) in `CPlaceInner::Addr`; `place_ref()`
  recovers it for re-borrows (matches upstream `place_deref`/`place_ref`).
  Reachability scans for unsizing casts to discover vtable impl methods;
  skips abstract trait defs.
- **Fn pointers and indirect calls** — `ReifyFnPointer` cast converts
  `FnDef` types to fn pointers via `func_addr`. `TyKind::FnPtr` calls
  use `call_indirect` with signature built from the `FnPtr` type.
  Reachability scans for `ReifyFnPointer` casts to discover fn targets.
- **Closures** — Closure construction via `AggregateKind::Closure` (same
  code path as tuples). `ClosureField` projection resolves capture types
  via `InferenceResult::for_body` + `closure_info`. Closure calls detected
  when `Fn::call`/`FnMut::call_mut`/`FnOnce::call_once` has a concrete
  closure self type — redirects to the closure's own MIR body via
  `monomorphized_mir_body_for_closure`. Simple mangling scheme
  (`_Rclosure_{crate}_{disamb}_{id}`). `collect_reachable_fns` discovers
  closures by scanning for `AggregateKind::Closure` in statements;
  recursively scans closure bodies for nested callees/closures.
- **Non-scalar constants** — ScalarPair constants (two-element tuples,
  wide pointers). Memory-repr constants (arrays, large structs) stored
  as anonymous data sections via `DataDescription`. `MemoryMap::Empty`
  only (no embedded pointer relocations yet).

## Upstream comparison (`./cg_clif`)

Recently aligned with upstream:
- **Integer cast skeleton matches upstream** — our `codegen_intcast`
  follows the same `equal -> extend -> reduce` pattern as upstream's
  `clif_intcast` in `cg_clif/src/cast.rs`.
- **Float `Rem` lowering matches upstream for `f32`/`f64`** — both use
  libcalls (`fmodf`/`fmod`) from Cranelift codegen.
- **Pointer `Offset` scaling now matches upstream** — we now multiply the
  offset by `pointee_size` before adding to the base pointer.
- **Pointer distance intrinsics match upstream behavior** — we now lower
  `ptr_offset_from` and `ptr_offset_from_unsigned` as pointer subtraction
  divided by pointee size, consistent with `cg_clif/src/intrinsics/mod.rs`.
- **ADT aggregate construction matches upstream pattern** — both downcast
  to variant layout, write fields, then set discriminant. Upstream does this
  in `base.rs:875-898` with `lval.downcast_variant()` + `place_field()` +
  `codegen_set_discriminant()`. We follow the same structure.
- **SetDiscriminant matches upstream** — both use `place.place_field(tag_field)`
  to access the tag field uniformly (`discriminant.rs:31,56`). Our version
  handles Var/VarPair/Addr via `place_field` without manual backend_repr
  dispatch. Both Direct and Niche tag encodings implemented.
- **GetDiscriminant matches upstream** — both read the tag and decode it.
  Direct encoding uses intcast (`discriminant.rs:126-134`). Niche encoding
  uses relative_tag/is_niche/select pattern (`discriminant.rs:135-216`).
  `codegen_icmp_imm` ported from `common.rs:105-147` for I128 support.
- **CPlaceInner::Addr carries metadata like upstream** — our `Addr(Pointer,
  Option<Value>)` matches upstream's `Addr(Pointer, Option<Value>)` in
  `value_and_place.rs:366`. Fat pointer deref uses `for_ptr_with_extra` to
  carry metadata, `place_ref()` emits thin or fat pointer based on metadata
  presence. Matches upstream's `place_deref` (`value_and_place.rs:815-823`)
  and `place_ref` (`value_and_place.rs:825-836`).
- **Intrinsics match upstream patterns** — `volatile_load` uses
  `CValue::by_ref(Pointer::new(ptr), layout)` matching upstream
  (`intrinsics/mod.rs:804`). `volatile_store` uses
  `CPlace::for_ptr(Pointer::new(ptr), layout)` + `write_cvalue` matching
  upstream (`intrinsics/mod.rs:815-816`). Bit manipulation (`ctlz`/`cttz`/
  `ctpop`/`bswap`/`bitreverse`), rotates, `copy`/`copy_nonoverlapping`/
  `write_bytes` (memcpy/memmove/memset with `elem_size != 1` guard),
  `ptr_mask`, `abort` (trap user(2)), and `black_box` (passthrough) all
  follow upstream's exact lowering.

Still diverges from upstream:
- **Pointer coercion casts partially handled** — `PointerCoercion::Unsize`
  (`&T → &dyn Trait`) now produces fat pointers with vtable.
  `ReifyFnPointer` emits `func_addr` to convert fn items to fn pointers.
  Other coercions (`UnsafeFnPointer`, `ClosureFnPointer`) still treated
  as scalar casts. `MutToConstPointer`/`ArrayToPointer` are no-ops (correct).
- **Wide-pointer cast behavior is partial** — trait object fat pointers work.
  Upstream has additional wide<->wide and wide->thin cast paths for slices
  and trait upcasting that we don't yet implement.
- **Cast edge semantics are still simplified** — upstream handles i128
  conversion libcalls and float->int nuances (NaN behavior / saturating cast
  details) that we have not fully ported.
- **VarPair multi-variant enums spill to memory** — upstream's
  `place_field()` handles VarPair field projections natively after
  `downcast_variant()` because its layout system computes field indices
  within the overall ScalarPair. Our simpler `place_field()` maps
  `field_idx` directly to VarPair elements, which is wrong after
  downcast (variant field 0 ≠ VarPair element 0 when a tag occupies
  element 0). We work around this by spilling Var/VarPair places to a
  stack slot before downcast/field access on multi-variant enums, then
  reading back. Correct but adds a round-trip through memory.
- **ADT constructor calls handled differently** — upstream treats
  `CallableDefId::StructId`/`EnumVariantId` as normal function calls
  through its full ABI machinery. We handle them as inline aggregate
  construction since we don't have the full call ABI for constructors.
- **Intrinsic divergences from upstream** —
  - `size_of`/`min_align_of`/`pref_align_of`: upstream uses fallback MIR
    bodies (not explicitly handled). We emit `iconst` directly — correct
    and simpler, since we lack fallback MIR.
  - `size_of_val`/`min_align_of_val`: upstream handles unsized types via
    `size_and_align_of(fx, layout, meta)`. We only handle sized types
    (return static layout size/align). Needs extending for `dyn Trait`
    and `[T]`.
  - `needs_drop`: upstream uses fallback MIR. We hardcode `false`.
    Needs actual type-level check.
  - `assert_inhabited`/`assert_zero_valid`/`assert_mem_uninitialized_valid`:
    upstream checks validity and can emit panics. We no-op.
  - `exact_div`: upstream uses `codegen_int_binop(fx, BinOp::Div, x, y)`
    on CValues (signedness inferred from layout). We manually check
    `ty_is_signed_int` and emit `sdiv`/`udiv`. Same result.
  - `wrapping_add`/`sub`/`mul`: upstream uses fallback MIR. We emit
    `iadd`/`isub`/`imul` directly. Same semantics.
  - `write_bytes`: upstream gets elem size from pointee deref
    (`dst.layout().ty.builtin_deref`). We use generic arg layout.
    Both resolve to the same `T`.
- **Explicit discriminant values not supported** — upstream uses
  `layout.ty.discriminant_for_variant(fx.tcx, variant_index)` to get
  the actual discriminant value (e.g. `A = 100` → 100). We use the
  variant index directly, so `A = 100` would incorrectly write 0.
  Needs `db.const_eval_discriminant()`.
Known bugs (divergence from upstream cg_clif):
- **Constants lack relocation/pointer support** — Scalar, ScalarPair, and
  memory-repr constants all work for plain values (integers, floats, arrays of
  scalars). Missing: pointer constants (references to allocations/statics),
  slice constants (`&str`, `&[T]`), i128. The memory-repr path creates a data
  section from raw bytes but writes no relocations (`MemoryMap::Empty` only).
  ScalarPair path uses `iconst` for both halves — wrong for float pairs.
  Upstream dispatches on `ConstValue::Scalar`/`Indirect`/`Slice` and handles
  pointer relocations via `define_all_allocs` + `data.write_data_addr()`.
- **No `PassMode::Uniform` / `PassMode::Cast`** — we handle Scalar,
  ScalarPair (direct), and Memory-repr (indirect/sret), but not the
  `Uniform` or `Cast` pass modes used on some targets for small aggregates.
- **`Variants::Single` discriminant uses variant index** — for enums with
  explicit discriminant values (e.g. `A = 100`) the codegen returns the
  variant index (0) instead of the discriminant value (100). Needs
  `db.const_eval_discriminant()` lookup.
- **JIT helper `jit_run` compiles only explicitly listed roots** — calls from
  a tested function into other local functions not listed in
  `jit_run(..., fn_names, ...)` remain unresolved at runtime. The newer
  `jit_run_reachable` helper uses `collect_reachable_fns` to automatically
  discover and compile all reachable functions and closures.
- **Array index tests require explicit `usize` index** — `arr[1]` fails MIR
  lowering with `UnresolvedMethod("[overloaded index]")`. The MIR lowerer only
  emits `ProjectionElem::Index` when the index is already `usize`; otherwise it
  falls through to `Index` trait dispatch. Even with minicore `index` feature,
  r-a's method resolution can't resolve `Index<I> for [T; N] where I:
  SliceIndex<[T]>` (the `SliceIndex` bound chain fails). Using `arr[1usize]`
  sidesteps both issues. Real code with full std would work via trait resolution.

What's missing:
- Explicit discriminant values (`A = 100`) — need `db.const_eval_discriminant()`
- Union and RawPtr aggregates
- Remaining casts (`DynStar`, wide-pointer coercions)
- Drop glue
- Cross-crate generic function codegen (MIR bodies now available via .mirdata)

## MIR export from sysroot (M13/M14)

`ra-mir-export` (rustc driver) now exports optimized MIR for sysroot functions
to `.mirdata` files. Shared types live in `crates/ra-mir-types/`.

- **8111 function bodies** exported (2382 generic, 5729 monomorphic/#[inline])
- Covers all 20 sysroot crates (core, alloc, std, etc.)
- Translation handles: all statement/terminator/rvalue/operand/type variants
  that appear in optimized MIR. Unsupported constructs fall back to
  `Ty::Opaque` / `ConstKind::Todo` with debug strings.
- `cg-clif` deserializes `MirData` for crate disambiguators and converts
  the layout table (`layout.rs`: `ExportedXxx` → `rustc_abi` types) so
  pre-computed layouts can be used directly by codegen. `compile_fn` and
  `build_fn_sig` accept a `mirdata_layouts` parameter (currently `&[]` for
  r-a MIR; will be used when compiling mirdata bodies).

### .mirdata format

`postcard(MirData { crates, bodies, layouts })`

Each `FnBody` carries a `DefPathHash` (StableCrateId + local hash) for
stable cross-crate identity, the human-readable path, generic param count,
and a full `Body` (locals, arg_count, basic blocks with statements and
terminators).

Each `Local` has an optional `layout: Option<u32>` index into
`MirData.layouts`. The layout table (`Vec<TypeLayoutEntry>`) stores
deduplicated `LayoutInfo` per concrete type: size, align, backend_repr
(Scalar/ScalarPair/Memory), field offsets, variant info (enum tag encoding),
and largest niche. Computed via `tcx.layout_of()` in ra-mir-export.
Locals with generic type params get `layout: None`.

### Regenerating

```
cd ra-mir-export && cargo run --release -- -o /tmp/sysroot.mirdata
```

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

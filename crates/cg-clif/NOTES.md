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

Still diverges from upstream:
- **Pointer coercion casts partially handled** — `PointerCoercion::Unsize`
  (`&T → &dyn Trait`) now produces fat pointers with vtable. Other coercions
  (`ReifyFnPointer`, `UnsafeFnPointer`, `ClosureFnPointer`) still treated
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
- **Explicit discriminant values not supported** — upstream uses
  `layout.ty.discriminant_for_variant(fx.tcx, variant_index)` to get
  the actual discriminant value (e.g. `A = 100` → 100). We use the
  variant index directly, so `A = 100` would incorrectly write 0.
  Needs `db.const_eval_discriminant()`.
Known bugs (divergence from upstream cg_clif):
- **Constants only handle small scalars** — `const_to_i64` extracts raw bytes
  into i64. Missing: pointer constants (references to allocations/statics),
  slice constants (fat pointers), i128 (upstream uses `iconcat(lsb, msb)`),
  indirect constants (stored in allocations). String literals, `const &[T]`,
  etc. won't work.
- **No `PassMode` / ABI handling in calls** — we pass Scalar and ScalarPair
  args/returns directly and skip ZSTs, but have no `PassMode::Indirect`
  (pass-by-pointer for large structs) or `PassMode::Uniform`. Struct
  args/returns that aren't Scalar/ScalarPair will produce wrong ABI.
- **`Variants::Single` discriminant uses variant index** — for enums with
  explicit discriminant values (e.g. `A = 100`) the codegen returns the
  variant index (0) instead of the discriminant value (100). Needs
  `db.const_eval_discriminant()` lookup.
- **JIT helper compiles only explicitly listed roots** — calls from a tested
  function into other local functions not listed in `jit_run(..., fn_names, ...)`
  remain unresolved at runtime (`can't resolve symbol ...`). This especially
  affects source-level minicore method calls (e.g. raw-pointer inherent methods)
  unless tests use direct intrinsic calls or object-only compile checks.

What's missing:
- Closure field type resolution
- Explicit discriminant values (`A = 100`) — need `db.const_eval_discriminant()`
- Union and RawPtr aggregates
- Indirect calls (fn pointers, closures)
- Remaining casts (`DynStar`, wide-pointer coercions)
- Intrinsics beyond pointer offset/distance
- Drop glue

Next steps (easiest to port):
1. Explicit discriminant values via `db.const_eval_discriminant()`
2. Closure field type resolution
3. Remaining cast/intrinsic/drop-glue coverage

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

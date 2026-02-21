# Facts Discovered About rust-analyzer and rustc Internals

## r-a's MIR Representation

- MIR data structures live in `crates/hir-ty/src/mir.rs`
- `MirBody` contains `basic_blocks: Arena<BasicBlock>`, `locals: Arena<Local>`, `projection_store: ProjectionStore`, `start_block`, `param_locals`
- `BasicBlock` has `statements: Vec<Statement>`, `terminator: Option<Terminator>`, `is_cleanup: bool`

### Terminators (14 variants)
- `Goto`, `SwitchInt`, `UnwindResume`, `Abort`, `Return`, `Unreachable`
- `Drop { place, target, unwind: Option<BasicBlockId> }`
- `DropAndReplace { place, value, target, unwind }`
- `Call { func, args, destination, target, cleanup, from_hir_call }`
- `Assert { cond, expected, target, cleanup }`
- `Yield`, `CoroutineDrop`, `FalseEdge`, `FalseUnwind`

### Rvalues
- `Use`, `Repeat`, `Ref`, `Len`, `Cast`, `CheckedBinaryOp`, `UnaryOp`, `Discriminant`, `Aggregate`, `ShallowInitBox`, `ShallowInitBoxWithAlloc`, `CopyForDeref`
- `BinaryOp` is an `Infallible` stub — all binary ops go through `CheckedBinaryOp`
- `NullaryOp` is an `Infallible` stub
- `AddressOf` is an `Infallible` stub
- `ThreadLocalRef` is an `Infallible` stub

### Statements (6 variants)
- `Assign`, `FakeRead`, `Deinit`, `StorageLive`, `StorageDead`, `Nop`
- `SetDiscriminant` is commented out at `mir.rs:1020`

### Place Projections (7 variants, 1 disabled)
- `Deref`, `Field(Either<FieldId, TupleFieldId>)`, `ClosureField(usize)`, `Index(LocalId)`, `ConstantIndex`, `Subslice`, `OpaqueCast`
- `Downcast` is commented out at `mir.rs:153`

### Operands
- `Copy(Place)`, `Move(Place)`, `Constant { konst, ty }`, `Static(StaticId)`
- Wrapped in `Operand` struct with `kind: OperandKind` and `span: Option<MirSpan>`

### BinOp enum
- `Add`, `Sub`, `Mul`, `Div`, `Rem`, `BitXor`, `BitAnd`, `BitOr`, `Shl`, `Shr`, `Eq`, `Lt`, `Le`, `Ne`, `Ge`, `Gt`, `Offset`

## r-a's MIR Lowering Gaps

- `async`/`await` → `not_supported!("await")` at `lower.rs:956-958`
- `yield` / async blocks → `not_supported!`
- `become` (tail calls) → `not_supported!("tail-calls")`
- `yeet` → `not_supported!("yeet")`
- `const` blocks → `not_supported!("const block")`
- `builtin#offset_of` → not supported
- `builtin#asm` (inline asm) → not supported
- 56 `NotSupported`/`not_supported!` cases in the const eval interpreter (`eval.rs`) in this checkout

## r-a's Salsa Integration

- `mir_body(def: DefWithBodyId)` — per-function generic MIR, Salsa-cached
- `monomorphized_mir_body(def, subst: StoredGenericArgs, env: StoredParamEnvAndCrate)` — monomorphized MIR, Salsa-cached per (function, generic args, param env)
- `mir_body_for_closure` and `monomorphized_mir_body_for_closure` — separate queries for closures
- MIR is lowered once in generic form, then monomorphized on demand by `Filler` which clones and substitutes type parameters
- Monomorphization at `crates/hir-ty/src/mir/monomorphization.rs:225`
- Body-only edits do NOT invalidate `crate_local_def_map` — the `ItemTree` is stable over function body changes
- Type inference results cached per function: `InferenceResult::for_body()` is `#[salsa::tracked]`
- `body_with_source_map()` has LRU(512)
- `borrowck()` has LRU(2024)
- Library file texts use `Durability::HIGH`, local workspace file texts use `Durability::LOW`
- Library source roots use `Durability::MEDIUM`, local source roots use `Durability::LOW`

## r-a's Shared rustc Crates

r-a already depends on (version 0.143, `ra-ap-*` mirrors):

| Crate | Usage |
|-------|-------|
| `ra-ap-rustc_type_ir` | `TyKind`, `GenericArgs`, type folding/visiting — r-a's `Ty` IS `rustc_type_ir::TyKind<DbInterner>` |
| `ra-ap-rustc_abi` | `LayoutCalculator`, `TargetDataLayout`, `Scalar`, `Primitive`, struct/enum layout, `TyAbiInterface` trait |
| `ra-ap-rustc_next_trait_solver` | Trait resolution — r-a implements `SolverDelegate` and uses `GoalEvaluation`, `SolverDelegateEvalExt` |
| `ra-ap-rustc_ast_ir` | Basic AST IR types (Mutability, etc.) |
| `ra-ap-rustc_index` | `IndexVec`, arena types |
| `ra-ap-rustc_pattern_analysis` | Pattern exhaustiveness checking |
| `ra-ap-rustc_lexer` | Lexing |
| `ra-ap-rustc_parse_format` | Format string parsing |

r-a does NOT currently depend on:
- `rustc_target` (calling conventions, `FnAbi`, `PassMode`)
- `rustc_middle` (TyCtxt, Instance, rustc's Body)
- `rustc_codegen_ssa`
- `rustc_session`

## r-a's Layout Computation

- Uses `rustc_abi::LayoutCalculator` directly — NOT a reimplementation
- `LayoutCx` wraps `LayoutCalculator<&TargetDataLayout>` at `crates/hir-ty/src/layout.rs:120`
- `TargetDataLayout::parse_from_llvm_datalayout_string` used at `crates/hir-ty/src/layout/target.rs:15`
- Struct layouts: `cx.calc.univariant()`
- Array layouts: `cx.calc.array_like()`
- Union layouts: `cx.calc.layout_of_union()` at `layout/adt.rs:75`
- Enum layouts: `cx.calc.layout_of_struct_or_enum()` at `layout/adt.rs:77`
- SIMD layouts: `cx.calc.simd_type()` at `layout.rs:157`
- Primitive scalar layouts include `Primitive::Int`, `Primitive::Float`, and `Primitive::Pointer` (for refs/raw pointers/fat pointers)
- Layouts match rustc's when r-a and rustc use the same layout code revision and target data layout

## r-a's Type System

- `Ty<'db>` wraps `WithCachedTypeInfo<TyKind<'db>>` where `TyKind<'db> = rustc_type_ir::TyKind<DbInterner<'db>>`
- `DbInterner<'db>` implements `rustc_type_ir::Interner` with 50+ associated types
- `GenericArgKind<'db> = rustc_type_ir::GenericArgKind<DbInterner<'db>>`
- Full `TypeVisitable`, `TypeFoldable`, `TypeFolder`, `FallibleTypeFolder`, `Relate` implementations
- Interner implementation at `crates/hir-ty/src/next_solver/interner.rs` (~2717 lines)

## r-a's FnAbi Situation

- r-a has a local `FnAbi` enum with 35+ variants (Rust, C, Win64, SysV64, etc.) at `crates/hir-ty/src/lib.rs:200`
- It only **parses** ABI from attributes — does NOT compute argument passing
- `PartialEq for FnAbi` is stubbed out (always returns true) with a FIXME
- No `PassMode`, `ArgAbi`, or `CastTarget` usage anywhere in r-a
- No parameter passing or register allocation implemented

## r-a's Sysroot/std Handling

- r-a cannot process `.rlib` files — needs source code
- Source installed via `rustup component add rust-src`
- r-a parses and type-checks `core`, `alloc`, and `std` from source, and lowers MIR for supported constructs
- Comment at `crates/project-model/src/sysroot.rs:4`: "we can't process .rlib and need source code instead"
- Sysroot crates are tracked in `LibraryRoots`; their source roots are `Durability::MEDIUM` while file texts are `Durability::HIGH`

## TyAbiInterface Trait (in rustc_abi)

- Defined at `rustc_abi/src/layout/ty.rs:158`
- Generic: `pub trait TyAbiInterface<'a, C>: Sized + Debug + Display`
- 10 required methods:
  - `ty_and_layout_for_variant`
  - `ty_and_layout_field`
  - `ty_and_layout_pointee_info_at`
  - `is_adt`, `is_never`, `is_tuple`, `is_unit`, `is_transparent`
  - `is_scalable_vector`
  - `is_pass_indirectly_in_non_rustic_abis_flag_set`
- r-a does NOT implement this trait currently
- All `rustc_target::callconv` functions are generic over `Ty: TyAbiInterface` — they don't depend on `TyCtxt`
- `adjust_for_rust_abi` and `adjust_for_foreign_abi` only need `HasDataLayout + HasTargetSpec`

## rustc's FnAbi Computation

- Main logic in `rustc_ty_utils/src/abi.rs`, function `fn_abi_new_uncached` at line 487
- Steps: get calling convention → compute argument layouts → classify into PassMode → adjust by target ABI
- Default PassMode based on `backend_repr`: ZST→Ignore, Scalar→Direct, ScalarPair→Pair, Memory→Indirect
- Target-specific adjustments in `rustc_target/src/callconv/{x86_64,aarch64,riscv,...}.rs`
- Each target has `compute_abi_info(cx, fn_abi)` that modifies PassMode in place
- Required traits: `HasDataLayout`, `HasTargetSpec`, `HasX86AbiOpt`, `TyAbiInterface`

## rustc's MIR in rlibs

- MIR serialization at `rustc_metadata/src/rmeta/encoder.rs`
- `should_encode_mir` at line 1096 determines what's included
- MIR is included for: const items (always), generic functions, `#[inline]` functions, cross-crate inlinable functions
- NOT included by default for: non-generic non-inline functions
- `-Z always-encode-mir` flag forces MIR encoding for ALL functions
- `optimized_mir` table stores `LazyValue<mir::Body<'static>>`
- `mir_for_ctfe` table stores CTFE-specific MIR
- Deserialization requires `TyCtxt` — cannot be read independently
- Monomorphized MIR is never stored — monomorphization is always on-the-fly (comment at `rustc_middle/src/ty/instance.rs:27`)

## cg_clif (rustc_codegen_cranelift)

- Lives in `compiler/rustc_codegen_cranelift` in the rustc repo
- Dependencies: `cranelift-codegen`, `cranelift-frontend`, `cranelift-module`, `cranelift-native`, `cranelift-jit` (optional), `cranelift-object`
- JIT mode behind `feature = "jit"`, requires `cranelift-jit` + `libloading`
- Uses `rustc_private` — links against all of `rustc_middle`, `rustc_codegen_ssa`, `rustc_session`, `rustc_target`, etc.

### cg_clif Code Structure

| File | Lines | Purpose |
|------|-------|---------|
| `base.rs` | ~950 | Core MIR→CLIF translation: `codegen_fn`, statement/terminator dispatch |
| `value_and_place.rs` | ~1000 | `CValue`/`CPlace` — wraps Cranelift Values/StackSlots with layout info |
| `abi/mod.rs` | ~945 | Calling conventions, call codegen, drop codegen |
| `abi/pass_mode.rs` | ~335 | `ArgAbi` → Cranelift `AbiParam` translation, `PassMode` handling |
| `abi/returning.rs` | small | Return value ABI |
| `num.rs` | ~400 | Arithmetic operations |
| `cast.rs` | ~200 | Type casts |
| `discriminant.rs` | ~100 | Enum discriminant read/write |
| `vtable.rs` | ~150 | Vtable layout generation |
| `unsize.rs` | ~200 | Unsizing coercions |
| `intrinsics/mod.rs` | ~2500 | Intrinsic implementations |
| `constant.rs` | variable | Constant value codegen |
| `driver/jit.rs` | ~215 | JIT execution driver |
| `driver/aot.rs` | variable | AOT compilation driver |

### cg_clif's CValue/CPlace

- `CValue<'tcx>(CValueInner, TyAndLayout<'tcx>)` — read-only value
- `CValueInner`: `ByRef(Pointer, Option<Value>)`, `ByVal(Value)`, `ByValPair(Value, Value)`
- `CPlace { inner: CPlaceInner, layout: TyAndLayout<'tcx> }` — writable location
- `CPlaceInner`: `Var(Local, Variable)`, `VarPair(Local, Variable, Variable)`, `Addr(Pointer, Option<Value>)`
- Mostly Cranelift API operations — type info only used for layout decisions

### cg_clif's JIT Driver

- `jit.rs:19-33`: Creates `JITModule` via `JITBuilder::with_isa`
- `jit.rs:25`: Symbol lookup via `dep_symbol_lookup_fn` — dlopens dependency dylibs
- `jit.rs:169-215`: `dep_symbol_lookup_fn` — iterates `CrateInfo.used_crates`, loads dylibs with `libloading`, returns closure that looks up symbols by name
- JIT mode requires all dependencies as dylibs (no static libs)
- Produces no object files — compiles and runs in-process

## Structural Differences: r-a MIR vs rustc MIR

### Terminators
- r-a uses `Option<BasicBlockId>` for unwind; rustc uses `UnwindAction` enum (Terminate, Cleanup, Continue, Unreachable)
- r-a has `DropAndReplace` as separate variant; rustc uses `Drop` with `replace: bool`
- r-a has `Abort`; rustc has `UnwindTerminate(reason)`
- rustc has `TailCall`, `InlineAsm` — r-a has neither

### Rvalues
- r-a stubs: `BinaryOp(Infallible)`, `NullaryOp(Infallible)`, `AddressOf(Infallible)`, `ThreadLocalRef(Infallible)`
- rustc has: `RawPtr(RawPtrKind, Place)` (r-a uses Ref + cast instead), `WrapUnsafeBinder`
- r-a `Aggregate` uses `Box<[Operand]>`; rustc uses `IndexVec<FieldIdx, Operand>`

### Statements
- rustc has 13 variants; r-a has 6
- r-a missing: `SetDiscriminant` (commented out), `Retag`, `PlaceMention`, `AscribeUserType` (commented out), `Coverage`, `Intrinsic` (commented out), `ConstEvalCounter`, `BackwardIncompatibleDropHint`

### Projections
- r-a `Field` uses `Either<FieldId, TupleFieldId>`; rustc uses `FieldIdx` (u32)
- r-a has `ClosureField(usize)` — rustc handles via generic `Field`
- r-a `Downcast` is commented out; rustc has it as a full variant
- r-a `ConstantIndex` lacks `min_length` field
- r-a `Subslice` lacks `from_end` field

### Type System
- Both use `rustc_type_ir::TyKind` but with different interners
- r-a: `TyKind<DbInterner<'db>>` with Salsa-based interning
- rustc: `TyKind<TyCtxt<'tcx>>` with arena-based interning
- Both support `TypeFoldable`, `TypeVisitable`

## r-a's Const Eval Interpreter (for reference, not directly relevant)

- At `crates/hir-ty/src/mir/eval.rs`
- Has simulated stack (`Vec<u8>`) and heap (`Vec<u8>`)
- Execution limit: 10M instructions in non-test builds (100k in tests), stack depth limit: 100, memory limit: 1GB
- Supports: function calls (with MIR), closures, trait dispatch, heap allocation, custom Drop
- Shims for ~10 extern "C" functions (memcmp, write, getenv, pthread_*)
- Shims for ~30 intrinsics (size_of, copy, atomic ops, float math)
- Cannot call external crate functions without MIR or shims
- Supports stdout/stderr capture via `write(fd=1/2, ...)`

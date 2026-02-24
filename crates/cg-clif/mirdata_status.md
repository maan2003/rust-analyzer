# mirdata JIT test harness — status & next steps

## What works

The seamless mirdata JIT harness (`jit_run_with_std`) still works for basic
cross-crate generic calls:

- `mirdata_jit_identity_i32` passes (`core::convert::identity(42)`).
- `mirdata_jit_ptr_write_read_i32` passes.
- `mirdata_jit_mem_replace_i32` passes.

Harness flow:
1. Generate `.mirdata` via `ra-mir-export`.
2. Load real sysroot sources (`core`, `alloc`, `std`) into r-a TestDB.
3. Compile local user code from r-a MIR (`collect_reachable_fns`).
4. Compile cross-crate generic callees from mirdata.
5. Resolve cross-crate non-generic callees through `dlopen` + `libstd.so`.

## Recently fixed

The old blocker in this file is fixed:

- Cross-crate generic associated fns (impl methods) no longer fail lookup as
  `alloc::vec::new`.
- `fn_display_path` now includes impl self-type names when available
  (`alloc::vec::Vec::new`).
- mirdata name matching now strips generic segments (`::<...>` / `<...>`) before lookup.
- Candidate selection also filters by expected `StableCrateId` before generic-count matching.
- Lookup now reports ambiguity explicitly instead of silently picking a wrong body.

This unblocked resolution of `Vec::new`/`Vec::push` bodies and moved failure to
the next real issue.

## Current blocker

`mirdata_jit_vec_push_len` now fails later during codegen with missing layout:

```text
mirdata layout not found for type:
Adt((core_stable_crate_id, ...), "std::marker::PhantomData", [Ty(Int(I32))])
```

This is a layout-availability problem for monomorphized generic mirdata bodies,
not a function-body lookup problem.

### Why this happens

- Generic mirdata bodies often have locals with `layout = None` in the generic template.
- After substitution (`T = i32`), those locals need concrete layouts.
- We currently rely on the exported global layout table (`MirData.layouts`) to
  already contain every needed concrete instantiation.
- For `Vec::<i32>::new` that assumption is false.

## Next step (real fix)

Need one of:

1. Export all required monomorphized layouts for generic mirdata bodies that we
   might instantiate at runtime (from `ra-mir-export`), or
2. Compute missing layouts on demand during mirdata codegen from semantic type
   info (not just from pre-exported table).

Without one of these, complex generic std code paths (like `Vec`) will keep
failing after body lookup succeeds.

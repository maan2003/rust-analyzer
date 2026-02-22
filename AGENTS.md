read goal.md for high level goal

plan.md contains the high level plan

cg_clif/ is the original rustc cranelift backend (git subtree). Read-only reference.

crates/cg-clif/ is our MIR->Cranelift codegen crate using r-a types. This is where active development happens. See crates/cg-clif/NOTES.md for architecture notes and porting status.

rustc/ is the rust tree, but not a subtree. it is read only.

cargo .. -p .. is slow, avoid it.

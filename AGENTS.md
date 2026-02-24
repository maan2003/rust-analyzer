read goal.md for high level goal

plan.md contains the high level plan
milestones.md contains the current status

cg_clif/ is the original rustc cranelift backend (git subtree). Read-only reference.
Always reference the upstream cg_clif/ when working on our codegen.

crates/cg-clif/ is our MIR->Cranelift codegen crate using r-a types. This is where active development happens. See crates/cg-clif/NOTES.md for architecture notes and porting status.

rustc/ is the rust tree, but not a subtree. it is read only.

cargo .. -p .. is slow, avoid it.

AVOID HACKS, just say it to the user

You are in a "vibe coding" setting, user doesn't review the code, so you have to be very clear when
you are doing a hack or something that will bite us in future.

This is a new project, backward compatibility is never needed.

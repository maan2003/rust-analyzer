//! ra-mir-export: Extract sysroot crate info and MIR bodies from rustc.
//!
//! A rustc driver that compiles a dummy file, loads sysroot crates, and
//! extracts their StableCrateIds plus optimized MIR for generic/#[inline]
//! functions. Output is a postcard-serialized `MirData` written to a
//! `.mirdata` file.
//!
//! Usage:
//!   ra-mir-export -o sysroot.mirdata [rustc flags...] [input.rs]

#![feature(rustc_private, box_patterns)]

extern crate rustc_abi;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

mod translate;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use rustc_driver::Compilation;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::ty::TyCtxt;

use rustc_middle::ty;

use ra_mir_types::{CrateInfo, FnBody, MirData, TypeLayoutEntry};

struct ExportCallbacks {
    result: RefCell<Option<MirData>>,
    output_path: PathBuf,
}

impl rustc_driver::Callbacks for ExportCallbacks {
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        let crates = extract_crate_ids(tcx);
        let (bodies, layouts) = extract_mir_bodies(tcx);
        let mir_data = MirData { crates, bodies, layouts };

        // Write the mirdata file *inside* the callback, before rustc's delayed
        // ICE handler can kill the process.
        let data = postcard::to_allocvec(&mir_data).expect("postcard serialize");
        std::fs::write(&self.output_path, &data)
            .unwrap_or_else(|e| panic!("failed to write {}: {e}", self.output_path.display()));

        *self.result.borrow_mut() = Some(mir_data);
        Compilation::Stop
    }
}

fn extract_crate_ids(tcx: TyCtxt<'_>) -> Vec<CrateInfo> {
    let mut result = vec![];

    // Local crate
    let local_name = tcx.crate_name(LOCAL_CRATE);
    let local_id = tcx.stable_crate_id(LOCAL_CRATE);
    result.push(CrateInfo {
        name: local_name.to_string(),
        stable_crate_id: local_id.as_u64(),
    });

    // All extern crates loaded by this compilation
    for &cnum in tcx.crates(()) {
        let name = tcx.crate_name(cnum);
        let id = tcx.stable_crate_id(cnum);
        result.push(CrateInfo {
            name: name.to_string(),
            stable_crate_id: id.as_u64(),
        });
    }

    result
}

// ---------------------------------------------------------------------------
// MIR body extraction
// ---------------------------------------------------------------------------

fn extract_mir_bodies(tcx: TyCtxt<'_>) -> (Vec<FnBody>, Vec<TypeLayoutEntry>) {
    let mut bodies = Vec::new();
    let mut stats = ExportStats::default();
    let mut visited = HashSet::new();
    let mut layout_table = LayoutTable::new();

    for &cnum in tcx.crates(()) {
        let crate_def_id = cnum.as_def_id();
        visit_module(tcx, crate_def_id, &mut bodies, &mut stats, &mut visited, &mut layout_table, 0);
    }

    // Print stats
    eprintln!(
        "MIR export: {} functions found, {} with MIR available, {} translated, {} skipped",
        stats.total_fns, stats.mir_available, stats.translated, stats.skipped
    );

    (bodies, layout_table.entries)
}

#[derive(Default)]
struct ExportStats {
    total_fns: usize,
    mir_available: usize,
    translated: usize,
    skipped: usize,
}

struct LayoutTable<'tcx> {
    entries: Vec<TypeLayoutEntry>,
    dedup: HashMap<ty::Ty<'tcx>, u32>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> LayoutTable<'tcx> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            dedup: HashMap::new(),
            typing_env: ty::TypingEnv::fully_monomorphized(),
        }
    }

    fn get_or_insert(&mut self, tcx: TyCtxt<'tcx>, ty: ty::Ty<'tcx>) -> Option<u32> {
        if let Some(&idx) = self.dedup.get(&ty) {
            return Some(idx);
        }
        let ty_and_layout = tcx.layout_of(self.typing_env.as_query_input(ty)).ok()?;
        let exported_ty = translate::translate_ty(tcx, ty);
        let layout_info = translate::translate_layout_data(&ty_and_layout.layout);
        let idx = self.entries.len() as u32;
        self.entries.push(TypeLayoutEntry {
            ty: exported_ty,
            layout: layout_info,
        });
        self.dedup.insert(ty, idx);
        Some(idx)
    }
}

/// Max module nesting depth to prevent stack overflow from re-exports.
const MAX_MODULE_DEPTH: usize = 20;

fn visit_module<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    out: &mut Vec<FnBody>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
    depth: usize,
) {
    if depth > MAX_MODULE_DEPTH {
        return;
    }
    if !visited.insert(def_id) {
        return;
    }
    for child in tcx.module_children(def_id) {
        let Some(child_def_id) = child.res.opt_def_id() else {
            continue;
        };
        // Skip items not from an extern crate
        if child_def_id.is_local() {
            continue;
        }
        match tcx.def_kind(child_def_id) {
            DefKind::Mod => {
                visit_module(tcx, child_def_id, out, stats, visited, layout_table, depth + 1);
            }
            DefKind::Fn | DefKind::AssocFn => {
                try_export_fn(tcx, child_def_id, out, stats, visited, layout_table);
            }
            DefKind::Impl { .. } => {
                visit_impl_or_trait(tcx, child_def_id, out, stats, visited, layout_table);
            }
            DefKind::Trait => {
                visit_impl_or_trait(tcx, child_def_id, out, stats, visited, layout_table);
            }
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                // Discover inherent impl methods (e.g. Vec::push, Vec::new)
                for impl_def_id in tcx.inherent_impls(child_def_id) {
                    visit_impl_or_trait(tcx, *impl_def_id, out, stats, visited, layout_table);
                }
            }
            _ => {}
        }
    }
}

fn visit_impl_or_trait<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    out: &mut Vec<FnBody>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
) {
    if !visited.insert(def_id) {
        return;
    }
    for &item in tcx.associated_item_def_ids(def_id) {
        try_export_fn(tcx, item, out, stats, visited, layout_table);
    }
}

fn try_export_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    out: &mut Vec<FnBody>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
) {
    if !visited.insert(def_id) {
        return;
    }
    stats.total_fns += 1;

    if !tcx.is_mir_available(def_id) {
        return;
    }
    stats.mir_available += 1;

    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let body = tcx.optimized_mir(def_id);
        let translated = translate::translate_body(tcx, body);
        let hash = translate::def_path_hash(tcx, def_id);
        let name = tcx.def_path_str(def_id);
        let num_generic_params = tcx.generics_of(def_id).count();

        FnBody {
            def_path_hash: hash,
            name,
            num_generic_params,
            body: translated,
        }
    })) {
        Ok(mut fn_body) => {
            // Compute layouts for each local
            let body = tcx.optimized_mir(def_id);
            for (i, decl) in body.local_decls.iter().enumerate() {
                fn_body.body.locals[i].layout = layout_table.get_or_insert(tcx, decl.ty);
            }
            stats.translated += 1;
            out.push(fn_body);
        }
        Err(_) => {
            stats.skipped += 1;
            let name = tcx.def_path_str(def_id);
            eprintln!("  warning: skipped {name} (translation panicked)");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    // Parse our own flags before passing the rest to rustc.
    let args: Vec<String> = std::env::args().collect();

    let mut output_path = None;
    let mut filtered_args = Vec::new();
    let mut skip_next = false;

    for (i, arg) in args.iter().enumerate() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if arg == "-o" || arg == "--output" {
            if let Some(path) = args.get(i + 1) {
                output_path = Some(PathBuf::from(path));
                skip_next = true;
                continue;
            }
        }
        filtered_args.push(arg.clone());
    }

    let output_path = output_path.unwrap_or_else(|| {
        eprintln!("Usage: ra-mir-export -o <output.mirdata> [rustc flags...] <input.rs>");
        std::process::exit(1);
    });

    // If no source file provided, create a temporary dummy file
    let has_source = filtered_args.iter().skip(1).any(|a| !a.starts_with('-'));
    let _tmpfile; // keep alive
    if !has_source {
        let tmp = std::env::temp_dir().join("ra_mir_export_dummy.rs");
        std::fs::write(&tmp, "fn main() {}\n").expect("write dummy.rs");
        filtered_args.push(tmp.to_str().unwrap().to_string());
        _tmpfile = tmp;
    }

    let mut callbacks = ExportCallbacks {
        result: RefCell::new(None),
        output_path: output_path.clone(),
    };

    // run_compiler may panic due to delayed ICEs in rustc, but our data is
    // already serialized & written inside the after_analysis callback.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rustc_driver::run_compiler(&filtered_args, &mut callbacks);
    }));

    // Write output (may have already been written in the callback)
    let mir_data = callbacks.result.into_inner().unwrap_or_else(|| {
        eprintln!("error: no data extracted from compilation");
        std::process::exit(1);
    });

    eprintln!(
        "Wrote {} crate infos + {} function bodies + {} type layouts to {}",
        mir_data.crates.len(),
        mir_data.bodies.len(),
        mir_data.layouts.len(),
        output_path.display()
    );
    for info in &mir_data.crates {
        eprintln!("  {} = 0x{:016x}", info.name, info.stable_crate_id);
    }

    // Print some body stats
    let generic_count = mir_data.bodies.iter().filter(|b| b.num_generic_params > 0).count();
    let mono_count = mir_data.bodies.len() - generic_count;
    eprintln!("  {} generic functions, {} monomorphic (#[inline])", generic_count, mono_count);

    // Print layout stats
    let locals_with_layout = mir_data.bodies.iter()
        .flat_map(|b| &b.body.locals)
        .filter(|l| l.layout.is_some())
        .count();
    let locals_without_layout = mir_data.bodies.iter()
        .flat_map(|b| &b.body.locals)
        .filter(|l| l.layout.is_none())
        .count();
    eprintln!("  {} locals with layout, {} without", locals_with_layout, locals_without_layout);

    use ra_mir_types::ExportedBackendRepr;
    let scalar_count = mir_data.layouts.iter()
        .filter(|l| matches!(l.layout.backend_repr, ExportedBackendRepr::Scalar(_)))
        .count();
    let pair_count = mir_data.layouts.iter()
        .filter(|l| matches!(l.layout.backend_repr, ExportedBackendRepr::ScalarPair(_, _)))
        .count();
    let memory_count = mir_data.layouts.iter()
        .filter(|l| matches!(l.layout.backend_repr, ExportedBackendRepr::Memory { .. }))
        .count();
    eprintln!("  Layouts: {} Scalar, {} ScalarPair, {} Memory", scalar_count, pair_count, memory_count);

    let data = postcard::to_allocvec(&mir_data).expect("postcard serialize");
    std::fs::write(&output_path, &data)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", output_path.display()));
}

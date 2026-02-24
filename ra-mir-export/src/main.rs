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

use rustc_middle::{mir, ty};
use rustc_middle::ty::TypeVisitableExt;

use ra_mir_types::{
    AdtDefEntry, CrateInfo, FnBody, GenericFnLookupEntry, GenericFnLookupKey, MirData,
    TypeLayoutEntry, normalize_def_path,
};

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
        let (bodies, layouts, generic_fn_lookup, adt_defs) = extract_mir_bodies(tcx);
        let mir_data = MirData { crates, bodies, layouts, generic_fn_lookup, adt_defs };

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

fn extract_mir_bodies(tcx: TyCtxt<'_>) -> (Vec<FnBody>, Vec<TypeLayoutEntry>, Vec<GenericFnLookupEntry>, Vec<AdtDefEntry>) {
    let mut bodies = Vec::new();
    let mut generic_fn_lookup = Vec::new();
    let mut stats = ExportStats::default();
    let mut visited = HashSet::new();
    let mut layout_table = LayoutTable::new();
    let mut adt_table = AdtTable::new();
    let mut exported_def_ids = Vec::new();

    for &cnum in tcx.crates(()) {
        let crate_def_id = cnum.as_def_id();
        visit_module(
            tcx,
            crate_def_id,
            &mut bodies,
            &mut generic_fn_lookup,
            &mut stats,
            &mut visited,
            &mut layout_table,
            &mut exported_def_ids,
            0,
        );
    }

    // Export methods from trait impls (e.g. Drop::drop, Clone::clone)
    // which aren't visible as module children.
    // Skip impls with const generic parameters to avoid rustc ICEs.
    for &cnum in tcx.crates(()) {
        for &impl_def_id in tcx.trait_impls_in_crate(cnum) {
            let generics = tcx.generics_of(impl_def_id);
            let has_const_param = generics.own_params.iter().any(|p| {
                matches!(p.kind, rustc_middle::ty::GenericParamDefKind::Const { .. })
            });
            if has_const_param {
                continue;
            }
            visit_impl_or_trait(
                tcx,
                impl_def_id,
                &mut bodies,
                &mut generic_fn_lookup,
                &mut stats,
                &mut visited,
                &mut layout_table,
                &mut exported_def_ids,
            );
        }
    }

    // Export methods on primitive types (i32, str, bool, etc.) which use
    // "incoherent impls" and aren't discoverable via module_children traversal.
    {
        use rustc_middle::ty::{IntTy, UintTy, FloatTy};
        use rustc_middle::ty::fast_reject::SimplifiedType;
        let primitive_types = [
            SimplifiedType::Bool,
            SimplifiedType::Char,
            SimplifiedType::Str,
            SimplifiedType::Int(IntTy::Isize),
            SimplifiedType::Int(IntTy::I8),
            SimplifiedType::Int(IntTy::I16),
            SimplifiedType::Int(IntTy::I32),
            SimplifiedType::Int(IntTy::I64),
            SimplifiedType::Int(IntTy::I128),
            SimplifiedType::Uint(UintTy::Usize),
            SimplifiedType::Uint(UintTy::U8),
            SimplifiedType::Uint(UintTy::U16),
            SimplifiedType::Uint(UintTy::U32),
            SimplifiedType::Uint(UintTy::U64),
            SimplifiedType::Uint(UintTy::U128),
            SimplifiedType::Float(FloatTy::F32),
            SimplifiedType::Float(FloatTy::F64),
        ];
        for simp in &primitive_types {
            for &impl_def_id in tcx.incoherent_impls(*simp) {
                visit_impl_or_trait(
                    tcx,
                    impl_def_id,
                    &mut bodies,
                    &mut generic_fn_lookup,
                    &mut stats,
                    &mut visited,
                    &mut layout_table,
                    &mut exported_def_ids,
                );
            }
        }
    }

    // Collect layouts for types that appear in monomorphized generic function instances.
    // When e.g. a monomorphic function calls Vec::<i32>::push, the monomorphized body
    // contains types like RawVec<i32, Global> that need layouts for codegen.
    let layouts_before = layout_table.entries.len();
    // Pass 1: recursively walk ALL exported function locals (monomorphic ones)
    // to export layouts for field types, pointees, etc. that codegen needs during
    // place projections but aren't direct locals.
    {
        let typing_env = ty::TypingEnv::fully_monomorphized();
        let mut visited_types: HashSet<ty::Ty<'_>> = HashSet::new();
        for &def_id in &exported_def_ids {
            if tcx.generics_of(def_id).count() > 0 {
                continue;
            }
            let body = tcx.optimized_mir(def_id);
            for local_decl in body.local_decls.iter() {
                if local_decl.ty.has_param() {
                    continue;
                }
                let normalized = tcx.try_normalize_erasing_regions(typing_env, local_decl.ty)
                    .unwrap_or(local_decl.ty);
                export_type_layout_recursive(tcx, normalized, &mut layout_table, &mut adt_table, &mut visited_types);
            }
        }
    }
    let mono_local_layouts = layout_table.entries.len() - layouts_before;

    // Pass 2: collect layouts for monomorphized generic function instances
    let mono_layouts_before = layout_table.entries.len();
    collect_mono_layouts(tcx, &exported_def_ids, &mut layout_table, &mut adt_table);
    let mono_inst_layouts = layout_table.entries.len() - mono_layouts_before;

    // Print stats
    eprintln!(
        "MIR export: {} functions found, {} with MIR available, {} translated, {} skipped",
        stats.total_fns, stats.mir_available, stats.translated, stats.skipped
    );
    eprintln!("  Layout passes: +{} from local type walk, +{} from mono instances",
        mono_local_layouts, mono_inst_layouts);
    eprintln!("  ADT defs exported: {}", adt_table.entries.len());

    (bodies, layout_table.entries, generic_fn_lookup, adt_table.entries)
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

struct AdtTable<'tcx> {
    entries: Vec<AdtDefEntry>,
    dedup: HashSet<DefId>,
    _marker: std::marker::PhantomData<&'tcx ()>,
}

impl<'tcx> AdtTable<'tcx> {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            dedup: HashSet::new(),
            _marker: std::marker::PhantomData,
        }
    }

    fn get_or_insert(&mut self, tcx: TyCtxt<'tcx>, adt_def: ty::AdtDef<'tcx>) {
        let def_id = adt_def.did();
        if !self.dedup.insert(def_id) {
            return;
        }
        let entry = translate::translate_adt_def(tcx, adt_def);
        self.entries.push(entry);
    }
}

/// Max module nesting depth to prevent stack overflow from re-exports.
const MAX_MODULE_DEPTH: usize = 20;

fn visit_module<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    out: &mut Vec<FnBody>,
    generic_fn_lookup: &mut Vec<GenericFnLookupEntry>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
    exported_def_ids: &mut Vec<DefId>,
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
                visit_module(
                    tcx,
                    child_def_id,
                    out,
                    generic_fn_lookup,
                    stats,
                    visited,
                    layout_table,
                    exported_def_ids,
                    depth + 1,
                );
            }
            DefKind::Fn | DefKind::AssocFn => {
                try_export_fn(
                    tcx,
                    child_def_id,
                    out,
                    generic_fn_lookup,
                    stats,
                    visited,
                    layout_table,
                    exported_def_ids,
                );
            }
            DefKind::Impl { .. } => {
                visit_impl_or_trait(
                    tcx,
                    child_def_id,
                    out,
                    generic_fn_lookup,
                    stats,
                    visited,
                    layout_table,
                    exported_def_ids,
                );
            }
            DefKind::Trait => {
                visit_impl_or_trait(
                    tcx,
                    child_def_id,
                    out,
                    generic_fn_lookup,
                    stats,
                    visited,
                    layout_table,
                    exported_def_ids,
                );
            }
            DefKind::Struct | DefKind::Enum | DefKind::Union => {
                // Discover inherent impl methods (e.g. Vec::push, Vec::new)
                for impl_def_id in tcx.inherent_impls(child_def_id) {
                    visit_impl_or_trait(
                        tcx,
                        *impl_def_id,
                        out,
                        generic_fn_lookup,
                        stats,
                        visited,
                        layout_table,
                        exported_def_ids,
                    );
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
    generic_fn_lookup: &mut Vec<GenericFnLookupEntry>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
    exported_def_ids: &mut Vec<DefId>,
) {
    if !visited.insert(def_id) {
        return;
    }
    for &item in tcx.associated_item_def_ids(def_id) {
        try_export_fn(tcx, item, out, generic_fn_lookup, stats, visited, layout_table, exported_def_ids);
    }
}

fn try_export_fn<'tcx>(
    tcx: TyCtxt<'tcx>,
    def_id: DefId,
    out: &mut Vec<FnBody>,
    generic_fn_lookup: &mut Vec<GenericFnLookupEntry>,
    stats: &mut ExportStats,
    visited: &mut HashSet<DefId>,
    layout_table: &mut LayoutTable<'tcx>,
    exported_def_ids: &mut Vec<DefId>,
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
        // Use with_no_visible_paths to get the definition-site path
        // (e.g. core::convert::identity) instead of the re-export path
        // (e.g. std::convert::identity). This matches how r-a resolves
        // functions to their defining crate.
        let name = rustc_middle::ty::print::with_no_visible_paths!(
            tcx.def_path_str(def_id)
        );
        let num_generic_params = tcx.generics_of(def_id).count();

        let fn_body = FnBody {
            def_path_hash: hash,
            name,
            num_generic_params,
            body: translated,
        };
        let generic_lookup_entry = if num_generic_params > 0 {
            Some(GenericFnLookupEntry {
                key: GenericFnLookupKey {
                    stable_crate_id: hash.0,
                    normalized_path: normalize_def_path(&fn_body.name),
                    num_generic_params,
                },
                def_path_hash: hash,
            })
        } else {
            None
        };
        (fn_body, generic_lookup_entry)
    })) {
        Ok((mut fn_body, generic_lookup_entry)) => {
            // Compute layouts for each local
            let body = tcx.optimized_mir(def_id);
            for (i, decl) in body.local_decls.iter().enumerate() {
                fn_body.body.locals[i].layout = layout_table.get_or_insert(tcx, decl.ty);
            }
            stats.translated += 1;
            exported_def_ids.push(def_id);
            if let Some(entry) = generic_lookup_entry {
                generic_fn_lookup.push(entry);
            }
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
// Monomorphized layout collection
// ---------------------------------------------------------------------------

/// Collect layouts for types that appear in monomorphized generic function instances.
///
/// Scans monomorphic functions' rustc MIR for calls to generic functions with concrete
/// type args. For each such call, instantiates the callee's local types with the concrete
/// args and recursively exports layouts for all component types (ADT fields, pointees, etc.).
fn collect_mono_layouts<'tcx>(
    tcx: TyCtxt<'tcx>,
    exported_def_ids: &[DefId],
    layout_table: &mut LayoutTable<'tcx>,
    adt_table: &mut AdtTable<'tcx>,
) {
    let typing_env = ty::TypingEnv::fully_monomorphized();
    let mut visited_types: HashSet<ty::Ty<'tcx>> = HashSet::new();
    let mut visited_instances: HashSet<(DefId, ty::GenericArgsRef<'tcx>)> = HashSet::new();

    for &def_id in exported_def_ids {
        // Only scan monomorphic functions
        if tcx.generics_of(def_id).count() > 0 {
            continue;
        }
        let body = tcx.optimized_mir(def_id);
        for bb in body.basic_blocks.iter() {
            let Some(term) = bb.terminator.as_ref() else { continue };
            if let mir::TerminatorKind::Call { func, .. } = &term.kind {
                if let mir::Operand::Constant(box c) = func {
                    let func_ty = c.const_.ty();
                    if let ty::TyKind::FnDef(callee_def_id, substs) = func_ty.kind() {
                        // Skip non-generic callees
                        if tcx.generics_of(*callee_def_id).count() == 0 {
                            continue;
                        }
                        // Skip if any type arg still contains Param
                        if func_ty.has_param() {
                            continue;
                        }
                        // Deduplicate
                        if !visited_instances.insert((*callee_def_id, substs)) {
                            continue;
                        }
                        // Get callee's MIR
                        if !tcx.is_mir_available(*callee_def_id) {
                            continue;
                        }
                        let callee_body = tcx.optimized_mir(*callee_def_id);

                        // Instantiate each local's type and export layouts recursively
                        for local_decl in callee_body.local_decls.iter() {
                            let mono_ty = ty::EarlyBinder::bind(local_decl.ty)
                                .instantiate(tcx, substs);
                            let normalized = if mono_ty.has_param() {
                                mono_ty
                            } else {
                                tcx.try_normalize_erasing_regions(typing_env, mono_ty)
                                    .unwrap_or(mono_ty)
                            };
                            export_type_layout_recursive(
                                tcx,
                                normalized,
                                layout_table,
                                adt_table,
                                &mut visited_types,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Recursively export layouts for a type and all its component types.
///
/// Walks through ADT fields, ref/ptr pointees, array/slice elements, and tuple
/// elements, computing and exporting layouts for each concrete type encountered.
/// Also collects ADT definitions into adt_table for codegen-time layout computation.
fn export_type_layout_recursive<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: ty::Ty<'tcx>,
    layout_table: &mut LayoutTable<'tcx>,
    adt_table: &mut AdtTable<'tcx>,
    visited: &mut HashSet<ty::Ty<'tcx>>,
) {
    if !visited.insert(ty) {
        return;
    }
    // Skip types with unresolved params (can't compute layout)
    if ty.has_param() {
        return;
    }

    // Try to compute and export layout (best effort — unsized types will fail)
    let _ = layout_table.get_or_insert(tcx, ty);

    // Recurse into component types so codegen has layouts for field projections,
    // derefs, array indexing, etc.
    let typing_env = ty::TypingEnv::fully_monomorphized();
    match ty.kind() {
        ty::TyKind::Ref(_, inner, _) | ty::TyKind::RawPtr(inner, _) => {
            export_type_layout_recursive(tcx, *inner, layout_table, adt_table, visited);
        }
        ty::TyKind::Array(elem, _) | ty::TyKind::Slice(elem) => {
            export_type_layout_recursive(tcx, *elem, layout_table, adt_table, visited);
        }
        ty::TyKind::Tuple(tys) => {
            for t in tys.iter() {
                export_type_layout_recursive(tcx, t, layout_table, adt_table, visited);
            }
        }
        ty::TyKind::Adt(adt_def, args) => {
            // Export ADT definition for codegen-time layout computation
            adt_table.get_or_insert(tcx, *adt_def);
            for variant in adt_def.variants() {
                for field in &variant.fields {
                    let field_ty = field.ty(tcx, args);
                    let normalized = if field_ty.has_param() {
                        field_ty
                    } else {
                        tcx.try_normalize_erasing_regions(typing_env, field_ty)
                            .unwrap_or(field_ty)
                    };
                    export_type_layout_recursive(tcx, normalized, layout_table, adt_table, visited);
                }
            }
        }
        _ => {}
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

//! ra-mir-export: extract stable crate IDs from rustc.
//!
//! This rustc driver now emits only crate metadata needed for symbol
//! disambiguation: crate name + `StableCrateId`.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;

use std::cell::RefCell;
use std::path::PathBuf;

use ra_mir_types::{CrateInfo, MirData};
use rustc_driver::Compilation;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;

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
        let mir_data = MirData { crates: extract_crate_ids(tcx) };
        write_mirdata(&self.output_path, &mir_data);
        *self.result.borrow_mut() = Some(mir_data);
        Compilation::Stop
    }
}

fn extract_crate_ids(tcx: TyCtxt<'_>) -> Vec<CrateInfo> {
    let mut result = Vec::new();

    // Local crate
    let local_name = tcx.crate_name(LOCAL_CRATE);
    let local_id = tcx.stable_crate_id(LOCAL_CRATE);
    result.push(CrateInfo { name: local_name.to_string(), stable_crate_id: local_id.as_u64() });

    // Extern crates loaded by this compilation
    for &cnum in tcx.crates(()) {
        let name = tcx.crate_name(cnum);
        let id = tcx.stable_crate_id(cnum);
        result.push(CrateInfo { name: name.to_string(), stable_crate_id: id.as_u64() });
    }

    result
}

fn write_mirdata(output_path: &PathBuf, mir_data: &MirData) {
    let data = postcard::to_allocvec(mir_data).expect("postcard serialize");
    std::fs::write(output_path, &data)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", output_path.display()));
}

fn main() {
    // Parse our own flags before passing through the rest to rustc.
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

    // If no source file was provided, compile a temporary dummy file.
    let has_source = filtered_args.iter().skip(1).any(|a| !a.starts_with('-'));
    let _tmpfile;
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

    // Keep catch_unwind for robustness against rustc delayed ICEs.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rustc_driver::run_compiler(&filtered_args, &mut callbacks);
    }));

    let mir_data = callbacks.result.into_inner().unwrap_or_else(|| {
        eprintln!("error: no crate metadata extracted from compilation");
        std::process::exit(1);
    });

    // Ensure output exists even if rustc aborted late.
    write_mirdata(&output_path, &mir_data);

    eprintln!("Wrote {} crate infos to {}", mir_data.crates.len(), output_path.display());
    for info in &mir_data.crates {
        eprintln!("  {} = 0x{:016x}", info.name, info.stable_crate_id);
    }
}

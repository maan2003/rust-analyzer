//! ra-mir-export: Extract StableCrateId from rustc sysroot crates.
//!
//! A rustc driver that compiles a dummy file, loads sysroot crates, and
//! extracts their StableCrateIds. Output is a postcard-serialized
//! `Vec<CrateInfo>` written to a `.mirdata` file.
//!
//! Usage:
//!   ra-mir-export -o sysroot.mirdata dummy.rs
//!
//! Where dummy.rs can be `fn main() {}`. The driver uses the sysroot from
//! the rustc it's linked against. Pass `--sysroot <path>` if needed.

#![feature(rustc_private)]

extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;

use std::cell::RefCell;
use std::path::PathBuf;

use rustc_driver::Compilation;
use rustc_hir::def_id::LOCAL_CRATE;
use rustc_middle::ty::TyCtxt;
use serde::{Deserialize, Serialize};

/// Crate name + StableCrateId, serialized to .mirdata files.
#[derive(Serialize, Deserialize, Debug)]
pub struct CrateInfo {
    pub name: String,
    pub stable_crate_id: u64,
}

struct ExportCallbacks {
    result: RefCell<Vec<CrateInfo>>,
}

impl rustc_driver::Callbacks for ExportCallbacks {
    fn after_analysis<'tcx>(
        &mut self,
        _compiler: &rustc_interface::interface::Compiler,
        tcx: TyCtxt<'tcx>,
    ) -> Compilation {
        let infos = extract_crate_ids(tcx);
        *self.result.borrow_mut() = infos;
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
        result: RefCell::new(Vec::new()),
    };

    rustc_driver::run_compiler(&filtered_args, &mut callbacks);

    // Write output
    let infos = callbacks.result.into_inner();
    if infos.is_empty() {
        eprintln!("warning: no crate info extracted");
    }

    let data = postcard::to_allocvec(&infos).expect("postcard serialize");
    std::fs::write(&output_path, &data)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", output_path.display()));

    eprintln!("Wrote {} crate infos to {}", infos.len(), output_path.display());
    for info in &infos {
        eprintln!("  {} = 0x{:016x}", info.name, info.stable_crate_id);
    }
}

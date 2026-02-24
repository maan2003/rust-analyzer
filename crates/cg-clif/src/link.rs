//! Linker invocation for producing executables.
//!
//! Links against `libstd-*.so` from the host rustc sysroot using dynamic linking.
//! The `.so` contains core, alloc, std, and the allocator shim — no need for
//! rlib linking, `--start-group`, or allocator shim generation.
//! Hardcoded for Linux x86_64 for now.
//!
//! Crate disambiguators are loaded from `.mirdata` files produced by
//! `ra-mir-export`, which extracts `StableCrateId` from sysroot rlibs via
//! a rustc driver.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use ra_mir_types::MirData;

use crate::LayoutArc;

/// Get the target library directory from the host `rustc`.
pub fn find_target_libdir() -> Result<PathBuf, String> {
    let output = Command::new("rustc")
        .args(["--print", "target-libdir"])
        .output()
        .map_err(|e| format!("failed to run `rustc --print target-libdir`: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "rustc --print target-libdir failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let path = String::from_utf8(output.stdout)
        .map_err(|e| format!("non-UTF8 rustc output: {e}"))?;
    Ok(PathBuf::from(path.trim()))
}

/// Find `libstd-*.so` in the sysroot libdir.
pub fn find_libstd_so(libdir: &Path) -> Result<PathBuf, String> {
    let entries = std::fs::read_dir(libdir)
        .map_err(|e| format!("failed to read {}: {e}", libdir.display()))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("readdir error: {e}"))?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("libstd-") && name.ends_with(".so") {
                return Ok(path);
            }
        }
    }
    Err(format!("libstd-*.so not found in {}", libdir.display()))
}

/// Link an object file against `libstd.so` to produce a dynamically-linked executable.
pub fn link_executable(obj_path: &Path, output_path: &Path) -> Result<(), String> {
    let libdir = find_target_libdir()?;
    let libstd_so = find_libstd_so(&libdir)?;

    let mut cmd = Command::new("cc");
    cmd.arg(obj_path);
    cmd.arg("-o").arg(output_path);

    // Link against libstd.so (contains core + alloc + std + allocator shim)
    cmd.arg(&libstd_so);

    // Embed rpath so the binary can find libstd.so at runtime
    cmd.arg(format!("-Wl,-rpath,{}", libdir.display()));

    // Suppress unused library warnings if the linker supports it
    cmd.arg("-Wl,--as-needed");

    let output = cmd
        .output()
        .map_err(|e| format!("failed to run cc: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "linking failed (exit {}):\nstdout: {}\nstderr: {}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Crate disambiguator loading from .mirdata files
// ---------------------------------------------------------------------------

/// Load crate disambiguators from a `.mirdata` file produced by `ra-mir-export`.
///
/// Returns a map of crate name → StableCrateId (as u64).
pub fn load_crate_disambiguators(mirdata_path: &Path) -> Result<HashMap<String, u64>, String> {
    let data = std::fs::read(mirdata_path)
        .map_err(|e| format!("failed to read {}: {e}", mirdata_path.display()))?;

    let mir_data: MirData = postcard::from_bytes(&data)
        .map_err(|e| format!("failed to deserialize {}: {e}", mirdata_path.display()))?;

    let mut map = HashMap::new();
    for info in mir_data.crates {
        map.insert(info.name, info.stable_crate_id);
    }
    Ok(map)
}

/// Load and deserialize a `.mirdata` file, returning the full `MirData`.
pub fn load_mirdata(mirdata_path: &Path) -> Result<MirData, String> {
    let data = std::fs::read(mirdata_path)
        .map_err(|e| format!("failed to read {}: {e}", mirdata_path.display()))?;
    postcard::from_bytes(&data)
        .map_err(|e| format!("failed to deserialize {}: {e}", mirdata_path.display()))
}

/// Load mirdata and convert its layout table to `Vec<LayoutArc>`.
pub fn load_mirdata_layouts(mirdata_path: &Path) -> Result<(MirData, Vec<LayoutArc>), String> {
    let mir_data = load_mirdata(mirdata_path)?;
    let layouts = crate::layout::convert_mirdata_layouts(&mir_data.layouts);
    Ok((mir_data, layouts))
}

/// Find the `.mirdata` file path.
///
/// Uses `RA_MIRDATA` environment variable, or looks for `sysroot.mirdata`
/// next to the sysroot libdir.
pub fn find_mirdata_path() -> Result<PathBuf, String> {
    // Check environment variable first
    if let Ok(path) = std::env::var("RA_MIRDATA") {
        let p = PathBuf::from(path);
        if p.exists() {
            return Ok(p);
        }
        return Err(format!("RA_MIRDATA set to {} but file does not exist", p.display()));
    }

    // Look next to the sysroot libdir
    let libdir = find_target_libdir()?;
    let mirdata = libdir.join("sysroot.mirdata");
    if mirdata.exists() {
        return Ok(mirdata);
    }

    Err(format!(
        "no .mirdata file found. Run `ra-mir-export` to generate it, \
         or set RA_MIRDATA environment variable.\n\
         Looked at: {}",
        mirdata.display()
    ))
}

/// Load crate disambiguators from the `.mirdata` file.
///
/// Finds the mirdata file via `find_mirdata_path()` and deserializes it.
pub fn extract_crate_disambiguators() -> Result<HashMap<String, u64>, String> {
    let mirdata_path = find_mirdata_path()?;
    load_crate_disambiguators(&mirdata_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ra_mir_types::CrateInfo;

    #[test]
    fn find_libstd_so_in_sysroot() {
        let libdir = find_target_libdir().expect("rustc not available");
        let so_path = find_libstd_so(&libdir).expect("libstd.so not found");
        assert!(so_path.exists(), "libstd.so path should exist: {}", so_path.display());
    }

    #[test]
    fn mirdata_roundtrip() {
        let mir_data = MirData {
            crates: vec![
                CrateInfo { name: "std".to_string(), stable_crate_id: 0x1234 },
                CrateInfo { name: "core".to_string(), stable_crate_id: 0x5678 },
            ],
            bodies: vec![],
            layouts: vec![],
            generic_fn_lookup: vec![],
        };
        let data = postcard::to_allocvec(&mir_data).expect("serialize");
        let decoded: MirData = postcard::from_bytes(&data).expect("deserialize");
        assert_eq!(decoded.crates.len(), 2);
        assert_eq!(decoded.crates[0].name, "std");
        assert_eq!(decoded.crates[0].stable_crate_id, 0x1234);
        assert_eq!(decoded.crates[1].name, "core");
        assert_eq!(decoded.crates[1].stable_crate_id, 0x5678);
    }
}

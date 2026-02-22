//! Linker invocation for producing executables.
//!
//! Shells out to `cc` with sysroot rlibs from the host rustc toolchain.
//! Hardcoded for Linux x86_64 for now.

use std::path::{Path, PathBuf};
use std::process::Command;

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

/// Collect all `.rlib` files in a directory.
pub fn collect_rlibs(libdir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(libdir)
        .map_err(|e| format!("failed to read {}: {e}", libdir.display()))?;

    let mut rlibs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("readdir error: {e}"))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rlib") {
            rlibs.push(path);
        }
    }
    rlibs.sort();
    Ok(rlibs)
}

/// Link an object file against the Rust sysroot to produce an executable.
pub fn link_executable(obj_path: &Path, output_path: &Path) -> Result<(), String> {
    let libdir = find_target_libdir()?;
    let rlibs = collect_rlibs(&libdir)?;

    let mut cmd = Command::new("cc");
    cmd.arg(obj_path);
    cmd.arg("-o").arg(output_path);
    cmd.arg("-L").arg(&libdir);

    // Add all rlibs
    for rlib in &rlibs {
        cmd.arg(rlib);
    }

    // System libraries needed by std on Linux
    cmd.args(["-lc", "-lgcc_s", "-lpthread", "-lm", "-ldl", "-lrt"]);

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

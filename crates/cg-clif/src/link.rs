//! Linker invocation for producing executables.
//!
//! Shells out to `cc` with sysroot rlibs from the host rustc toolchain.
//! Hardcoded for Linux x86_64 for now.

use std::collections::HashMap;
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

/// Collect all `.rlib` files in a directory, filtering out non-standard rlibs
/// (e.g., `rustc-dev` crates that may be symlinked in on NixOS).
pub fn collect_rlibs(libdir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = std::fs::read_dir(libdir)
        .map_err(|e| format!("failed to read {}: {e}", libdir.display()))?;

    let mut rlibs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| format!("readdir error: {e}"))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "rlib") {
            // Skip rlibs that are symlinks to rustc-dev (not part of std sysroot)
            if let Ok(target) = std::fs::read_link(&path) {
                let target_str = target.to_string_lossy();
                if target_str.contains("rustc-dev") {
                    continue;
                }
            }
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

    // Add all rlibs inside a group to resolve circular dependencies
    cmd.arg("-Wl,--start-group");
    for rlib in &rlibs {
        cmd.arg(rlib);
    }
    cmd.arg("-Wl,--end-group");

    // System libraries needed by std on Linux
    cmd.args(["-lc", "-lgcc_s", "-lpthread", "-lm", "-ldl", "-lrt"]);

    // Garbage-collect unused sections — std pulls in a lot of code we don't need
    cmd.arg("-Wl,--gc-sections");

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
// Crate disambiguator extraction from rlib symbols
// ---------------------------------------------------------------------------

/// Decode a base-62 encoded string (reverse of `base_62_encode` in symbol_mangling.rs).
fn base_62_decode(s: &str) -> u64 {
    s.bytes().fold(0u64, |acc, c| {
        let d = match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'z' => c - b'a' + 10,
            b'A'..=b'Z' => c - b'A' + 36,
            _ => panic!("invalid base-62 digit: {}", c as char),
        };
        acc * 62 + d as u64
    })
}

/// Try to parse a crate root marker `C` at position `pos` in a v0 symbol (after `_R` prefix).
/// Returns `(crate_name, disambiguator)` if successful.
fn try_parse_crate_at(rest: &str, pos: usize) -> Option<(String, u64)> {
    let after_c = &rest[pos + 1..];

    let (dis, after_dis) = if after_c.starts_with('s') {
        // Disambiguator present: 's' + integer_62
        // integer_62 encoding: 0 → "_", n > 0 → base62(n-1) + "_"
        // push_disambiguator(D) calls push_opt_integer_62("s", D) which
        // writes "s" + integer_62(D-1). So: D = decode_integer_62(...) + 1.
        let after_s = &after_c[1..];
        let underscore = after_s.find('_')?;
        let dis_str = &after_s[..underscore];
        // Validate all chars are base-62 digits
        if !dis_str.bytes().all(|c| c.is_ascii_alphanumeric()) {
            return None;
        }
        // Reverse integer_62: empty → 0, non-empty → base_62_decode + 1
        let integer_62_val = if dis_str.is_empty() {
            0
        } else {
            base_62_decode(dis_str) + 1
        };
        // Reverse push_opt_integer_62: D = integer_62_val + 1
        let dis = integer_62_val + 1;
        (dis, &after_s[underscore + 1..])
    } else if after_c.starts_with(|c: char| c.is_ascii_digit()) {
        // No 's' — disambiguator is 0, next must be a digit (start of ident length)
        (0, after_c)
    } else {
        return None;
    };

    // Parse ident: decimal length followed by the name
    let digit_end = after_dis
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_dis.len());
    if digit_end == 0 {
        return None;
    }
    let len: usize = after_dis[..digit_end].parse().ok()?;
    let mut name_start = digit_end;
    // Handle optional '_' separator (when ident starts with '_' or digit)
    if after_dis[name_start..].starts_with('_') {
        name_start += 1;
    }
    if name_start + len > after_dis.len() {
        return None;
    }
    let name = &after_dis[name_start..name_start + len];
    Some((name.to_owned(), dis))
}

/// Find the disambiguator for a specific crate name in a v0 mangled symbol.
/// Scans all `C` markers in the symbol to find one matching the target crate name.
fn find_crate_disambiguator(symbol: &str, target_crate: &str) -> Option<u64> {
    let rest = symbol.strip_prefix("_R")?;

    for (pos, _) in rest.char_indices().filter(|&(_, c)| c == 'C') {
        if let Some((name, dis)) = try_parse_crate_at(rest, pos) {
            if name == target_crate {
                return Some(dis);
            }
        }
    }
    None
}

/// Extract crate name from an rlib filename like `libstd-957b1fe07cf96b13.rlib` → `"std"`.
fn crate_name_from_rlib(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let name = stem.strip_prefix("lib")?;
    // Strip the hash suffix: everything after the last '-'
    let crate_name = if let Some(pos) = name.rfind('-') {
        &name[..pos]
    } else {
        name
    };
    Some(crate_name.replace('-', "_"))
}

/// Scan all rlibs in a sysroot libdir, extract crate name → disambiguator mapping.
/// Uses the archive symbol table directly (avoids parsing individual object files).
pub fn extract_crate_disambiguators(libdir: &Path) -> Result<HashMap<String, u64>, String> {
    let rlibs = collect_rlibs(libdir)?;
    let mut map = HashMap::new();

    for rlib_path in &rlibs {
        let expected_crate = match crate_name_from_rlib(rlib_path) {
            Some(name) => name,
            None => continue,
        };

        let data = std::fs::read(rlib_path)
            .map_err(|e| format!("failed to read {}: {e}", rlib_path.display()))?;

        let archive = object::read::archive::ArchiveFile::parse(&*data)
            .map_err(|e| format!("failed to parse archive {}: {e}", rlib_path.display()))?;

        // Read symbols from the archive symbol table (index at the start of the archive).
        let symbols = match archive.symbols() {
            Ok(Some(syms)) => syms,
            _ => continue,
        };

        let mut found_expected = false;
        for sym in symbols {
            let Ok(sym) = sym else { continue };
            let name = std::str::from_utf8(sym.name()).unwrap_or("");
            if !name.starts_with("_R") {
                continue;
            }
            // Look for the expected crate (derived from rlib filename)
            if !found_expected {
                if let Some(dis) = find_crate_disambiguator(name, &expected_crate) {
                    map.insert(expected_crate.clone(), dis);
                    found_expected = true;
                }
            }
            // Also extract __rustc crate disambiguator from any symbol that references it
            // (the __rustc crate has no separate rlib)
            if !map.contains_key("__rustc") {
                if let Some(dis) = find_crate_disambiguator(name, "__rustc") {
                    map.insert("__rustc".to_owned(), dis);
                }
            }
            if found_expected && map.contains_key("__rustc") {
                break;
            }
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn find_std_disambiguator() {
        let sym = "_RNvNtCs6FpuVETr9fs_3std7process4exit";
        let dis = find_crate_disambiguator(sym, "std").unwrap();
        assert!(dis > 0, "std disambiguator should be non-zero, got {dis}");
    }

    #[test]
    fn find_core_in_std_symbol() {
        // Symbol from std rlib that references core
        let sym = "_RINvMNtCsi96gERPWvbJ_4core3stre12trim_matchesNvMNtNtB5_4char7methodsc13is_whitespaceECs6FpuVETr9fs_3std";
        let core_dis = find_crate_disambiguator(sym, "core").unwrap();
        assert!(core_dis > 0);
        let std_dis = find_crate_disambiguator(sym, "std").unwrap();
        assert!(std_dis > 0);
    }

    #[test]
    fn find_zero_disambiguator() {
        let sym = "_RNvC4main4func";
        let dis = find_crate_disambiguator(sym, "main").unwrap();
        assert_eq!(dis, 0);
    }

    #[test]
    fn base_62_roundtrip() {
        assert_eq!(base_62_decode("0"), 0);
        assert_eq!(base_62_decode("9"), 9);
        assert_eq!(base_62_decode("a"), 10);
        assert_eq!(base_62_decode("z"), 35);
        assert_eq!(base_62_decode("A"), 36);
        assert_eq!(base_62_decode("Z"), 61);
        assert_eq!(base_62_decode("10"), 62);
    }

    #[test]
    fn crate_name_from_rlib_path() {
        let p = Path::new("/some/path/libstd-957b1fe07cf96b13.rlib");
        assert_eq!(crate_name_from_rlib(p).unwrap(), "std");
        let p = Path::new("/some/path/libstd_detect-2b11eee15f93fb2f.rlib");
        assert_eq!(crate_name_from_rlib(p).unwrap(), "std_detect");
    }

    #[test]
    fn extract_disambiguators_from_sysroot() {
        let libdir = find_target_libdir().expect("rustc not available");
        let map = extract_crate_disambiguators(&libdir).expect("extraction failed");
        assert!(map.contains_key("std"), "should find std crate, got keys: {:?}", map.keys().collect::<Vec<_>>());
        assert!(map.contains_key("core"), "should find core crate, got keys: {:?}", map.keys().collect::<Vec<_>>());
    }
}

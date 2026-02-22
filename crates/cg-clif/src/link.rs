//! Linker invocation for producing executables.
//!
//! Links against `libstd-*.so` from the host rustc sysroot using dynamic linking.
//! The `.so` contains core, alloc, std, and the allocator shim — no need for
//! rlib linking, `--start-group`, or allocator shim generation.
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
// Crate disambiguator extraction from libstd.so symbols
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

/// Scan `libstd-*.so` dynamic symbols to extract crate name → disambiguator mapping.
/// The single .so contains symbols from core, alloc, std, and __rustc.
pub fn extract_crate_disambiguators(libdir: &Path) -> Result<HashMap<String, u64>, String> {
    use object::{Object, ObjectSymbol};

    let so_path = find_libstd_so(libdir)?;
    let data = std::fs::read(&so_path)
        .map_err(|e| format!("failed to read {}: {e}", so_path.display()))?;
    let obj = object::read::File::parse(&*data)
        .map_err(|e| format!("failed to parse {}: {e}", so_path.display()))?;

    let mut map = HashMap::new();
    // Crates we expect to find in the std .so
    let target_crates = ["std", "core", "alloc", "__rustc"];

    for sym in obj.dynamic_symbols() {
        let Ok(name) = sym.name() else { continue };
        if !name.starts_with("_R") {
            continue;
        }
        for &crate_name in &target_crates {
            if map.contains_key(crate_name) {
                continue;
            }
            if let Some(dis) = find_crate_disambiguator(name, crate_name) {
                map.insert(crate_name.to_owned(), dis);
            }
        }
        if map.len() == target_crates.len() {
            break;
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
    fn find_libstd_so_in_sysroot() {
        let libdir = find_target_libdir().expect("rustc not available");
        let so_path = find_libstd_so(&libdir).expect("libstd.so not found");
        assert!(so_path.exists(), "libstd.so path should exist: {}", so_path.display());
    }

    #[test]
    fn extract_disambiguators_from_sysroot() {
        let libdir = find_target_libdir().expect("rustc not available");
        let map = extract_crate_disambiguators(&libdir).expect("extraction failed");
        assert!(map.contains_key("std"), "should find std crate, got keys: {:?}", map.keys().collect::<Vec<_>>());
        assert!(map.contains_key("core"), "should find core crate, got keys: {:?}", map.keys().collect::<Vec<_>>());
    }
}

test-clif:
    if [ -n "${RA_MIRDATA:-}" ]; then \
      mirdata="$RA_MIRDATA"; \
    else \
      mirdata_dir="$(mktemp -d)"; \
      trap 'rm -rf "$mirdata_dir"' EXIT; \
      mirdata="$mirdata_dir/sysroot.mirdata"; \
    fi; \
    cargo run --manifest-path ra-mir-export/Cargo.toml --release -- -o "$mirdata"; \
    RA_MIRDATA="$mirdata" cargo nextest run -p cg-clif --color=never

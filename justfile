set positional-arguments := true

# Generate minimal crate metadata for symbol disambiguation.
update-mirdata:
    REPO_ROOT="{{justfile_directory()}}"; \
    mirdata="$REPO_ROOT/target/sysroot.mirdata"; \
    mkdir -p "$REPO_ROOT/target"; \
    cargo run --manifest-path "$REPO_ROOT/ra-mir-export/Cargo.toml" --release -- -o "$mirdata"

test-clif *args:
    REPO_ROOT="{{justfile_directory()}}"; \
    mirdata="$REPO_ROOT/target/sysroot.mirdata"; \
    if [ ! -f "$mirdata" ]; then \
      echo "missing $mirdata; generating stable crate-id metadata"; \
      cargo run --manifest-path "$REPO_ROOT/ra-mir-export/Cargo.toml" --release -- -o "$mirdata"; \
    fi; \
    RA_MIRDATA="$mirdata" cargo nextest run -p cg-clif --color=never "$@"

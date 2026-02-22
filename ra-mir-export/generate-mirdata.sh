#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="${1:-/tmp/sysroot.mirdata}"

cd "$SCRIPT_DIR"
cargo build --release -q
cargo run --release -q -- -o "$OUTPUT"

echo "export RA_MIRDATA=$OUTPUT"

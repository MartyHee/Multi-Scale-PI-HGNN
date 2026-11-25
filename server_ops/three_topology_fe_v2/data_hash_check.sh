#!/usr/bin/env bash
# Small data-identity check used by preflight.  Does NOT read responses.
# FULL=1 additionally hashes the whole package once; do that at most once per
# upload and save the evidence, not once per job.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_config.sh"

OUT_DIR="${1:-}"
ARGS=(--package-root "$PACKAGE_ROOT")
[ -n "$OUT_DIR" ] && ARGS+=(--out "$OUT_DIR/package_verify.json")
[ "${FULL:-0}" = "1" ] && ARGS+=(--full)
"$PYTHON" -B "$THREE_TOPO_HERE/verify_package.py" "${ARGS[@]}"

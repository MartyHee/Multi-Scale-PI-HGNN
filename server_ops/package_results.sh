#!/usr/bin/env bash
# =============================================================================
# package_results.sh — Lightweight artifact packaging for server experiments
#
# Usage:
#   bash server_ops/package_results.sh <run_dir> <job_name> [--include-last]
#
# Arguments:
#   run_dir      Path to the experiment output directory, typically
#                outputs/<ModelName>/<YYYYMMDDHHMMSS>/
#   job_name     Short identifier used in the archive filename, e.g. server_smoke_mlp
#   --include-last  (optional) also include last_model.pt and last_checkpoint.pt
#
# Output:
#   remote_artifacts/<job_name>_<YYYYMMDDHHMMSS>.tar.gz
#
# Artifact contents (default):
#   - job yaml copy (if found)
#   - config_resolved.yaml
#   - model_summary.json
#   - train_log.csv
#   - metrics_summary.json
#   - metrics_test.json (if exists)
#   - loss_curve.png (if exists)
#   - metric_curve.png / metric_curve_disp.png / metric_curve_force.png (if exist)
#   - best_model.pt
#   - server stdout/stderr log (if found)
#   - error traceback (if exists)
#   - last_model.pt and last_checkpoint.pt (only with --include-last)
#
# NOT included (always excluded):
#   - processed/
#   - raw_data/
#   - full outputs/ tree
#   - periodic checkpoints
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash $0 <run_dir> <job_name> [--include-last]"
    echo ""
    echo "  run_dir    e.g. outputs/MLP/20260617163452"
    echo "  job_name   e.g. server_smoke_mlp"
    exit 1
fi

RUN_DIR="$1"
JOB_NAME="$2"
INCLUDE_LAST=false
if [ "${3:-}" = "--include-last" ]; then
    INCLUDE_LAST=true
fi

# --- Validate run_dir ---
if [ ! -d "$RUN_DIR" ]; then
    echo "[ERROR] Run directory not found: $RUN_DIR"
    exit 1
fi

# --- Resolve paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)
ARTIFACT_NAME="${JOB_NAME}_${TIMESTAMP}"
ARTIFACT_DIR="${PROJECT_DIR}/remote_artifacts/${ARTIFACT_NAME}"
ARCHIVE="${PROJECT_DIR}/remote_artifacts/${ARTIFACT_NAME}.tar.gz"

mkdir -p "$ARTIFACT_DIR"

echo "=========================================="
echo " Packaging results"
echo "=========================================="
echo "  Run dir:   ${RUN_DIR}"
echo "  Job:       ${JOB_NAME}"
echo "  Artifact:  ${ARCHIVE}"

# --- Copy files ---

# 1. Config & metadata
for f in config_resolved.yaml model_summary.json; do
    src="${RUN_DIR}/${f}"
    if [ -f "$src" ]; then
        cp "$src" "${ARTIFACT_DIR}/"
        echo "  [ok]  ${f}"
    fi
done

# 2. Training log
src="${RUN_DIR}/train_log.csv"
if [ -f "$src" ]; then
    cp "$src" "${ARTIFACT_DIR}/"
    echo "  [ok]  train_log.csv"
fi

# 3. Metrics
for f in metrics_summary.json metrics_test.json; do
    src="${RUN_DIR}/${f}"
    if [ -f "$src" ]; then
        cp "$src" "${ARTIFACT_DIR}/"
        echo "  [ok]  ${f}"
    fi
done

# 4. Plots
for pat in loss_curve.png metric_curve*.png; do
    # shellcheck disable=SC2086
    found=$(find "$RUN_DIR" -maxdepth 1 -name "$pat" 2>/dev/null || true)
    if [ -n "$found" ]; then
        # shellcheck disable=SC2086
        cp $found "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "  [ok]  $(basename $pat) (glob)"
    fi
done

# 5. Best model
src="${RUN_DIR}/best_model.pt"
if [ -f "$src" ]; then
    cp "$src" "${ARTIFACT_DIR}/"
    SIZE=$(stat -c%s "$src" 2>/dev/null || stat -f%z "$src" 2>/dev/null)
    echo "  [ok]  best_model.pt  (${SIZE} bytes)"
fi

# 6. Last model (optional)
if [ "$INCLUDE_LAST" = true ]; then
    for f in last_model.pt last_checkpoint.pt; do
        src="${RUN_DIR}/${f}"
        if [ -f "$src" ]; then
            cp "$src" "${ARTIFACT_DIR}/"
            echo "  [ok]  ${f}"
        fi
    done
fi

# 7. Job yaml (look in remote_jobs/ for a matching name)
JOB_YAML="${PROJECT_DIR}/remote_jobs/${JOB_NAME}.yaml"
if [ ! -f "$JOB_YAML" ]; then
    # try broader: match any yaml whose basename contains job_name
    JOB_YAML=$(find "${PROJECT_DIR}/remote_jobs" -maxdepth 1 -name "${JOB_NAME}*.yaml" 2>/dev/null | head -1 || true)
fi
if [ -n "$JOB_YAML" ] && [ -f "$JOB_YAML" ]; then
    cp "$JOB_YAML" "${ARTIFACT_DIR}/"
    echo "  [ok]  $(basename "$JOB_YAML")"
fi

# 8. Git info
GIT_INFO="${ARTIFACT_DIR}/git_info.txt"
{
    echo "Git branch: $(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    echo "Git commit: $(cd "$PROJECT_DIR" && git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo "Git date:   $(cd "$PROJECT_DIR" && git log -1 --format=%ci 2>/dev/null || echo 'unknown')"
    echo "Hostname:   $(hostname 2>/dev/null || echo 'unknown')"
    echo "User:       $(whoami 2>/dev/null || echo 'unknown')"
    echo "Date:       $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
} > "$GIT_INFO"
echo "  [ok]  git_info.txt"

# 9. Server log (look in logs/remote/ for matching job name)
SERVER_LOG=$(find "${PROJECT_DIR}/logs/remote" -maxdepth 1 -name "${JOB_NAME}*.log" 2>/dev/null | sort | tail -1 || true)
if [ -n "$SERVER_LOG" ] && [ -f "$SERVER_LOG" ]; then
    cp "$SERVER_LOG" "${ARTIFACT_DIR}/server_output.log"
    echo "  [ok]  server_output.log"
fi

# 10. Error traceback (look for common error files in run_dir)
for err_pat in "error*.txt" "traceback*.txt" "*.stderr"; do
    # shellcheck disable=SC2086
    err_found=$(find "$RUN_DIR" -maxdepth 1 -name "$err_pat" 2>/dev/null || true)
    if [ -n "$err_found" ]; then
        # shellcheck disable=SC2086
        cp $err_found "$ARTIFACT_DIR/" 2>/dev/null || true
        echo "  [ok]  $(basename $err_pat) (error log)"
    fi
done

# --- Create archive ---
cd "$PROJECT_DIR/remote_artifacts"
tar czf "${ARCHIVE_NAME}.tar.gz" "$ARTIFACT_NAME" 2>/dev/null || {
    # fallback to tar without cd
    cd "$PROJECT_DIR"
    tar czf "$ARCHIVE" -C "$(dirname "$ARTIFACT_DIR")" "$ARTIFACT_NAME"
}
rm -rf "$ARTIFACT_DIR"

ARCHIVE_SIZE=$(stat -c%s "$ARCHIVE" 2>/dev/null || stat -f%z "$ARCHIVE" 2>/dev/null)
echo ""
echo "=========================================="
echo " Artifact ready"
echo "=========================================="
echo "  Archive: ${ARCHIVE}"
echo "  Size:    ${ARCHIVE_SIZE} bytes ($(( ARCHIVE_SIZE / 1024 )) KB)"
echo "=========================================="

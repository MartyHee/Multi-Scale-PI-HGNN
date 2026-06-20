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
ARTIFACT_DIR="${PROJECT_DIR}/remote_artifacts"
ARTIFACT_STAGING="${ARTIFACT_DIR}/${JOB_NAME}_${TIMESTAMP}"
ARCHIVE_NAME="${JOB_NAME}_${TIMESTAMP}.tar.gz"
ARCHIVE_PATH="${ARTIFACT_DIR}/${ARCHIVE_NAME}"

ARTIFACT_FILE_COUNT=0

mkdir -p "$ARTIFACT_STAGING"

echo "=========================================="
echo " Packaging results"
echo "=========================================="
echo "  Run dir:  ${RUN_DIR}"
echo "  Job:      ${JOB_NAME}"
echo "  Archive:  ${ARCHIVE_PATH}"

# --- Helper: safe-copy a file (no failure if missing) ---
_safe_copy() {
    local src="$1"
    local dst_dir="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst_dir/" 2>/dev/null || true
        ARTIFACT_FILE_COUNT=$((ARTIFACT_FILE_COUNT + 1))
        echo "  [ok]  $(basename "$src")"
    fi
}

# 1. Core config & metadata
_safe_copy "${RUN_DIR}/config_resolved.yaml" "$ARTIFACT_STAGING"
_safe_copy "${RUN_DIR}/model_summary.json" "$ARTIFACT_STAGING"

# 2. Training log
_safe_copy "${RUN_DIR}/train_log.csv" "$ARTIFACT_STAGING"

# 3. Metrics
_safe_copy "${RUN_DIR}/metrics_summary.json" "$ARTIFACT_STAGING"
_safe_copy "${RUN_DIR}/metrics_test.json" "$ARTIFACT_STAGING"

# 4. Plots (glob — loop over actual matches, not pattern strings)
for pat in "loss_curve.png" "metric_curve"*.png; do
    # shellcheck disable=SC2086
    while IFS= read -r -d '' f; do
        _safe_copy "$f" "$ARTIFACT_STAGING"
    done < <(find "$RUN_DIR" -maxdepth 1 -name "$pat" -print0 2>/dev/null || true)
done

# 5. Best model (always included if exists)
_safe_copy "${RUN_DIR}/best_model.pt" "$ARTIFACT_STAGING"

# 6. Last model / checkpoint (optional, --include-last)
if [ "$INCLUDE_LAST" = true ]; then
    _safe_copy "${RUN_DIR}/last_model.pt" "$ARTIFACT_STAGING"
    _safe_copy "${RUN_DIR}/last_checkpoint.pt" "$ARTIFACT_STAGING"
fi

# 7. Job yaml (look in remote_jobs/ for a matching name)
JOB_YAML="${PROJECT_DIR}/remote_jobs/${JOB_NAME}.yaml"
if [ ! -f "$JOB_YAML" ]; then
    JOB_YAML=$(find "${PROJECT_DIR}/remote_jobs" -maxdepth 1 -name "${JOB_NAME}*.yaml" 2>/dev/null | head -1 || true)
fi
_safe_copy "$JOB_YAML" "$ARTIFACT_STAGING"

# 8. Git info
GIT_INFO_PATH="${ARTIFACT_STAGING}/git_info.txt"
{
    echo "Git branch: $(cd "$PROJECT_DIR" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo 'unknown')"
    echo "Git commit: $(cd "$PROJECT_DIR" && git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    echo "Git date:   $(cd "$PROJECT_DIR" && git log -1 --format=%ci 2>/dev/null || echo 'unknown')"
    echo "Hostname:   $(hostname 2>/dev/null || echo 'unknown')"
    echo "User:       $(whoami 2>/dev/null || echo 'unknown')"
    echo "Date:       $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
} > "$GIT_INFO_PATH"
ARTIFACT_FILE_COUNT=$((ARTIFACT_FILE_COUNT + 1))
echo "  [ok]  git_info.txt"

# 9. Server log (look in logs/remote/ for matching job name)
SERVER_LOG=$(find "${PROJECT_DIR}/logs/remote" -maxdepth 1 -name "${JOB_NAME}*.log" 2>/dev/null | sort | tail -1 || true)
_safe_copy "$SERVER_LOG" "$ARTIFACT_STAGING"
# Rename to server_output.log for consistency
if [ -f "${ARTIFACT_STAGING}/$(basename "$SERVER_LOG")" ]; then
    mv "${ARTIFACT_STAGING}/$(basename "$SERVER_LOG")" "${ARTIFACT_STAGING}/server_output.log" 2>/dev/null || true
fi

# 10. Error traceback (look for common error files in run_dir)
for err_pat in "error*.txt" "traceback*.txt" "*.stderr"; do
    while IFS= read -r -d '' f; do
        _safe_copy "$f" "$ARTIFACT_STAGING"
    done < <(find "$RUN_DIR" -maxdepth 1 -name "$err_pat" -print0 2>/dev/null || true)
done

# --- Create archive ---
echo ""
echo "  Files staged: ${ARTIFACT_FILE_COUNT}"
echo "  Creating archive ..."

# Use tar -C to avoid cd; if the staging dir is empty (no files were copied),
# still create an archive containing an empty directory marker.
if [ "$ARTIFACT_FILE_COUNT" -gt 0 ]; then
    tar czf "$ARCHIVE_PATH" -C "$ARTIFACT_DIR" "${JOB_NAME}_${TIMESTAMP}" 2>/dev/null
    ARCHIVE_OK=$?
else
    # Create minimal archive with a placeholder
    echo "placeholder - no result files found" > "${ARTIFACT_STAGING}/README.txt"
    tar czf "$ARCHIVE_PATH" -C "$ARTIFACT_DIR" "${JOB_NAME}_${TIMESTAMP}" 2>/dev/null
    ARCHIVE_OK=$?
fi

rm -rf "$ARTIFACT_STAGING"

ARCHIVE_SIZE=$(stat -c%s "$ARCHIVE_PATH" 2>/dev/null || stat -f%z "$ARCHIVE_PATH" 2>/dev/null)
echo ""
echo "=========================================="
echo " Artifact ready"
echo "=========================================="
echo "  Archive: ${ARCHIVE_PATH}"
echo "  Size:    ${ARCHIVE_SIZE} bytes ($(( ARCHIVE_SIZE / 1024 )) KB) [${ARCHIVE_SIZE} bytes]"
echo "  Contents:"
tar -tzf "$ARCHIVE_PATH" 2>/dev/null | sed 's/^/    /' || echo "    (unable to list)"
echo "=========================================="

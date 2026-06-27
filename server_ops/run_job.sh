#!/usr/bin/env bash
# =============================================================================
# run_job.sh — Unified job launcher for GPU server training
#
# Usage:
#   bash server_ops/run_job.sh remote_jobs/<job_name>.yaml
#
# This script:
#   1. Prints environment info (conda, torch, CUDA, nvidia-smi, git commit)
#   2. Parses the job YAML (simple key=value grep, not a full YAML parser)
#   3. Launches training with stdout+stderr logged to logs/remote/<job_name>_<timestamp>.log
#   4. On completion (or failure), optionally packages results via package_results.sh
#   5. Failing commands return non-zero and still attempt artifact packaging
#
# Prerequisites:
#   - Must be run from the Multi-Scale-PI-HGNN project root
#   - Conda environment must already be activated (conda activate pi_hgnn)
# =============================================================================
set -euo pipefail

# ---- Config ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

# ---- Parse args ----
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 remote_jobs/<job_name>.yaml"
    exit 1
fi
JOB_YAML="$1"
JOB_NAME=$(basename "$JOB_YAML" .yaml)

if [ ! -f "$JOB_YAML" ]; then
    echo "[ERROR] Job YAML not found: $JOB_YAML"
    exit 1
fi

# ---- Helper: extract value from YAML by key ----
_yaml_val() {
    local key="$1"
    grep -E "^${key}:" "$JOB_YAML" | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"' || echo ""
}

# ---- Parse job params ----
MODEL=$(_yaml_val "model")
DATASET=$(_yaml_val "dataset")
SPLIT_MODE=$(_yaml_val "split_mode")
EPOCHS=$(_yaml_val "epochs")
BATCH_SIZE=$(_yaml_val "batch_size")
MAX_GRAPHS=$(_yaml_val "max_graphs")
NUM_WORKERS=$(_yaml_val "num_workers")
DEVICE=$(_yaml_val "device")
RUN_NAME=$(_yaml_val "run_name")
LAMBDA_BC=$(_yaml_val "lambda_bc")
LAMBDA_LINK=$(_yaml_val "lambda_link")
PACKAGE_ARTIFACT=$(_yaml_val "package_artifact")
SAVE_BEST=$(_yaml_val "save_best_model")
SAVE_LAST=$(_yaml_val "save_last_model")

echo "=========================================="
echo " Job Launcher"
echo "=========================================="
echo "  Job:          ${JOB_NAME}"
echo "  Timestamp:    ${TIMESTAMP}"
echo "  Project:      ${PROJECT_DIR}"
echo "=========================================="
echo ""

# ---- 1. Environment info ----
echo "=========================================="
echo " [1/5] Environment"
echo "=========================================="
echo "  Python:"
which python
python --version 2>&1 || true

echo ""
echo "  Conda:"
conda info --envs 2>/dev/null | grep '^\*' || echo "    (no active conda env detected, continuing)"

echo ""
echo "  PyTorch:"
python -c "import torch; print(f'  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>&1 || echo "    (torch import failed)"

echo ""
echo "  CUDA devices:"
python -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'  [{i}] {torch.cuda.get_device_name(i)}')
else:
    print('  (no CUDA devices)')
" 2>&1 || true

echo ""
echo "  nvidia-smi (head):"
nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | head -8 || echo "    (nvidia-smi not available)"

echo ""
echo "  Git commit:"
cd "$PROJECT_DIR" && git log -1 --format='  %h %s (%ci)' 2>/dev/null || echo "    (not a git repo)"

# ---- 2. Dataset check ----
echo ""
echo "=========================================="
echo " [2/5] Dataset check"
echo "=========================================="
bash "$PROJECT_DIR/server_ops/check_dataset.sh" || {
    echo "[WARN] Dataset check failed — continuing anyway."
}

# ---- 3. Build command ----
echo ""
echo "=========================================="
echo " [3/5] Launching training"
echo "=========================================="

LOG_DIR="${PROJECT_DIR}/logs/remote"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${JOB_NAME}_${TIMESTAMP}.log"

CMD_EXTRA=""
[ -n "$MAX_GRAPHS" ] && [ "$MAX_GRAPHS" != "null" ] && CMD_EXTRA="$CMD_EXTRA --max-graphs $MAX_GRAPHS"
[ -n "$EPOCHS" ] && [ "$EPOCHS" != "null" ] && CMD_EXTRA="$CMD_EXTRA --epochs $EPOCHS"
[ -n "$BATCH_SIZE" ] && [ "$BATCH_SIZE" != "null" ] && CMD_EXTRA="$CMD_EXTRA --batch-size $BATCH_SIZE"
[ -n "$NUM_WORKERS" ] && [ "$NUM_WORKERS" != "null" ] && CMD_EXTRA="$CMD_EXTRA --num-workers $NUM_WORKERS"
[ -n "$SPLIT_MODE" ] && [ "$SPLIT_MODE" != "null" ] && CMD_EXTRA="$CMD_EXTRA --split-mode $SPLIT_MODE"
[ -n "$DEVICE" ] && [ "$DEVICE" != "null" ] && CMD_EXTRA="$CMD_EXTRA --device $DEVICE"
[ -n "$RUN_NAME" ] && [ "$RUN_NAME" != "null" ] && CMD_EXTRA="$CMD_EXTRA --run-name $RUN_NAME"
[ -n "$LAMBDA_BC" ] && [ "$LAMBDA_BC" != "null" ] && CMD_EXTRA="$CMD_EXTRA --lambda-bc $LAMBDA_BC"
[ -n "$LAMBDA_LINK" ] && [ "$LAMBDA_LINK" != "null" ] && CMD_EXTRA="$CMD_EXTRA --lambda-link $LAMBDA_LINK"

echo "  Training command:"
echo "    python train_baseline.py --model ${MODEL} ${CMD_EXTRA}"
echo ""
echo "  Log file: ${LOG_FILE}"

# ---- 4. Run training ----
echo ""
echo "=========================================="
echo " [4/5] Training"
echo "=========================================="

set +e  # allow errors to be captured
python -u train_baseline.py \
    --model "$MODEL" \
    $CMD_EXTRA \
    > "$LOG_FILE" 2>&1

TRAIN_EXIT_CODE=$?
set -e

echo ""
echo "  Training exit code: ${TRAIN_EXIT_CODE}"
echo "  Log tail (last 20 lines):"
tail -20 "$LOG_FILE"

# ---- 5. Package artifact ----
echo ""
echo "=========================================="
echo " [5/5] Artifact packaging"
echo "=========================================="

# Capture RUN_DIR from training output (printed by train_baseline.py as RUN_DIR=<path>)
# This is robust against parallel execution because each job has its own LOG_FILE.
RUN_DIR=$(grep -oP '^RUN_DIR=\K.+' "$LOG_FILE" | tail -1 || true)

if [ -z "$RUN_DIR" ]; then
    echo "[ERROR] Could not determine RUN_DIR from training output."
    echo "        train_baseline.py must print 'RUN_DIR=<output_dir>' for packaging."
    echo "        Skipping artifact packaging to avoid packaging wrong directory."
    echo ""
    echo "  Training log: ${LOG_FILE}"
    echo "  Exit code: ${TRAIN_EXIT_CODE}"
    echo ""
    echo "=========================================="
    echo " Job finished (no artifact): ${JOB_NAME}"
    echo "=========================================="
    exit $TRAIN_EXIT_CODE
fi

if [ "$PACKAGE_ARTIFACT" = "true" ] && [ -d "$RUN_DIR" ]; then
    INCLUDE_LAST_FLAG=""
    if [ "$SAVE_LAST" = "true" ]; then
        INCLUDE_LAST_FLAG="--include-last"
    fi
    bash "$PROJECT_DIR/server_ops/package_results.sh" "$RUN_DIR" "$JOB_NAME" $INCLUDE_LAST_FLAG
elif [ "$PACKAGE_ARTIFACT" = "true" ] && [ ! -d "$RUN_DIR" ]; then
    echo "  [WARN] Run directory not found: ${RUN_DIR} — skipping artifact packaging."
else
    echo "  Artifact packaging disabled (package_artifact != true)."
fi

echo ""
echo "=========================================="
echo " Job finished: ${JOB_NAME}"
echo " Log: ${LOG_FILE}"
echo " Exit code: ${TRAIN_EXIT_CODE}"
echo "=========================================="

exit $TRAIN_EXIT_CODE

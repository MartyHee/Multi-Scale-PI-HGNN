#!/usr/bin/env bash
# =============================================================================
# check_dataset.sh — Quick smoke-test dataset integrity check
#
# Usage:
#   bash server_ops/check_dataset.sh
#
# Checks that the canonical dataset directory exists and contains the
# expected files.  Does NOT do a full scan — only stat + head operations.
#
# Dataset expected:
#   processed/hetero_graph_dataset_v2/
#     ├── metadata.json
#     ├── schema.json
#     ├── index.csv
#     ├── feature_stats.json
#     ├── splits/
#     └── graphs/
# =============================================================================
set -euo pipefail

DATA_DIR="processed/hetero_graph_dataset_v2"

echo "=========================================="
echo " Dataset check: ${DATA_DIR}"
echo "=========================================="

# 1. Root directory
if [ -d "$DATA_DIR" ]; then
    echo "  [ok]  Directory exists: ${DATA_DIR}"
else
    echo "  [FAIL] Directory NOT found: ${DATA_DIR}"
    echo ""
    echo "  --> Expected at: $(pwd)/${DATA_DIR}"
    echo "  --> Run the dataset builder first, or copy the pre-built dataset."
    exit 1
fi

# 2. Required files
REQUIRED_FILES=(
    "metadata.json"
    "schema.json"
    "index.csv"
    "feature_stats.json"
)

for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "${DATA_DIR}/${f}" ]; then
        echo "  [ok]  ${f}  ($(stat -c%s "${DATA_DIR}/${f}" 2>/dev/null || stat -f%z "${DATA_DIR}/${f}" 2>/dev/null) bytes)"
    else
        echo "  [WARN] ${f} not found"
    fi
done

# 3. Splits directory
if [ -d "${DATA_DIR}/splits" ]; then
    echo "  [ok]  splits/"
    for sf in "${DATA_DIR}/splits/"*.json; do
        [ -f "$sf" ] && echo "        $(basename "$sf")  ($(stat -c%s "$sf" 2>/dev/null || stat -f%z "$sf" 2>/dev/null) bytes)"
    done
else
    echo "  [WARN] splits/ not found"
fi

# 4. Graphs directory — check first few entries
if [ -d "${DATA_DIR}/graphs" ]; then
    GRAPH_COUNT=$(find "${DATA_DIR}/graphs" -maxdepth 2 -name "*.pt" | wc -l)
    SAMPLE_DIRS=$(find "${DATA_DIR}/graphs" -maxdepth 1 -type d | wc -l)
    echo "  [ok]  graphs/  (${SAMPLE_DIRS} sample dirs, ~${GRAPH_COUNT} .pt files)"
    # Show first 3 sample dirs
    echo "        Sample directories:"
    find "${DATA_DIR}/graphs" -maxdepth 1 -type d | sort | head -4 | while read -r d; do
        if [ "$d" != "${DATA_DIR}/graphs" ]; then
            count=$(find "$d" -maxdepth 1 -name "*.pt" | wc -l)
            echo "          $(basename "$d")  (${count} graphs)"
        fi
    done
else
    echo "  [FAIL] graphs/ not found"
    exit 1
fi

# 5. Quick index.csv row count
if [ -f "${DATA_DIR}/index.csv" ]; then
    ROWS=$(tail -n +2 "${DATA_DIR}/index.csv" | wc -l)
    echo "  [ok]  index.csv: ${ROWS} rows (excl. header)"
fi

echo ""
echo "=========================================="
echo " Dataset check complete."
echo "=========================================="

#!/usr/bin/env bash
# COMPATIBILITY WRAPPER -> run_job.sh remote_jobs/three_topology_v2/mlp.yaml
# Prefer: CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/mlp.yaml
# Requires CUDA_VISIBLE_DEVICES (GPU part); CPU subtasks hide CUDA themselves.
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES=<gpu> (GPU A) before running}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_job.sh" "$HERE/../../remote_jobs/three_topology_v2/mlp.yaml"

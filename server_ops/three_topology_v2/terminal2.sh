#!/usr/bin/env bash
# COMPATIBILITY WRAPPER -> run_job.sh remote_jobs/three_topology_v2/hgt_global_seed20261117.yaml
# Prefer: CUDA_VISIBLE_DEVICES=<gpu> bash server_ops/three_topology_v2/run_job.sh remote_jobs/three_topology_v2/hgt_global_seed20261117.yaml
set -euo pipefail
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES=<gpu> (GPU B) before running}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_job.sh" "$HERE/../../remote_jobs/three_topology_v2/hgt_global_seed20261117.yaml"

# =============================================================================
# server_config.example.sh — copy to server_config.sh and edit ONCE.
# server_config.sh is gitignored, so `git pull` will not overwrite it.
#
#   cp server_ops/three_topology_fe_v2/server_config.example.sh \
#      server_ops/three_topology_fe_v2/server_config.sh
#
# Holds ONLY server-wide paths and threads.  GPU selection stays at command
# time (CUDA_VISIBLE_DEVICES=...), so there is one source of truth per launch.
# =============================================================================

# Uploaded raw package root (contains samples.jsonl + dataset_manifest.json)
PACKAGE_ROOT="/home/miniconda/Bishe/raw_data/three_topology_fe_v2_9216"

# Uploaded/derived processed root (splits, stats; no response copy)
PROCESSED_ROOT="/home/miniconda/Bishe/Multi-Scale-PI-HGNN/processed/three_topology_fe_v2_9216"

# Python of the pi_hgnn env
PYTHON="/home/miniconda/miniconda3/envs/pi_hgnn/bin/python"

# CPU threads per task (also exported as OMP/MKL/OPENBLAS)
THREADS=4

# Shared-cache read path (formal requires 1). Shard LRU per process; 32 runs
# ≈ 32 x (6-17 MB compressed) per template, bounded and configurable.
USE_SHARED=1
SHARD_CACHE=32

# Optional: override project root (default: derived from script location)
# PROJECT_ROOT="/home/miniconda/Bishe/Multi-Scale-PI-HGNN"

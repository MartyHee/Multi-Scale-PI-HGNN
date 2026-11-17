# =============================================================================
# server_config.example.sh — copy to server_config.sh and edit ONCE.
# server_config.sh is gitignored, so `git pull` will not overwrite it.
#
#   cp server_ops/three_topology_v2/server_config.example.sh \
#      server_ops/three_topology_v2/server_config.sh
#
# This file holds ONLY server-wide settings (data root, Python, threads).
# Task model/seed/batch/protocol live in remote_jobs/three_topology_v2/*.yaml.
# The GPU is selected per command via CUDA_VISIBLE_DEVICES=... (not stored here),
# so there is a single source of truth per launch.
# =============================================================================

# Data package root (contains data/processed/three_topology_graph_v1/samples.jsonl)
DATA_ROOT="/home/miniconda/Bishe/data/three_topology_graph_v1"

# Python of the pi_hgnn env
PYTHON="/home/miniconda/miniconda3/envs/pi_hgnn/bin/python"

# CPU threads per task
THREADS=4

# Optional: override project root (default: derived from script location)
# PROJECT_ROOT="/home/miniconda/Bishe/Multi-Scale-PI-HGNN"

#!/usr/bin/env bash
# DEPRECATED — the first batch now runs in FOUR independent terminals.
# Use: terminal1.sh (GPU A: MLP x3 + Ridge/HistGB), terminal2.sh (GPU B),
#      terminal3.sh (GPU C), terminal4.sh (GPU D).
# This file is kept only to avoid stale 4-GPU background scheduling.
echo "run_first_batch.sh is deprecated. Run terminal1.sh..terminal4.sh in four terminals." >&2
exit 2

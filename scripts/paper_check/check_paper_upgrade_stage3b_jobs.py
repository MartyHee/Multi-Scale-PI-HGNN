# =============================================================================
# check_paper_upgrade_stage3b_jobs.py — Stage Paper-3B job/config validation
#
# Validates (NO training, NO model instantiation):
#   1. run_job.sh-style grep extraction of top-level keys for the 5 Stage-3B
#      jobs (random-anchor seed2026/seed3407, gated seed3407, hgt_global seed42,
#      edge_aware_mpnn seed42).
#   2. train_baseline.load_baseline_configs() resolution for the two MS-HGT
#      variants — confirms anchor_mode/anchor_seed and the coordinate-gated
#      default.
#
# Usage (local Windows interpreter):
#   KMP_DUPLICATE_LIB_OK=TRUE D:\\CodeData\\software\\Anaconda\\Anaconda3\\envs\\llm\\python.exe \\
#     scripts\\paper_check\\check_paper_upgrade_stage3b_jobs.py
#
# Expected output — all five jobs extract cleanly and resolve to the right
# model config:
#   ms_hgt          -> model_config_key='ms_hgt'            (coordinate gated)
#   ms_hgt_random_anchor -> model_config_key='ms_hgt_random_anchor'
#                        anchor_mode=random_fixed, anchor_seed=42
# =============================================================================

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_baseline import load_baseline_configs  # noqa: E402

JOBS = [
    "paper_ms_hgt_random_anchor_seed2026.yaml",
    "paper_ms_hgt_random_anchor_seed3407.yaml",
    "paper_ms_hgt_gated_seed3407.yaml",
    "paper_hgt_global_mesh_seed42.yaml",
    "paper_edge_aware_mpnn_seed42.yaml",
]
JOB_DIR = PROJECT_ROOT / "remote_jobs" / "paper_upgrade"


def yaml_val(text: str, key: str) -> str:
    """Mirror run_job.sh _yaml_val grep extraction."""
    m = re.search(rf"^{key}:\s*(.+)$", text, flags=re.MULTILINE)
    if not m:
        return ""
    return m.group(1).strip().strip('"')


def main() -> int:
    failures = 0

    print("=" * 78)
    print("Stage Paper-3B — job yaml parse (run_job.sh grep logic)")
    print("=" * 78)
    for job in JOBS:
        path = JOB_DIR / job
        if not path.is_file():
            print(f"[FAIL] missing: {job}")
            failures += 1
            continue
        text = path.read_text(encoding="utf-8")
        model = yaml_val(text, "model")
        seed = yaml_val(text, "seed")
        batch = yaml_val(text, "batch_size")
        run_name = yaml_val(text, "run_name")
        split = yaml_val(text, "split_mode")
        ok = (
            model
            and seed
            and batch
            and run_name
            and split == "by_sample"
        )
        # run_name must embed the seed so config_resolved/run_dir record it
        seed_ok = seed in run_name
        print(
            f"  {'[OK] ' if ok and seed_ok else '[FAIL]'} {job}\n"
            f"        model={model:<22} seed={seed:<6} batch={batch:<3} "
            f"split={split} run_name={run_name} "
            f"(seed-in-run_name={seed_ok})"
        )
        if not (ok and seed_ok):
            failures += 1

    print()
    print("=" * 78)
    print("Stage Paper-3B — config resolution (load_baseline_configs)")
    print("=" * 78)
    for model in ["ms_hgt", "ms_hgt_random_anchor"]:
        cfg = load_baseline_configs(str(PROJECT_ROOT / "configs"), model)
        mc = cfg.get("model_config", {})
        anchor_mode = mc.get("anchor_mode", "(default=coordinate)")
        anchor_seed = mc.get("anchor_seed", "(none)")
        fusion = mc.get("fusion_method")
        per_layer = mc.get("fusion_per_layer")
        n_seg = mc.get("n_segments")
        key = cfg.get("model_config_key")
        print(f"  {model:<22} -> model_config_key={key}")
        print(
            f"        anchor_mode={anchor_mode:<22} anchor_seed={anchor_seed:<12} "
            f"fusion={fusion} per_layer={per_layer} n_segments={n_seg}"
        )
        if model == "ms_hgt":
            ok = anchor_mode == "(default=coordinate)" and fusion == "gated_residual"
        else:
            ok = anchor_mode == "random_fixed" and str(anchor_seed) == "42"
        if not ok:
            failures += 1
            print("        [FAIL] unexpected resolved config")

    print()
    if failures:
        print(f"RESULT: {failures} FAILURE(S)")
        return 1
    print("RESULT: ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

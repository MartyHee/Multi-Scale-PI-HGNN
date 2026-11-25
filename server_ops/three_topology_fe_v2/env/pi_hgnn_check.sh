#!/usr/bin/env bash
# Print the actual server environment and GPU visibility (read-only).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/_load_config.sh"
echo "python: $PYTHON"
"$PYTHON" - <<'PY'
import sys, torch, torch_geometric, numpy, sklearn, yaml
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("numpy", numpy.__version__, "sklearn", sklearn.__version__, "pyyaml", yaml.__version__)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"gpu{i}", p.name, round(p.total_memory / 1024 ** 3, 2), "GB")
PY
echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}'"
nvidia-smi -L 2>/dev/null || true

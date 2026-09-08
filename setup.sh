#!/usr/bin/env bash
# Rebuild a working environment for this project from a clean checkout.
#
#   ./setup.sh [--data-root <dir>] [--venv <dir>]
#
# Creates the virtualenv, installs the pinned dependencies, locates the
# dataset, and verifies the committed artifacts. Everything else the project
# needs is in this repository.
#
# Not created here (they are training outputs, and the README says so):
#   - per-epoch checkpoints and descriptor caches from the cross-validation runs
#   - the two source caches behind artifacts/teacher_cache
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${REPO}/venv"
DATA_ROOT="/var/tmp/luli38se-geomatch/data/geo_dataset"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --venv)      VENV="$2";      shift 2 ;;
    -h|--help)   sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
fail() { printf '\033[31mFAIL\033[0m %s\n' "$1" >&2; exit 1; }

step "Python"
command -v python3 >/dev/null || fail "python3 not found"
python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"need Python >= 3.10, found {sys.version.split()[0]}")
print(f"python {sys.version.split()[0]} (trained on 3.12.3)")
PY

step "Virtualenv at ${VENV}"
if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
  echo "created"
else
  echo "reusing existing venv"
fi
PY_BIN="${VENV}/bin/python"
"${PY_BIN}" -m pip install --upgrade pip -q

step "Dependencies (pinned in requirements.txt)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  "${PY_BIN}" -m pip install -q torch==2.12.1 torchvision==0.27.1 --index-url "${TORCH_INDEX}"
else
  echo "no NVIDIA GPU detected -- installing the default torch build."
  echo "Training and predict_holdout.py both require a bfloat16-capable CUDA GPU."
  "${PY_BIN}" -m pip install -q torch==2.12.1 torchvision==0.27.1
fi
"${PY_BIN}" -m pip install -q -r "${REPO}/requirements.txt"
"${PY_BIN}" - <<'PY'
import numpy, pandas, PIL, torch, torchvision, matplotlib
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | "
      f"numpy {numpy.__version__} | pandas {pandas.__version__}")
if torch.cuda.is_available():
    print(f"cuda {torch.version.cuda} | {torch.cuda.get_device_name(0)} | "
          f"bf16 {torch.cuda.is_bf16_supported()}")
else:
    print("cuda unavailable -- inference and training will not run")
PY

step "Dataset"
if [[ -d "${DATA_ROOT}/train" && -d "${DATA_ROOT}/holdout_public" ]]; then
  echo "${DATA_ROOT}"
  echo "  train          $(find "${DATA_ROOT}/train" -type f | wc -l) images (expect 11758)"
  echo "  holdout_public $(find "${DATA_ROOT}/holdout_public" -type f | wc -l) images (expect 2400)"
else
  echo "NOT FOUND at ${DATA_ROOT}"
  echo "  The dataset is not redistributed with this repository. Place it so that"
  echo "  <data-root>/train/ and <data-root>/holdout_public/ exist, then re-run"
  echo "  with --data-root <dir>, or pass --data-root to the scripts directly."
fi

step "Committed artifacts"
cd "${REPO}"
PYTHONPATH=src "${PY_BIN}" src/build_teacher_cache.py --verify
PYTHONPATH=src "${PY_BIN}" - <<'PY'
import hashlib, sys
from pathlib import Path
sys.path.insert(0, "src")
import torch
from model import count_trainable_parameters
from retrieval_model import GeoCPRegNetRetrieval

ckpt = torch.load("model/model_final.pt", map_location="cpu", weights_only=False)
model = GeoCPRegNetRetrieval(local_features=64, local_grid_size=4)
model.load_state_dict(ckpt["model"], strict=True)
n = count_trainable_parameters(model)
print(f"model/model_final.pt   {n:,} parameters (limit 5,000,000) -- "
      f"{'OK' if n <= 5_000_000 else 'OVER LIMIT'}")

for fold in range(5):
    path = Path(f"artifacts/ssl_backbone/fold_{fold}.pt")
    c = torch.load(path, map_location="cpu", weights_only=False)
    assert c["format"] == "geomatch-regnet-cp-ssl-byol-backbone-v1", path
    assert int(c["fold"]) == fold, path
print("artifacts/ssl_backbone 5 BYOL backbones, 160 epochs each")

rows = sum(1 for _ in open("predictions.csv")) - 1
print(f"predictions.csv        {rows} rows")
PY

step "Ready"
cat <<EOF2
Activate:   source ${VENV}/bin/activate

Reproduce the submitted predictions (~5 min on one GPU):
  python src/predict_holdout.py --model model/model_final.pt \\
      --data-root ${DATA_ROOT} --output predictions_recheck.csv

Retrain the submitted model (~70 min, backbone already committed):
  python src/full_data_finetune.py --images ${DATA_ROOT}/train --resume

Cross-validation (adds ~2h40m per fold for BYOL if you rebuild the backbones):
  python src/country_aware_retrieval_finetune.py \\
      --config configs/final_recipe.json --folds 0 1 2 3 4 --seed-base 220517

Redraw the figures (needs only the dataset, for the example panel):
  python images/make_figures.py
EOF2

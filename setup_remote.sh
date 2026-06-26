#!/usr/bin/env bash
# setup_remote.sh — bootstrap the project environment on a remote GPU server.
# Run once after cloning the repo:  bash setup_remote.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-python3}
echo "[1/4] creating virtualenv (.venv) with $($PY --version)"
$PY -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel

echo "[2/4] installing dependencies"
# On Linux+NVIDIA, the default PyPI torch wheels are CUDA-enabled, so this is
# usually all you need. If you must pin a specific CUDA build, comment the line
# below and instead run, e.g.:
#   .venv/bin/pip install torch==2.8.0 torchvision==0.23.0 \
#       --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt

echo "[3/4] (optional) Kaggle auth"
# The datasets used are public and download anonymously via kagglehub. If a
# download is rate-limited or private, drop your token at ~/.kaggle/kaggle.json
# (chmod 600) or export KAGGLE_USERNAME / KAGGLE_KEY.

echo "[4/4] verifying environment + GPU"
.venv/bin/python check_env.py

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then e.g.:  python check_env.py   ||   python run_experiment.py --seeds 0 --quick"

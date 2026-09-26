#!/usr/bin/env bash
# run_retinareach.sh — the RetinaReach build, one full pipeline per training seed,
# then the multi-seed capability table.
#
#   tmux new -s rr
#   cd ~/Oculomics && B=<BRSET root> M=<mBRSET root> bash run_retinareach.sh 0 1 2 3 4 2>&1 | tee -a exp_retinareach/sweep.log
#
# Per seed (train_retinareach.py): BRSET-trained MobileNetV4 trunk + DR/edema heads,
# AdaBN profiles for both cameras from their unlabelled pools, systemic linear
# probes on the calibrated mBRSET embedding, thresholds fixed on home val, the
# capability gate against the intake-form metadata model on the same patients,
# and the calibration-size sweep. The patient split is fixed (--split-seed 42)
# across seeds, so every per-seed quantity is paired; summarize_retinareach.py
# turns the seeds into the noise floor and the final gate.
# Idempotent: a seed whose results.json exists is skipped.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> M=<mBRSET root> [KNOB=value ...] bash run_retinareach.sh [seed ...]

Seeds default to 0 1 2 3 4 (the plan's multi-seed noise floor needs >= 5).
Every knob is an environment variable:

  required
    B              BRSET root  (labels csv + fundus_photos/)
    M              mBRSET root (labels_mbrset.csv + images/)
  outputs
    OUT            per-seed results + summary   (default exp_retinareach) -> $OUT/seed<s>/
    CK             checkpoints                  (default ck_retinareach)  -> $CK/seed<s>.pt
  model / training
    STUDENT        backbone (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
    SIZE           image size                   (default 384)
    EPOCHS         stage-1 epochs               (default 30)
    WORKERS        DataLoader workers           (default 5)
    AMP            "" = trainer default, "--amp" / "--no-amp" to force
    EXTRA          extra trainer flags (default "--ema-decay 0.999"), e.g.
                   "--probe-targets hypertension insulin" or "--calib-sizes 16 64 256"

example
    B=/data/BRSET/1.0.1 M=/data/mBRSET/1.0 CUDA_VISIBLE_DEVICES=0 bash run_retinareach.sh 0 1 2 3 4
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

: "${B:?set B=<BRSET root>; see --help}"
: "${M:?set M=<mBRSET root>; see --help}"
OUT=${OUT:-exp_retinareach}
CK=${CK:-ck_retinareach}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-5}
AMP=${AMP:-}
EXTRA=${EXTRA:---ema-decay 0.999}
PY=.venv/bin/python

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
for v in B M; do [ -d "${!v}" ] || { echo "[fatal] $v root not found: ${!v}"; exit 1; }; done
$PY -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }
mkdir -p "$OUT" "$CK"

for s in "${SEEDS[@]}"; do
  if [ -f "$OUT/seed$s/results.json" ]; then echo "[skip] seed $s (results.json exists)"; continue; fi
  echo; echo "=== RetinaReach seed $s   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_retinareach.py --source-root "$B" --target-root "$M" \
      --backbone "$STUDENT" --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" \
      --seed "$s" --out "$OUT/seed$s" --ckpt "$CK/seed$s.pt" $AMP $EXTRA
done

echo; echo "=== multi-seed capability table   $(date) ==="
$PY summarize_retinareach.py --dir "$OUT"
echo "done=$(date)  ->  $OUT/summary.md"

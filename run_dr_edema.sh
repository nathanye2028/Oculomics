#!/usr/bin/env bash
# run_dr_edema.sh — the four-way none / DR only / edema only / both classifier,
# one model per training seed (train_dr_edema.py).
#
#   tmux new -s dre
#   cd ~/Oculomics && CUDA_VISIBLE_DEVICES=0 bash run_dr_edema.sh 0 1 2 2>&1 | tee -a exp_dr_edema/sweep.log
#
# Trains on B (BRSET) and, when M (mBRSET) exists, scores it as the external set.
# Idempotent: a seed whose results.json exists is skipped.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: [KNOB=value ...] bash run_dr_edema.sh [seed ...]      (seeds default to 0 1 2)

  B        training root   (default: the lab box's BRSET mirror)
  M        external root   (default: the lab box's mBRSET mirror; "" = no external set)
  TRAIN    brset | mbrset: which of the two to train on; the other is external (default brset)
  DR       any | referable: DR present = ICDR >= 1 or >= 2                    (default any)
  OUT      results     (default exp_dr_edema) -> $OUT/seed<s>/
  CK       checkpoints (default ck_dr_edema)  -> $CK/seed<s>.pt
  STUDENT  backbone    (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
  SIZE     image size  (default 384)      EPOCHS (default 30)      WORKERS (default 5)
  EXTRA    extra trainer flags (default "--ema-decay 0.999"), e.g. "--threshold sens"
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DATA=${DATA:-/data/users4/nshaik3/Datasets}
B=${B:-$DATA/BRSET/physionet.org/files/brazilian-ophthalmological/1.0.1}
M=${M-$DATA/mBRSET/physionet.org/files/mbrset/1.0}
TRAIN=${TRAIN:-brset}
DR=${DR:-any}
OUT=${OUT:-exp_dr_edema}
CK=${CK:-ck_dr_edema}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-30}
WORKERS=${WORKERS:-5}
EXTRA=${EXTRA:---ema-decay 0.999}
PY=.venv/bin/python

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
case "$TRAIN" in
  brset)  ROOT=$B; EXT=$M; EXT_DS=mbrset;;
  mbrset) ROOT=$M; EXT=$B; EXT_DS=brset;;
  *) echo "[fatal] TRAIN must be brset or mbrset"; exit 1;;
esac
[ -d "$ROOT" ] || { echo "[fatal] training root not found: $ROOT"; exit 1; }
EXTERNAL=()
if [ -n "$EXT" ]; then
  [ -d "$EXT" ] || { echo "[fatal] external root not found: $EXT (set it to \"\" to skip)"; exit 1; }
  EXTERNAL=(--external-root "$EXT" --external-dataset "$EXT_DS")
fi
mkdir -p "$OUT" "$CK"

for s in "${SEEDS[@]}"; do
  if [ -f "$OUT/seed$s/results.json" ]; then echo "[skip] seed$s (results.json exists)"; continue; fi
  echo; echo "=== DR/edema four-way seed$s   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_dr_edema.py --root "$ROOT" --dataset "$TRAIN" --dr-def "$DR" \
      --backbone "$STUDENT" --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" \
      --seed "$s" --out "$OUT/seed$s" --ckpt "$CK/seed$s.pt" \
      ${EXTERNAL[@]+"${EXTERNAL[@]}"} $EXTRA
done

echo; echo "=== summary ==="
$PY - "$OUT" <<'PY'
import glob, json, sys
import numpy as np
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{sys.argv[1]}/seed*/results.json"))]
for name in ("test", "external"):
    r = [x[name] for x in rows if name in x]
    if not r:
        continue
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v, ddof=1) if len(v) > 1 else 0:.3f}"
    print(f"{name} ({len(r)} seeds): accuracy {ms([x['accuracy'] for x in r])}, "
          f"balanced accuracy {ms([x['balanced_accuracy'] for x in r])}, "
          f"macro-F1 {ms([x['macro_f1'] for x in r])}, "
          f"DR AUROC {ms([x['heads']['dr']['auroc'] for x in r])}, "
          f"edema AUROC {ms([x['heads']['edema']['auroc'] for x in r])}")
PY

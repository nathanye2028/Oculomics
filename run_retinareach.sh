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
  unfamiliar camera (checked, never trained on)
    U              its root, or kaggle:andrewmvd/ocular-disease-recognition-odir5k (fetched or
                   reused with kagglehub); unset = no unfamiliar-camera check      (default "")
    U_DATASET      its schema: odir | mbrset | brset | airogs | refuge | papila       (default odir)
    U_NAME         its label in the tables                                         (default $U_DATASET)
  checks
    PREFLIGHT      1 = before any seed, a few-minute dry run (data, encodings, per-target
                   power, one GPU training step, one calibration, disk, time estimate);
                   the sweep stops if it fails                                     (default 1)
  shared trunk vs per-target models (plan: "measure, don't assume")
    PERTARGET      space-separated probe targets that ALSO get their own fully fine-tuned
                   train_mbrset.py model per seed, on the same handheld split
                   (--split-file), e.g. "hypertension insulin"                      (default "")
    PT_EXTRA       extra flags for those runs                     (default "--ema-decay 0.999")
  controls
    SHUFFLE        1 = also run the first seed with permuted source labels into
                   $OUT/shuffle_seed<s>/ (every source-target AUROC must sit at chance) (default 0)
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
    # with ODIR-5K as the unfamiliar camera and the shuffled-label control
    B=... M=... U=kaggle:andrewmvd/ocular-disease-recognition-odir5k SHUFFLE=1 bash run_retinareach.sh 0 1 2 3 4
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
U=${U:-}
U_DATASET=${U_DATASET:-odir}
U_NAME=${U_NAME:-$U_DATASET}
SHUFFLE=${SHUFFLE:-0}
PREFLIGHT=${PREFLIGHT:-1}
PERTARGET=${PERTARGET:-}
PT_EXTRA=${PT_EXTRA:---ema-decay 0.999}
PY=.venv/bin/python

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2 3 4)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
for v in B M; do [ -d "${!v}" ] || { echo "[fatal] $v root not found: ${!v}"; exit 1; }; done
$PY -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }
mkdir -p "$OUT" "$CK"

UNF=()
if [ -n "$U" ]; then
  case "$U" in
    # kagglehub prints its version warning and progress lines on stdout too: keep only the path (last line)
    kaggle:*) U=$($PY -c "import sys, kagglehub; print(kagglehub.dataset_download(sys.argv[1]))" "${U#kaggle:}" 2>/dev/null | tail -n 1) \
                || { echo "[fatal] kagglehub download failed for the unfamiliar camera"; exit 1; };;
  esac
  [ -d "$U" ] || { echo "[fatal] unfamiliar-camera root not found: $U"; exit 1; }
  echo "[info] unfamiliar camera: $U_NAME ($U_DATASET) @ $U"
  UNF=(--unfamiliar "$U" "$U_DATASET" "$U_NAME")
fi

run() {  # run <dir name> <seed> [extra flags...]
  local name=$1 s=$2; shift 2
  if [ -f "$OUT/$name/results.json" ]; then echo "[skip] $name (results.json exists)"; return 0; fi
  echo; echo "=== RetinaReach $name   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_retinareach.py --source-root "$B" --target-root "$M" \
      --backbone "$STUDENT" --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" \
      --seed "$s" --out "$OUT/$name" --ckpt "$CK/$name.pt" ${UNF[@]+"${UNF[@]}"} $AMP $EXTRA "$@"
}

if [ "$PREFLIGHT" = 1 ]; then
  echo; echo "=== preflight   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_retinareach.py --source-root "$B" --target-root "$M" \
      --backbone "$STUDENT" --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" \
      --seed "${SEEDS[0]}" --out "$OUT/preflight" --ckpt "$CK/preflight.pt" \
      ${UNF[@]+"${UNF[@]}"} $AMP $EXTRA --preflight \
    || { echo "[fatal] preflight failed: fix the items above (PREFLIGHT=0 skips it)"; exit 1; }
fi

pertarget() {  # pertarget <seed>: one fine-tuned model per PERTARGET task, RetinaReach's split
  local s=$1 t pt
  for t in $PERTARGET; do
    pt="$OUT/seed$s/pertarget_$t.json"
    if [ -f "$pt" ]; then echo "[skip] per-target $t seed $s (exists)"; continue; fi
    echo; echo "=== per-target $t seed $s (fine-tuned, same handheld split)   $(date) ==="
    # shellcheck disable=SC2086
    $PY -u train_mbrset.py --dataset mbrset --root "$M" --task "$t" \
        --split-file "$OUT/seed$s/split_handheld.csv" --backbone "$STUDENT" --no-gcg \
        --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" --seed "$s" \
        --covariate-baseline --ckpt-dir "$CK/pertarget" --run-name "${t}_seed$s" \
        --results-json "$pt" $AMP $PT_EXTRA
  done
}

for s in "${SEEDS[@]}"; do
  run "seed$s" "$s"
  [ -n "$PERTARGET" ] && pertarget "$s"
done
if [ "$SHUFFLE" = 1 ]; then
  run "shuffle_seed${SEEDS[0]}" "${SEEDS[0]}" --shuffle-source-labels
fi

echo; echo "=== multi-seed capability table   $(date) ==="
$PY summarize_retinareach.py --dir "$OUT"
# poster figures need matplotlib, which the training venv may not have
for py in "$PY" python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "import matplotlib, pandas" 2>/dev/null; then
    "$py" plot_retinareach.py --dir "$OUT"; break
  fi
done || true
[ -d "$OUT/figures" ] || echo "[info] no matplotlib here: run 'python3 plot_retinareach.py --dir $OUT' where it is installed"
echo "done=$(date)  ->  $OUT/summary.md"

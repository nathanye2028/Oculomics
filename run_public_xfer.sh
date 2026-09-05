#!/usr/bin/env bash
# run_public_xfer.sh — glaucoma / AMD on the public fundus sets, with the SAME paired
# transfer design as the DR headline (run_kd_xfer.sh): train ctrl / teacher / kd on one
# source set, score zero-shot + AdaBN on a primary external set, then score every
# finished ctrl/kd checkpoint on the extra external sets with score_external.py.
#
#   glaucoma:  SRC=<AIROGS> SRC_DATASET=airogs TASK=glaucoma EXT=<REFUGE> EXT_DATASET=refuge \
#              MORE="papila=<PAPILA> odir=<ODIR-5K>" bash run_public_xfer.sh 0 1 2
#   AMD:       SRC=<ODIR-5K> SRC_DATASET=odir TASK=amd EXT=<BRSET> EXT_DATASET=brset bash run_public_xfer.sh 0 1 2
#
# No public smartphone glaucoma/AMD set exists, so the claim here is transfer ACROSS
# CAMERAS AND POPULATIONS (tabletop -> tabletop), not tabletop -> phone.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: SRC=<root> SRC_DATASET=<name> TASK=glaucoma|amd EXT=<root> EXT_DATASET=<name> \
       [MORE="<name>=<root> ..."] [KNOB=value ...] bash run_public_xfer.sh [seed ...]

  required
    SRC / SRC_DATASET   training set root + schema (airogs | refuge | papila | odir | brset)
    EXT / EXT_DATASET   primary external test set (scored inside every training run, with AdaBN)
    TASK                glaucoma | amd
    Any root may be "kaggle:<owner>/<dataset>": it is downloaded (or reused) with kagglehub
    into $KAGGLEHUB_CACHE (default ~/.cache/kagglehub) and replaced by the local path, e.g.
    SRC=kaggle:deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2  EXT=kaggle:andrewmvd/ocular-disease-recognition-odir5k
  optional
    MORE                extra external sets, space-separated "<dataset>=<root>" pairs, scored
                        after training with score_external.py (zero-shot + AdaBN)
    OUT / CK            results (default exp_<TASK>) and checkpoints (default ck_<TASK>)
    STUDENT, TEACHER, SIZE, EPOCHS, WORKERS, KD_ALPHA, KD_TEMP, EXTRA, AMP
                        forwarded to run_kd_xfer.sh (see: bash run_kd_xfer.sh --help)
    PAPILA_SUSPECT      exclude (default) | positive | negative, for any PAPILA set

Pre-flight: public_fundus.py --inspect --strict on SRC, EXT and every MORE set.
Seeds default to 0 1 2. Idempotent: existing JSONs are skipped.
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY=.venv/bin/python

: "${SRC:?set SRC=<training set root>; see --help}"
: "${SRC_DATASET:?set SRC_DATASET=airogs|refuge|papila|odir|brset}"
: "${EXT:?set EXT=<primary external root>}"
: "${EXT_DATASET:?set EXT_DATASET=airogs|refuge|papila|odir|brset}"
: "${TASK:?set TASK=glaucoma|amd}"
MORE=${MORE:-}
OUT=${OUT:-exp_$TASK}
CK=${CK:-ck_$TASK}
PAPILA_SUSPECT=${PAPILA_SUSPECT:-exclude}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
WORKERS=${WORKERS:-5}
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }

kaggle_root() {  # kaggle_root <root> -> local dir; "kaggle:<owner>/<dataset>" is fetched/reused via kagglehub
  case "$1" in
    kaggle:*) $PY - "${1#kaggle:}" <<'PY'
import sys, kagglehub
print(kagglehub.dataset_download(sys.argv[1]))
PY
    ;;
    *) printf '%s\n' "$1";;
  esac
}
SRC=$(kaggle_root "$SRC") || { echo "[fatal] could not fetch SRC"; exit 1; }
EXT=$(kaggle_root "$EXT") || { echo "[fatal] could not fetch EXT"; exit 1; }
resolved_more=""
for pair in $MORE; do
  r=$(kaggle_root "${pair#*=}") || { echo "[fatal] could not fetch ${pair#*=}"; exit 1; }
  resolved_more="$resolved_more ${pair%%=*}=$r"
done
MORE=${resolved_more# }
echo "[info] SRC=$SRC_DATASET @ $SRC"; echo "[info] EXT=$EXT_DATASET @ $EXT"; [ -n "$MORE" ] && echo "[info] MORE=$MORE"

inspect() {  # inspect <dataset> <root>
  local ds=$1 root=$2
  case "$ds" in
    brset)  $PY brset_dataset.py --csv "$(ls "$root"/labels_brset.csv "$root"/labels.csv 2>/dev/null | head -1)" --inspect | grep -E "^  (amd|TASK)" || true ;;
    mbrset) echo "[skip] no inspector for mbrset here" ;;
    *)      $PY public_fundus.py --root "$root" --dataset "$ds" --task "$TASK" --papila-suspect "$PAPILA_SUSPECT" --strict ;;
  esac
}
echo "=== pre-flight ==="
inspect "$SRC_DATASET" "$SRC"; inspect "$EXT_DATASET" "$EXT"
for pair in $MORE; do inspect "${pair%%=*}" "${pair#*=}"; done

echo; echo "=== train: $SRC_DATASET -> $EXT_DATASET  task=$TASK  seeds=${SEEDS[*]}   $(date) ==="
B="$SRC" M="$EXT" TRAIN_DATASET="$SRC_DATASET" EXT_DATASET="$EXT_DATASET" TASK="$TASK" \
  OUT="$OUT" CK="$CK" STUDENT="$STUDENT" SIZE="$SIZE" WORKERS="$WORKERS" bash run_kd_xfer.sh "${SEEDS[@]}"

if [ -n "$MORE" ]; then
  echo; echo "=== extra external sets   $(date) ==="
  for pair in $MORE; do
    ds=${pair%%=*}; root=${pair#*=}
    for s in "${SEEDS[@]}"; do
      for cond in ctrl kd; do
        out="$OUT/${cond}_seed${s}_on_${ds}.json"
        if [ -f "$out" ]; then echo "[skip] $out"; continue; fi
        $PY score_external.py --ckpt "$CK/${cond}_seed$s.pt" --root "$root" --dataset "$ds" --task "$TASK" \
            --bn-adapt --num-workers "$WORKERS" --papila-suspect "$PAPILA_SUSPECT" --out "$out"
      done
    done
  done
  $PY summarize_external.py --dir "$OUT" --treatment kd --control ctrl
fi
echo "done=$(date)  ->  $OUT/summary.md" $([ -n "$MORE" ] && echo "+ $OUT/summary_external.md")

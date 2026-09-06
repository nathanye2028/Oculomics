#!/usr/bin/env bash
# run_retinal_age.sh — retinal age on the BRSET healthy cohort, scored on held-out
# BRSET patients and zero-shot on mBRSET (+ label-free AdaBN). Run on the GPU box:
#
#   tmux new -s ra
#   cd ~/Oculomics && B=<BRSET root> M=<mBRSET root> bash run_retinal_age.sh 0 1 2 2>&1 | tee -a exp_retinal_age/sweep.log
#
# Per seed it trains the deployable student ($STUDENT) and, if TEACHER is set, a
# large timm backbone on the SAME patient split as a capacity reference:
#   student_seed<s>   $STUDENT (MobileNetV4-Small @384 by default)
#   teacher_seed<s>   $TEACHER (off by default; e.g. timm:convnext_small.fb_in22k_ft_in1k)
# Pre-flight: train_retinal_age.py --inspect prints the cohort (how many healthy
# images/patients survive the --healthy rule, age histogram per split) and fails
# loudly if the rule needs a column the CSV lacks (nodm needs BRSET's `diabetes`).
# Idempotent: a run whose JSON exists is skipped. Wall-clock: the healthy cohort is
# a fraction of BRSET (expect ~3-6k images), so a student run is well under an hour
# on an A100 @384 px.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> M=<mBRSET root> [KNOB=value ...] bash run_retinal_age.sh [seed ...]

  required
    B              BRSET root      (dir holding fundus_photos/ + labels_brset.csv)
    M              mBRSET root     (dir holding images/ + labels_mbrset.csv)
  cohort
    HEALTHY        nodm | dr0 | gradable | all     (default nodm: no diabetes, DR 0, adequate quality)
    EXCLUDE_PATHOLOGY  1 = also drop patients with any BRSET ophthalmic flag (default 0)
    AGE_BALANCE    1 = sample train images by 1/sqrt(age-bin frequency)          (default 0)
  outputs
    OUT            results JSONs + summary + pooled predictions   (default exp_retinal_age)
    CK             checkpoints + per-run predictions CSVs         (default ck_retinal_age)
  models / training
    STUDENT        backbone (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
    TEACHER        optional large backbone trained on the same split ("" = skip)
    TEACHER_LR     teacher learning rate                          (default 1e-4)
    SIZE           image size                                     (default 384)
    EPOCHS         epochs per run                                 (default 30)
    LOSS           l1 | huber | mse                               (default l1)
    WORKERS        DataLoader workers                             (default 5)
    AMP            "" = trainer default, "--amp" / "--no-amp" to force
    EXTRA          extra flags for every run, e.g. "--ema-decay 0.999 --min-age 18"

example
    B=/data/BRSET/1.0.1 M=/data/mBRSET/1.0 TEACHER=timm:convnext_small.fb_in22k_ft_in1k bash run_retinal_age.sh 0 1 2
USAGE
}
for a in "$@"; do case "$a" in -h|--help) usage; exit 0;; esac; done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

: "${B:?set B=<BRSET root> (dir holding fundus_photos/ + labels_brset.csv); see --help}"
: "${M:?set M=<mBRSET root> (dir holding images/ + labels_mbrset.csv); see --help}"
HEALTHY=${HEALTHY:-nodm}
EXCLUDE_PATHOLOGY=${EXCLUDE_PATHOLOGY:-0}
AGE_BALANCE=${AGE_BALANCE:-0}
OUT=${OUT:-exp_retinal_age}
CK=${CK:-ck_retinal_age}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
TEACHER=${TEACHER:-}
TEACHER_LR=${TEACHER_LR:-1e-4}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-30}
LOSS=${LOSS:-l1}
WORKERS=${WORKERS:-5}
AMP=${AMP:-}
EXTRA=${EXTRA:-}
PY=.venv/bin/python

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
mkdir -p "$OUT" "$CK"
[ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; }
[ -d "$M" ] || { echo "[fatal] mBRSET root not found: $M"; exit 1; }
$PY -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: $PY -m pip install -r requirements.txt"; exit 1; }

COMMON=(--dataset brset --root "$B" --external-test-root "$M" --external-test-dataset mbrset
        --healthy "$HEALTHY" --image-size "$SIZE" --epochs "$EPOCHS" --loss "$LOSS"
        --num-workers "$WORKERS" --bn-adapt --ckpt-dir "$CK" $AMP)
[ "$EXCLUDE_PATHOLOGY" = 1 ] && COMMON+=(--exclude-pathology)
[ "$AGE_BALANCE" = 1 ] && COMMON+=(--age-balance)

echo "=== pre-flight: cohort under --healthy $HEALTHY   $(date) ==="
# shellcheck disable=SC2086
$PY train_retinal_age.py "${COMMON[@]}" --inspect $EXTRA \
  || { echo "[fatal] cohort pre-flight failed (see above); nothing trained"; exit 1; }

run() {  # run <name> <flags...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (exists)"; return 0; fi
  echo; echo "=== $name   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_retinal_age.py "${COMMON[@]}" --run-name "$name" \
      --results-json "$OUT/$name.json" "$@" $EXTRA
}

for s in "${SEEDS[@]}"; do
  run "student_seed$s" --seed "$s" --backbone "$STUDENT"
  if [ -n "$TEACHER" ]; then
    run "teacher_seed$s" --seed "$s" --backbone "$TEACHER" --lr "$TEACHER_LR"
  fi
done

echo; echo "=== summary   $(date) ==="
$PY summarize_retinal_age.py --dir "$OUT"
echo "done=$(date)  ->  $OUT/summary.md  +  $OUT/predictions_pooled.csv"

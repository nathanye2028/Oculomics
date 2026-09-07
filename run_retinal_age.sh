#!/usr/bin/env bash
# run_retinal_age.sh — retinal age on the BRSET healthy cohort, scored on held-out
# BRSET patients and zero-shot on mBRSET (+ label-free AdaBN). Run on the GPU box:
#
#   tmux new -s ra
#   cd ~/Oculomics && mkdir -p exp_retinal_age && \
#     B=<BRSET root> M=<mBRSET root> bash run_retinal_age.sh 0 1 2 2>&1 | tee -a exp_retinal_age/sweep.log
#
# Per seed it trains the deployable student ($STUDENT) and, if TEACHER is set, a
# large timm backbone on the SAME patient split as a capacity reference:
#   student_seed<s>   $STUDENT (MobileNetV4-Small @384 by default)
#   teacher_seed<s>   $TEACHER (off by default; e.g. timm:convnext_small.fb_in22k_ft_in1k)
#   kd_seed<s>        with TEACHER (and KD=1, the default): $STUDENT distilled from teacher_seed<s>
#                     (regression KD: matches the teacher's predicted age; same split, same seed)
#   ceiling_seed<s>   CEILING=1: $STUDENT trained IN-DOMAIN on mBRSET's DR-grade-0 patients
#                     (--dataset mbrset --healthy dr0), scored on held-out mBRSET patients and
#                     on all of BRSET as the reverse transfer. This is the phone-domain ceiling:
#                     if it is far better than the transferred student, the phone images carry
#                     the age signal and the transfer is what fails.
# Every run also reports the external set DEVICE-CALIBRATED (2-fold linear fit on the
# external DR-0 patients, out-of-fold), next to zero-shot and AdaBN.
# Recipe knobs (each off by default): HEAD=ldl, TTA=1, PHONE_AUG=1, AGE_BALANCE=1, and
# MIX=1 for mixed-domain training (mBRSET DR-0 patients join training; the external numbers
# are then computed on mBRSET's held-out rows and conditions are named *_mix).
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
    CEILING        1 = also train the in-domain mBRSET DR-0 ceiling per seed      (default 0)
    RECAL_GROUP    dr0 | all: external rows the device calibration is fit on    (default dr0)
    MIX            1 = mixed-domain training: mBRSET DR-0 train/val rows join training;
                   external = mBRSET held-out rows; run names get TAG=_mix          (default 0)
    EXTRA_WEIGHT   sampling weight multiplier for the mixed-in images              (default 1.0)
  recipe
    HEAD           linear | ldl (label-distribution head)                          (default linear)
    TTA            1 = average four flip views at evaluation                       (default 0)
    PHONE_AUG      1 = smartphone-capture augmentation in training                 (default 0)
    KD             1 = with TEACHER, also distil it into the student (kd_seed<s>)  (default 1)
    KD_ALPHA       weight of the teacher-matching term                             (default 0.5)
    FEAT_W         cosine feature-matching weight                                  (default 0.0)
    TAG            suffix on condition names, e.g. _512 (MIX=1 defaults to _mix)   (default "")
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
CEILING=${CEILING:-0}
RECAL_GROUP=${RECAL_GROUP:-dr0}
MIX=${MIX:-0}
EXTRA_WEIGHT=${EXTRA_WEIGHT:-1.0}
HEAD=${HEAD:-linear}
TTA=${TTA:-0}
PHONE_AUG=${PHONE_AUG:-0}
KD=${KD:-1}
KD_ALPHA=${KD_ALPHA:-0.5}
FEAT_W=${FEAT_W:-0.0}
TAG=${TAG:-}
[ "$MIX" = 1 ] && [ -z "$TAG" ] && TAG=_mix
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
        --num-workers "$WORKERS" --bn-adapt --recal-group "$RECAL_GROUP" --ckpt-dir "$CK" $AMP)
# The ceiling swaps the roles: train on mBRSET (DR-0 patients), external = BRSET. argparse keeps
# the LAST occurrence of a repeated flag, so these override the entries in COMMON.
CEIL=(--dataset mbrset --root "$M" --external-test-root "$B" --external-test-dataset brset --healthy dr0)
[ "$EXCLUDE_PATHOLOGY" = 1 ] && COMMON+=(--exclude-pathology)
[ "$AGE_BALANCE" = 1 ] && COMMON+=(--age-balance)
COMMON+=(--head "$HEAD")
[ "$TTA" = 1 ] && COMMON+=(--tta)
[ "$PHONE_AUG" = 1 ] && COMMON+=(--phone-aug)
[ "$MIX" = 1 ] && COMMON+=(--extra-train-root "$M" --extra-train-dataset mbrset --extra-healthy dr0 \
                           --extra-weight "$EXTRA_WEIGHT")
if [ "$MIX" = 1 ] && [ "$CEILING" = 1 ]; then
  echo "[warn] CEILING=1 ignored with MIX=1 (the ceiling trains on mBRSET, which MIX already mixes in)"; CEILING=0
fi

echo "=== pre-flight: cohort under --healthy $HEALTHY   $(date) ==="
# shellcheck disable=SC2086
$PY train_retinal_age.py "${COMMON[@]}" --inspect $EXTRA \
  || { echo "[fatal] cohort pre-flight failed (see above); nothing trained"; exit 1; }
if [ "$CEILING" = 1 ]; then
  echo "=== pre-flight: mBRSET DR-0 ceiling cohort   $(date) ==="
  # shellcheck disable=SC2086
  $PY train_retinal_age.py "${COMMON[@]}" "${CEIL[@]}" --inspect $EXTRA \
    || { echo "[fatal] ceiling cohort pre-flight failed (see above); nothing trained"; exit 1; }
fi

run() {  # run <name> <flags...>
  local name=$1; shift
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (exists)"; return 0; fi
  echo; echo "=== $name   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_retinal_age.py "${COMMON[@]}" --run-name "$name" \
      --results-json "$OUT/$name.json" "$@" $EXTRA
}

for s in "${SEEDS[@]}"; do
  run "student${TAG}_seed$s" --seed "$s" --backbone "$STUDENT"
  if [ -n "$TEACHER" ]; then
    run "teacher${TAG}_seed$s" --seed "$s" --backbone "$TEACHER" --lr "$TEACHER_LR"
    if [ "$KD" = 1 ]; then
      tpt="$CK/teacher${TAG}_seed$s.pt"
      if [ -f "$tpt" ] && [ -f "$CK/teacher${TAG}_seed$s.done" ]; then
        run "kd${TAG}_seed$s" --seed "$s" --backbone "$STUDENT" --teacher "$tpt" \
            --kd-alpha "$KD_ALPHA" --distill-feat-weight "$FEAT_W"
      else
        echo "[warn] teacher${TAG}_seed$s has no finished checkpoint ($tpt); kd${TAG}_seed$s skipped"
      fi
    fi
  fi
  if [ "$CEILING" = 1 ]; then
    run "ceiling${TAG}_seed$s" --seed "$s" --backbone "$STUDENT" "${CEIL[@]}"
  fi
done

echo; echo "=== summary   $(date) ==="
$PY summarize_retinal_age.py --dir "$OUT"
echo; echo "=== age gap vs disease (patient-level, gradable images)   $(date) ==="
BCSV=$($PY -c "from brset_dataset import load_any; print(load_any('$B', 'brset')['csv'])" 2>/dev/null || true)
$PY analyze_age_gap.py --predictions "$OUT/predictions_pooled.csv" ${BCSV:+--brset-csv "$BCSV"} \
    --out "$OUT/associations" > /dev/null \
  && echo "-> $OUT/associations/age_gap_associations.md" \
  || echo "[warn] analyze_age_gap.py failed; run it by hand on $OUT/predictions_pooled.csv"
echo "done=$(date)  ->  $OUT/summary.md  +  $OUT/predictions_pooled.csv  +  $OUT/associations/"

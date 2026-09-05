#!/usr/bin/env bash
# run_systemic.sh — systemic (oculomics) targets on mBRSET: hypertension,
# nephropathy, neuropathy, myocardial infarction, ... from the smartphone fundus
# photograph alone, trained and tested IN-DOMAIN on mBRSET's patient-grouped split
# (these labels do not exist in BRSET, so there is no transfer arm).
#
#   tmux new -s sys
#   cd ~/Oculomics && M=<mBRSET root> DR_CK=ck_kd_v4_384/kd_seed1.pt bash run_systemic.sh 0 1 2 2>&1 | tee -a exp_systemic/sweep.log
#
# Per task and seed it trains up to TWO models on the same patient split:
#   ctrl_seed<s>     the deployable student ($STUDENT), ImageNet init
#   drinit_seed<s>   same architecture, every non-head tensor warm-started from
#                    $DR_CK (the BRSET-trained referable-DR student); only when DR_CK is set
# Both record the age+sex logistic baseline on the same split (--covariate-baseline),
# so summarize_systemic.py reports, per task: image AUROC, covariate AUROC, the
# paired image-minus-covariate delta (is there retinal signal beyond age?), and the
# paired drinit-minus-ctrl delta (does DR pre-training transfer?).
#
# Gate: inspect_mbrset.py --strict runs first and refuses tasks whose column is
# missing or single-class as encoded, so a 1/2-coded release fails here, not after
# an epoch. Idempotent: a run whose JSON exists is skipped.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: M=<mBRSET root> [DR_CK=<DR student .pt>] [KNOB=value ...] bash run_systemic.sh [seed ...]

Trains the image-only student on each systemic target in mBRSET (in-domain,
patient-grouped), records the age+sex covariate baseline per run, optionally a
second arm warm-started from the DR student, then prints paired statistics.
Seeds default to 0 1 2. Every knob is an environment variable:

  required
    M              mBRSET root     (dir holding images/ + labels_mbrset.csv)
  optional
    DR_CK          train_mbrset.py checkpoint to warm-start the 'drinit' arm from
                   (e.g. ck_kd_v4_384/kd_seed1.pt). Unset = ctrl arm only.
    TASKS          space-separated tasks (default: "hypertension nephropathy neuropathy
                   myocardial_infarction"); any key of dataset.SYSTEMIC_TASKS
    COVARIATES     columns for the logistic baseline (default "age sex"; add dm_time
                   for the stricter chart-knowledge baseline)
  outputs
    OUT            results JSONs + summary        (default exp_systemic)  -> $OUT/<task>/
    CK             checkpoints                    (default ck_systemic)   -> $CK/<task>/
  models / training
    STUDENT        backbone (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
    SIZE           image size                     (default 384)
    EPOCHS         epochs per run                 (default 25)
    WORKERS        DataLoader workers             (default 5)
    AMP            "" = trainer default, "--amp" / "--no-amp" to force
    EXTRA          extra flags for both arms, e.g. "--ema-decay 0.999"

example
    M=/data/mBRSET/1.0 DR_CK=ck_kd_v4_384/kd_seed1.pt TASKS="hypertension nephropathy" bash run_systemic.sh 0 1 2 3 4
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

: "${M:?set M=<mBRSET root> (dir holding images/ + labels_mbrset.csv); see --help}"
DR_CK=${DR_CK:-}
TASKS=${TASKS:-hypertension nephropathy neuropathy myocardial_infarction}
COVARIATES=${COVARIATES:-age sex}
OUT=${OUT:-exp_systemic}
CK=${CK:-ck_systemic}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-25}
WORKERS=${WORKERS:-5}
AMP=${AMP:-}
EXTRA=${EXTRA:-}
PY=.venv/bin/python

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
mkdir -p "$OUT" "$CK"
[ -d "$M" ] || { echo "[fatal] mBRSET root not found: $M"; exit 1; }
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
# M may be a parent of the real directory (PhysioNet: mBRSET/1.0/): resolve it to the
# dir holding the label CSV, or fail naming the CSVs that were found.
M=$($PY - "$M" <<'PY'
import sys
from brset_dataset import resolve_root
try:
    print(resolve_root(sys.argv[1], "mbrset"))
except FileNotFoundError as e:
    sys.exit(f"[fatal] {e}")
PY
) || exit 1
CSV=$(ls "$M"/labels_mbrset.csv "$M"/dataframe_brsetmobile.csv 2>/dev/null | head -1)
echo "[info] mBRSET root: $M  ($(basename "$CSV"))"
if [ -n "$DR_CK" ] && [ ! -f "$DR_CK" ]; then echo "[fatal] DR_CK not found: $DR_CK"; exit 1; fi
$PY -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }

# shellcheck disable=SC2086
$PY inspect_mbrset.py --csv "$CSV" --tasks $TASKS --features $COVARIATES --strict \
  || { echo "[fatal] pre-flight failed: fix the encoding (dataset._binary_flag tokens) or drop the task"; exit 1; }

STUDENT_GCG=()
case "$STUDENT" in timm:*) STUDENT_GCG=(--no-gcg);; esac

run() {  # run <task> <name> <flags...>
  local task=$1 name=$2; shift 2
  if [ -f "$OUT/$task/$name.json" ]; then echo "[skip] $task/$name (exists)"; return 0; fi
  mkdir -p "$OUT/$task" "$CK/$task"
  echo; echo "=== $task / $name   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_mbrset.py --dataset mbrset --root "$M" --task "$task" \
      --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" \
      --backbone "$STUDENT" ${STUDENT_GCG[@]+"${STUDENT_GCG[@]}"} \
      --covariate-baseline --covariate-features $COVARIATES \
      --ckpt-dir "$CK/$task" --run-name "$name" --results-json "$OUT/$task/$name.json" \
      $AMP $EXTRA "$@"
}

for t in $TASKS; do
  for s in "${SEEDS[@]}"; do
    run "$t" "ctrl_seed$s" --seed "$s"
    if [ -n "$DR_CK" ]; then
      run "$t" "drinit_seed$s" --seed "$s" --init-from "$DR_CK"
    fi
  done
done

echo; echo "=== paired statistics   $(date) ==="
# shellcheck disable=SC2086
if [ -n "$DR_CK" ]; then
  $PY summarize_systemic.py --dir "$OUT" --tasks $TASKS --treatment drinit --control ctrl
else
  $PY summarize_systemic.py --dir "$OUT" --tasks $TASKS --control ctrl
fi
echo "done=$(date)  ->  $OUT/summary.md"

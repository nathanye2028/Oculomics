#!/usr/bin/env bash
# run_brset_cm.sh — BRSET free-text comorbidity targets (cm_*, brset_comorbidities.py)
# from the tabletop fundus photograph, ONE camera, with the oculomics question
# answered in every run: does the photo add to age+sex?
#
#   tmux new -s cm
#   cd ~/oculomics-systemic && PY=~/Oculomics/.venv/bin/python GPU=0 \
#     B=<BRSET root> M=<mBRSET root> bash run_brset_cm.sh 0 1 2 2>&1 | tee -a exp_brset_cm/sweep_gpu0.log
#
# Per task and seed, one model on the camera-filtered, patient-grouped split
# (ctrl_seed<s>). Each run records (train_mbrset.py --covariate-baseline):
#   * the age+sex logistic baseline on the same split          (image minus covariates)
#   * age+sex vs age+sex+image score, fit on val, scored on test,
#     overall and in non-diabetic DR-grade-0 patients          (does the photo add?)
#   * for cm_hypertension, mBRSET's own hypertension column as an external test.
# summarize_systemic.py then pairs every number by seed.
#
# FOLDS=5 runs the 5-fold design instead (ctrl_fold<k>_seed<s>): every recorded
# patient is tested once, the folds' test rows pool out of fold, and the small
# before-lesions subsets (normal_looking, normal_looking_no_dm, all_flags_clear)
# get every eligible patient rather than a fifth of them.
#
# Why one camera: the comorbidity audit (2026-10-15) found histories for 75% of
# Canon CR patients and 3% of Nikon NF5050 ones. Idempotent: a run whose JSON
# exists is skipped. One job per GPU (the lab box cards are ~11.6 GiB).
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> [M=<mBRSET root>] [KNOB=value ...] bash run_brset_cm.sh [seed ...]

Seeds default to 0 1 2 (with FOLDS: 0). Every knob is an environment variable:

  required
    B              BRSET root (dir holding labels_brset.csv; a parent also works)
  optional
    M              mBRSET root: external test for tasks with an mBRSET equivalent
                   (cm_hypertension). Unset = in-domain only.
    TASKS          default "cm_hypertension"; any cm_* task in dataset.COMORBIDITY_TASKS
    CAMERA         default "Canon CR"
    COVARIATES     default "age sex"; "age sex diabetes" for the intake-form version
    PY             python (default .venv/bin/python; in a worktree pass ~/Oculomics/.venv/bin/python)
    GPU            CUDA_VISIBLE_DEVICES for this invocation (default: unchanged)
  cross-validation
    FOLDS          e.g. 5: patient-grouped K-fold, every recorded patient tested once,
                   pooled out of fold by summarize_systemic.py      (default "": 70/10/20)
    FOLD_LIST      the folds THIS invocation runs, to share them across GPUs, e.g. "0 1 2"
                   on GPU 0 and "3 4" on GPU 1                       (default: all folds)
    FOLD_SEED      seed of the fold partition, fixed across folds and seeds (default 42)
  outputs
    OUT            results + summary  (default exp_brset_cm)  -> $OUT/<task>/ctrl_seed<s>.json
                   (with FOLDS: ctrl_fold<k>_seed<s>.json + ctrl_fold<k>_seed<s>_test.csv)
    CK             checkpoints        (default ck_brset_cm)   -> $CK/<task>/
  model / training
    STUDENT        default timm:mobilenetv4_conv_medium.e500_r256_in1k (the retinal-age
                   clock of record: capacity helped there)
    SIZE           default 512
    EPOCHS         default 30
    BATCH          default 16 (Medium @ 512 on an 11.6 GiB card)
    EMA            default 0.999 (0 = off)
    ABLATION       vessel ablation at the end of each run: fraction of the field of view
                   inpainted as vessel, plus the rotated control (default 0.12; 0 = off)
    WORKERS        default 5
    EXTRA          extra train_mbrset.py flags

example (two GPUs)
    GPU=0 B=... M=... PY=... bash run_brset_cm.sh 0 1 2
    GPU=1 B=... M=... PY=... bash run_brset_cm.sh 3 4
    # 5-fold, one seed, the folds shared across the two cards
    GPU=0 FOLDS=5 FOLD_LIST="0 1 2" B=... M=... PY=... bash run_brset_cm.sh 0
    GPU=1 FOLDS=5 FOLD_LIST="3 4"   B=... M=... PY=... bash run_brset_cm.sh 0
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1

: "${B:?set B=<BRSET root>; see --help}"
M=${M:-}
TASKS=${TASKS:-cm_hypertension}
CAMERA=${CAMERA:-Canon CR}
COVARIATES=${COVARIATES:-age sex}
PY=${PY:-.venv/bin/python}
OUT=${OUT:-exp_brset_cm}
CK=${CK:-ck_brset_cm}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_medium.e500_r256_in1k}
SIZE=${SIZE:-512}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-16}
EMA=${EMA:-0.999}
WORKERS=${WORKERS:-5}
EXTRA=${EXTRA:-}
FOLDS=${FOLDS:-}
FOLD_SEED=${FOLD_SEED:-42}
ABLATION=${ABLATION:-0.12}
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi

SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then
  if [ -n "$FOLDS" ]; then SEEDS=(0); else SEEDS=(0 1 2); fi
fi
if [ -n "$FOLDS" ]; then
  FOLD_LIST=${FOLD_LIST:-$(seq 0 $((FOLDS - 1)) | tr '\n' ' ')}
  for k in $FOLD_LIST; do
    [ "$k" -ge 0 ] && [ "$k" -lt "$FOLDS" ] || { echo "[fatal] fold $k is outside 0..$((FOLDS - 1))"; exit 1; }
  done
  echo "[info] $FOLDS-fold design (fold seed $FOLD_SEED): this invocation runs folds $FOLD_LIST"
fi
[ -x "$PY" ] || { echo "[fatal] no python at $PY (in a worktree: PY=~/Oculomics/.venv/bin/python)"; exit 1; }
mkdir -p "$OUT" "$CK"

# Pre-flight: resolve the roots and count positive / negative PATIENTS per task on
# the camera-filtered table, before any GPU time. Fails on a task with < 50
# positive patients (a 20% test split would hold < 10).
# shellcheck disable=SC2086
EXT=$($PY - "$B" "$M" "$CAMERA" $TASKS <<'PY'
import sys
from brset_dataset import add_brset_extras, filter_camera, load_any, resolve_root
from brset_comorbidities import MBRSET_EQUIVALENTS
from dataset import COMORBIDITY_TASKS
b, m, cam, tasks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
src = load_any(b, "brset")
df = filter_camera(add_brset_extras(src["df"], src["csv"]), cam)
print(f"[info] BRSET {src['csv']}  camera {cam!r}: {len(df)} rows, {df['patient'].nunique()} patients",
      file=sys.stderr)
bad = []
for t in tasks:
    if t not in COMORBIDITY_TASKS:
        sys.exit(f"[fatal] {t} is not a cm_* task; known: {sorted(COMORBIDITY_TASKS)}")
    per = df.groupby("patient")[t].first()
    pos, neg = int((per == 1).sum()), int((per == 0).sum())
    print(f"[info]   {t:<28} positive patients {pos:5d}  negative {neg:5d}  "
          f"(unrecorded {int(per.isna().sum())})", file=sys.stderr)
    if pos < 50:
        bad.append(t)
if bad:
    sys.exit(f"[fatal] fewer than 50 positive patients for {bad}: association analysis only")
if m:
    print(resolve_root(m, "mbrset"))
PY
) || exit 1

one_run() {  # one_run <task> <seed> <run name> [extra train_mbrset flags...]
  local t=$1 s=$2 name=$3; shift 3
  local js="$OUT/$t/$name.json"
  if [ -f "$js" ]; then echo "[skip] $t $name (exists)"; return 0; fi
  mkdir -p "$OUT/$t" "$CK/$t"
  local ext=()
  if [ -n "$EXT" ] && $PY -c "import sys; from brset_comorbidities import MBRSET_EQUIVALENTS as E; sys.exit(0 if sys.argv[1] in E else 1)" "$t"; then
    ext=(--external-test-root "$EXT" --external-test-dataset mbrset)
  fi
  echo; echo "=== $t $name  camera '$CAMERA'  $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_mbrset.py --dataset brset --root "$B" --camera "$CAMERA" --task "$t" --seed "$s" \
      --backbone "$STUDENT" --no-gcg --image-size "$SIZE" --epochs "$EPOCHS" \
      --batch-size "$BATCH" --ema-decay "$EMA" --num-workers "$WORKERS" \
      --covariate-baseline --covariate-features $COVARIATES ${ext[@]+"${ext[@]}"} \
      --vessel-ablation "$ABLATION" \
      --ckpt-dir "$CK/$t" --run-name "$name" --results-json "$js" "$@" $EXTRA
}

for t in $TASKS; do
  for s in "${SEEDS[@]}"; do
    if [ -n "$FOLDS" ]; then
      for k in $FOLD_LIST; do
        one_run "$t" "$s" "ctrl_fold${k}_seed$s" --fold "$k" --n-folds "$FOLDS" --fold-seed "$FOLD_SEED"
      done
    else
      one_run "$t" "$s" "ctrl_seed$s"
    fi
  done
done

echo; echo "=== paired statistics   $(date) ==="
# shellcheck disable=SC2086
$PY summarize_systemic.py --dir "$OUT" --tasks $TASKS --control ctrl
echo "done=$(date)  ->  $OUT/summary.md"

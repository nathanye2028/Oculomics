#!/usr/bin/env bash
# run_probe_systemic.sh — Recommendation 1: a frozen RETFound-DINOv2 encoder plus a
# 33-number classifier (StandardScaler -> PCA(32) -> L2 logistic regression,
# balanced) on the systemic targets, on EXACTLY the CNN runs' 5 folds.
#
#   tmux new -s probe
#   cd ~/oculomics-systemic && PY=~/Oculomics/.venv/bin/python GPU=1 \
#     B=<BRSET root> M=<mBRSET root> bash run_probe_systemic.sh 2>&1 | tee -a exp_probe.log
#
# Steps (each skips what already exists, so a rerun resumes):
#   1. extract_retfound.py: one 1,024-number embedding per image, mBRSET and BRSET
#      (GPU, minutes; the only step that needs the gated weights)
#   2. probe_systemic.py on mBRSET: TASKS_M with COVARIATES_M   -> $OUT/mbrset
#   3. probe_systemic.py on BRSET, one camera: TASKS_B with COVARIATES_B, plus the
#      mBRSET external score for cm_hypertension               -> $OUT/brset_canon
#   4. summarize_systemic.py --control probe on both (pooled out of fold: the
#      incremental delta and the matched-pair concordance)
#   5. compare_probe_cnn.py against the CNN fold runs (it judges a task only when both
#      arms have every fold finished on the same partition and covariates)
#
# The folds equal the CNN's only if the table, the camera and FOLD_SEED match:
# run the CNNs with  run_systemic.sh FOLDS=5  (mBRSET) and  run_brset_cm.sh FOLDS=5
# (BRSET) with the same COVARIATES. CPU is enough after step 1.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> M=<mBRSET root> [KNOB=value ...] bash run_probe_systemic.sh

  required
    B, M           BRSET and mBRSET roots (a parent of the label CSV's dir also works)
  optional
    PY             python (default .venv/bin/python; in a worktree ~/Oculomics/.venv/bin/python)
    GPU            CUDA_VISIBLE_DEVICES for the extraction
    WEIGHTS        RETFound weights: local .pth or HF repo id (default YukunZhou/RETFound_dinov2_meh;
                   gated: request access, then python -c "from huggingface_hub import login; login()" once)
    FEAT           embeddings dir                        (default features)
    OUT            results root                          (default exp_probe)
    TASKS_M        default "hypertension nephropathy myocardial_infarction neuropathy"
    COVARIATES_M   default "age sex"           (= run_systemic.sh's default COVARIATES)
    TASKS_B        default "cm_hypertension"
    COVARIATES_B   default "age sex diabetes"  (= the RUNBOOK cm_folds run)
    CAMERA         default "Canon CR"
    FOLDS          default 5;  FOLD_SEED default 42 (must equal the CNN runs')
    SEED           run label, default 0
    CNN_M, CNN_B   CNN fold-run roots to compare against (default exp_systemic_folds, exp_cm_folds)
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
: "${B:?set B=<BRSET root>; see --help}"
: "${M:?set M=<mBRSET root>; see --help}"
PY=${PY:-.venv/bin/python}
WEIGHTS=${WEIGHTS:-YukunZhou/RETFound_dinov2_meh}
FEAT=${FEAT:-features}
OUT=${OUT:-exp_probe}
TASKS_M=${TASKS_M:-hypertension nephropathy myocardial_infarction neuropathy}
COVARIATES_M=${COVARIATES_M:-age sex}
TASKS_B=${TASKS_B:-cm_hypertension}
COVARIATES_B=${COVARIATES_B:-age sex diabetes}
CAMERA=${CAMERA:-Canon CR}
FOLDS=${FOLDS:-5}
FOLD_SEED=${FOLD_SEED:-42}
SEED=${SEED:-0}
CNN_M=${CNN_M:-exp_systemic_folds}
CNN_B=${CNN_B:-exp_cm_folds}
if [ -n "${GPU:-}" ]; then export CUDA_VISIBLE_DEVICES=$GPU; fi
[ -x "$PY" ] || { echo "[fatal] no python at $PY"; exit 1; }
mkdir -p "$FEAT" "$OUT"

echo "=== 1. RETFound embeddings   $(date) ==="
$PY extract_retfound.py --dataset mbrset --root "$M" --weights "$WEIGHTS" --out "$FEAT/mbrset_retfound.npz"
$PY extract_retfound.py --dataset brset  --root "$B" --weights "$WEIGHTS" --out "$FEAT/brset_retfound.npz"

common=(--n-folds "$FOLDS" --fold-seed "$FOLD_SEED" --seed "$SEED")
echo; echo "=== 2. mBRSET probes   $(date) ==="
for t in $TASKS_M; do
  # shellcheck disable=SC2086
  $PY probe_systemic.py --dataset mbrset --root "$M" --task "$t" --features "$FEAT/mbrset_retfound.npz" \
      --covariate-features $COVARIATES_M --out "$OUT/mbrset" "${common[@]}"
done

echo; echo "=== 3. BRSET ($CAMERA) probes   $(date) ==="
for t in $TASKS_B; do
  ext=()
  if $PY -c "import sys; from brset_comorbidities import MBRSET_EQUIVALENTS as E; sys.exit(0 if sys.argv[1] in E else 1)" "$t"; then
    ext=(--external-dataset mbrset --external-root "$M" --external-features "$FEAT/mbrset_retfound.npz")
  fi
  # shellcheck disable=SC2086
  $PY probe_systemic.py --dataset brset --root "$B" --camera "$CAMERA" --task "$t" \
      --features "$FEAT/brset_retfound.npz" --covariate-features $COVARIATES_B \
      --out "$OUT/brset_canon" "${common[@]}" ${ext[@]+"${ext[@]}"}
done

echo; echo "=== 4. pooled out of fold   $(date) ==="
# shellcheck disable=SC2086
$PY summarize_systemic.py --dir "$OUT/mbrset" --tasks $TASKS_M --control probe
# shellcheck disable=SC2086
$PY summarize_systemic.py --dir "$OUT/brset_canon" --tasks $TASKS_B --control probe

echo; echo "=== 5. probe vs CNN on the same patients   $(date) ==="
if [ -d "$CNN_M" ]; then
  # shellcheck disable=SC2086
  $PY compare_probe_cnn.py --probe-dir "$OUT/mbrset" --cnn-dir "$CNN_M" --tasks $TASKS_M --seed "$SEED"
else
  echo "[info] no $CNN_M yet. The CNN arm on the same folds and covariates:"
  echo "       M=$M FOLDS=$FOLDS FOLD_SEED=$FOLD_SEED COVARIATES=\"$COVARIATES_M\" TASKS=\"$TASKS_M\" OUT=$CNN_M bash run_systemic.sh $SEED"
fi
if [ -d "$CNN_B" ]; then
  # shellcheck disable=SC2086
  $PY compare_probe_cnn.py --probe-dir "$OUT/brset_canon" --cnn-dir "$CNN_B" --tasks $TASKS_B --seed "$SEED"
else
  echo "[info] no $CNN_B yet. The CNN arm on the same folds and covariates:"
  echo "       B=$B M=$M CAMERA=\"$CAMERA\" FOLDS=$FOLDS FOLD_SEED=$FOLD_SEED COVARIATES=\"$COVARIATES_B\" TASKS=\"$TASKS_B\" OUT=$CNN_B bash run_brset_cm.sh $SEED"
fi
echo "done=$(date)  ->  $OUT/mbrset/summary.md, $OUT/brset_canon/summary.md"

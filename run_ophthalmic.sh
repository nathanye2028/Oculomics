#!/usr/bin/env bash
# run_ophthalmic.sh — BRSET's ophthalmic labels beyond DR: AMD, drusen, increased
# cup-to-disc ratio, hypertensive retinopathy, vascular occlusion, hemorrhage,
# myopic fundus, retinal detachment, scar, nevus. IN-DOMAIN on BRSET (tabletop),
# patient-grouped split — mBRSET has none of these columns, so no transfer arm.
#
#   tmux new -s oph
#   cd ~/Oculomics && B=<BRSET root> bash run_ophthalmic.sh 0 1 2 2>&1 | tee -a exp_ophthalmic/sweep.log
#
# Per seed, on the same patient split:
#   multi_seed<s>            ONE model with a multi-label head over every label ($LABELS)
#   single_<label>_seed<s>   one dedicated binary model per label
# summarize_ophthalmic.py pairs multi-minus-single per label by seed: does sharing a
# trunk across labels help the rare ones (retinal detachment, vascular occlusion)?
# Idempotent: a run whose JSON exists is skipped.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> [KNOB=value ...] bash run_ophthalmic.sh [seed ...]

Trains a multi-label model over BRSET's ophthalmic labels and one binary model
per label on the same patient split, then prints paired per-label statistics.
Seeds default to 0 1 2. Every knob is an environment variable:

  required
    B              BRSET root  (dir holding fundus_photos/ + labels_brset.csv)
  optional
    LABELS         labels to train single models for (default: every label of the
                   multi-label head, dataset.OPHTHALMIC_LABELS); "" = multi only
    SINGLE         1 (default) trains the per-label models, 0 trains multi only
  outputs
    OUT            results JSONs + summary        (default exp_ophthalmic)
    CK             checkpoints                    (default ck_ophthalmic)
  models / training
    STUDENT        backbone (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
    SIZE           image size                     (default 384)
    EPOCHS         epochs per run                 (default 25)
    WORKERS        DataLoader workers             (default 5)
    IMBALANCE      sampler | loss | both          (default sampler; 'loss' = BCE pos_weight)
    AMP            "" = trainer default, "--amp" / "--no-amp" to force
    EXTRA          extra flags for every run, e.g. "--ema-decay 0.999"

example
    B=/data/BRSET/1.0.1 LABELS="amd drusens increased_cup_disc" bash run_ophthalmic.sh 0 1 2 3 4
USAGE
}
for a in "$@"; do
  case "$a" in -h|--help) usage; exit 0;; esac
done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
PY=.venv/bin/python

: "${B:?set B=<BRSET root> (dir holding fundus_photos/ + labels_brset.csv); see --help}"
OUT=${OUT:-exp_ophthalmic}
CK=${CK:-ck_ophthalmic}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
SIZE=${SIZE:-384}
EPOCHS=${EPOCHS:-25}
WORKERS=${WORKERS:-5}
IMBALANCE=${IMBALANCE:-sampler}
AMP=${AMP:-}
EXTRA=${EXTRA:-}
SINGLE=${SINGLE:-1}
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
LABELS=${LABELS-$($PY -c "from dataset import OPHTHALMIC_LABELS; print(' '.join(OPHTHALMIC_LABELS))")}

SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 1 2)
mkdir -p "$OUT" "$CK"
[ -d "$B" ] || { echo "[fatal] BRSET root not found: $B"; exit 1; }
$PY -c "import timm" 2>/dev/null || { echo "[fatal] timm missing: .venv/bin/pip install -r requirements.txt"; exit 1; }

# Pre-flight: the ophthalmic columns must survive the BRSET adapter with two classes each.
CSV=$(ls "$B"/labels_brset.csv "$B"/labels.csv "$B"/dataframe_brset.csv 2>/dev/null | head -1 || true)
[ -n "$CSV" ] || { echo "[fatal] no BRSET label CSV under $B"; exit 1; }
$PY brset_dataset.py --csv "$CSV" --inspect | sed -n '/OPHTHALMIC/,/^$/p'

STUDENT_GCG=()
case "$STUDENT" in timm:*) STUDENT_GCG=(--no-gcg);; esac

run() {  # run <name> <task> <flags...>
  local name=$1 task=$2; shift 2
  if [ -f "$OUT/$name.json" ]; then echo "[skip] $name (exists)"; return 0; fi
  echo; echo "=== $name  (task $task)   $(date) ==="
  # shellcheck disable=SC2086
  $PY -u train_mbrset.py --dataset brset --root "$B" --task "$task" \
      --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" --imbalance "$IMBALANCE" \
      --backbone "$STUDENT" ${STUDENT_GCG[@]+"${STUDENT_GCG[@]}"} \
      --ckpt-dir "$CK" --run-name "$name" --results-json "$OUT/$name.json" $AMP $EXTRA "$@"
}

for s in "${SEEDS[@]}"; do
  run "multi_seed$s" ophthalmic --seed "$s"
  if [ "$SINGLE" = "1" ]; then
    for lab in $LABELS; do
      run "single_${lab}_seed$s" "$lab" --seed "$s"
    done
  fi
done

echo; echo "=== paired statistics   $(date) ==="
$PY summarize_ophthalmic.py --dir "$OUT"
echo "done=$(date)  ->  $OUT/summary.md"

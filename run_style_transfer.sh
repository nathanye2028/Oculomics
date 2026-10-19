#!/usr/bin/env bash
# run_style_transfer.sh — tabletop -> handheld image synthesis, three methods, one verdict.
#
#   tmux new -s style
#   cd ~/Oculomics && B=<BRSET root> M=<mBRSET root> bash run_style_transfer.sh 2>&1 | tee -a exp_style/run.log
#
# Stages (each idempotent; STAGES picks a subset):
#   fit     the two CycleGAN translators: matched (same ICDR grade both sides, tabletop
#           diabetics only) and naive (everything vs everything). One per GPU when
#           GPUS lists two.
#   check   realism + label preservation of every method on each BASE checkpoint
#           (existing train_retinareach.py checkpoints trained on the real BRSET).
#   train   per ARM: a translated copy of BRSET, then train_retinareach.py on it, per seed.
#           The identity arm is the control (same files, same JPEG round trip, untranslated).
#   score   every retrained model on the REAL handheld and tabletop test splits.
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: B=<BRSET root> M=<mBRSET root> [KNOB=value ...] bash run_style_transfer.sh [seed ...]

Seeds default to 0 3 (the seeds with finished real-BRSET checkpoints).
  required
    B, M        BRSET root, mBRSET root
  stages
    STAGES      subset of "fit check train score"                 (default: all four)
    GPUS        GPU ids; the two translators train in parallel on two   (default "0")
  check
    BASE        checkpoints trained on the real BRSET   (default ck_retinareach/seed<s>.pt per seed)
  train
    ARMS        (default "identity colour fda_sharp cyc_matched cyc_naive")
    EPOCHS      (default 30)   SIZE (default 384)   WORKERS (default 5)
    STUDENT     (default timm:mobilenetv4_conv_small.e2400_r224_in1k)
    EXTRA       extra trainer flags                        (default "--ema-decay 0.999")
  translators
    ITERS       CycleGAN iterations at batch 2, 256 px crops           (default 20000)
  outputs
    DEVICE      auto | cuda | cpu | mps for the synthesis steps              (default auto)
    OUT (default exp_style)   CK (default ck_style)   DATA (default data_style; ~0.5 GB per arm)
USAGE
}
for a in "$@"; do case "$a" in -h|--help) usage; exit 0;; esac; done

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
: "${B:?set B=<BRSET root>; see --help}"
: "${M:?set M=<mBRSET root>; see --help}"
STAGES=${STAGES:-fit check train score}
GPUS=(${GPUS:-0})
ARMS=${ARMS:-identity colour fda_sharp cyc_matched cyc_naive}
OUT=${OUT:-exp_style}; CK=${CK:-ck_style}; DATA=${DATA:-data_style}
ITERS=${ITERS:-20000}; EPOCHS=${EPOCHS:-30}; SIZE=${SIZE:-384}; WORKERS=${WORKERS:-5}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}
EXTRA=${EXTRA:---ema-decay 0.999}
PY=.venv/bin/python
DEVICE=${DEVICE:-auto}
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0 3)
mkdir -p "$OUT" "$CK" "$DATA"
TR=(cyc_matched="$CK/matched.pt" cyc_naive="$CK/naive.pt")
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if has fit; then
  pids=()
  i=0
  for mode in matched naive; do
    if [ -f "$CK/$mode.pt" ]; then echo "[fit] $CK/$mode.pt exists, skipping"; continue; fi
    gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
    CUDA_VISIBLE_DEVICES=$gpu $PY style_transfer_xfer.py fit --device "$DEVICE" --mode $mode --source-root "$B" \
      --target-root "$M" --image-size "$SIZE" --iters "$ITERS" --num-workers "$WORKERS" \
      --out "$CK/$mode.pt" > "$OUT/fit_$mode.log" 2>&1 &
    pids+=($!)
    echo "[fit] $mode on GPU $gpu -> $OUT/fit_$mode.log"
    [ ${#GPUS[@]} -lt 2 ] && wait "${pids[@]}"
  done
  [ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
  for mode in matched naive; do [ -f "$CK/$mode.pt" ] || { echo "[fit] $mode failed: tail $OUT/fit_$mode.log"; exit 1; }; done
fi
export CUDA_VISIBLE_DEVICES=${GPUS[0]}

if has check; then
  for s in "${SEEDS[@]}"; do
    base=${BASE:-ck_retinareach/seed$s.pt}
    for b in $base; do
      [ -f "$b" ] || { echo "[check] no base checkpoint $b, skipping"; continue; }
      d="$OUT/check_$(basename "$b" .pt)"
      [ -f "$d/check.csv" ] && { echo "[check] $d exists, skipping"; continue; }
      $PY style_transfer_xfer.py check --device "$DEVICE" --ckpt "$b" --source-root "$B" --target-root "$M" \
        --translator "${TR[@]}" --num-workers "$WORKERS" --out "$d"
    done
    [ -n "${BASE:-}" ] && break
  done
fi

if has train; then
  for arm in $ARMS; do
    if [ ! -f "$DATA/$arm/style_transfer.json" ]; then
      $PY style_transfer_xfer.py export --device "$DEVICE" --method "$arm" --source-root "$B" --target-root "$M" \
        --image-size "$SIZE" --translator "${TR[@]}" --num-workers "$WORKERS" --out-root "$DATA/$arm"
    fi
    for s in "${SEEDS[@]}"; do
      [ -f "$OUT/${arm}_seed$s/results.json" ] && { echo "[train] $arm seed $s done, skipping"; continue; }
      $PY train_retinareach.py --source-root "$DATA/$arm" --target-root "$M" --backbone "$STUDENT" \
        --image-size "$SIZE" --epochs "$EPOCHS" --num-workers "$WORKERS" --seed "$s" \
        --probe-targets hypertension --calib-sizes 16 --calib-repeats 1 --envelope-repeats 2 \
        --prior-strengths 0 --n-shuffle 0 --ablation-frac 0 --n-boot 100 --cv-repeats 1 $EXTRA \
        --out "$OUT/${arm}_seed$s" --ckpt "$CK/${arm}_seed$s.pt"
    done
  done
fi

if has score; then
  cks=()
  for arm in $ARMS; do for s in "${SEEDS[@]}"; do
    [ -f "$CK/${arm}_seed$s.pt" ] && cks+=("$arm=$CK/${arm}_seed$s.pt")
  done; done
  [ ${#cks[@]} -gt 0 ] || { echo "[score] no retrained checkpoints in $CK"; exit 1; }
  $PY style_transfer_xfer.py score --device "$DEVICE" --source-root "$B" --target-root "$M" --ckpt "${cks[@]}" \
    --num-workers "$WORKERS" --out "$OUT/score"
fi
echo "[done] $OUT/check_*/check_summary.txt, $OUT/check_*/grid.jpg, $OUT/score/score_summary.txt"

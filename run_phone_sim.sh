#!/usr/bin/env bash
# run_phone_sim.sh — does training on simulated phone photos close the tabletop -> phone gap?
#
#   tmux new -s phonesim
#   cd ~/Oculomics && mkdir -p exp_phone_sim
#   B=<BRSET root> M=<mBRSET root> bash run_phone_sim.sh 0 2>&1 | tee -a exp_phone_sim/run.log
#   B=<BRSET root> M=<mBRSET root> STAGES="train score" bash run_phone_sim.sh 1 2 3 4
#
# Stages (each idempotent; STAGES picks a subset):
#   fit    phone_sim.py fit: the handheld bank from the mBRSET TRAIN split (no label read)
#          and the tabletop statistics from the Canon train split -> $CK/profile.json,
#          fit_report.txt (cross-fitted realism, per statistic) and fit_grid.jpg.
#   check  style_transfer_xfer.py check on the Canon sweep's checkpoint: realism among
#          ICDR-0 eyes (camera distance, camera probe) and label preservation (judge,
#          logit shifts, vessel_r) for phone_sim next to colour / fda_sharp.
#   train  the Canon sweep again, identical except --phone-sim (run_retinareach.sh with
#          SOURCE_CAMERA / HOLDOUT_CAMERA, so the Nikon rung and every output exist too).
#   score  style_transfer_xfer.py score, Canon-only vs Canon + simulated phone photos, on
#          the same real phone and Canon test rows: AUROC, sensitivity / specificity at
#          the shipped threshold, test-oracle specificity at 0.85 sensitivity.
#
# Knobs (environment): B, M (required); STAGES (default "fit check train score");
# CAM="Canon CR", HOLD="NIKON NF5050"; P=0.5 (share of Canon training photos simulated);
# STRENGTH=1.0; CK=ck_phone_sim (profile); OUT=exp_phone_sim; ARM_OUT=exp_rr_phonesim,
# ARM_CK=ck_rr_phonesim (the retrained sweep); BASE_CK=ck_rr_canon (the Canon-only sweep the
# arm is compared with); SIZE=384; DEVICE=auto; WORKERS=5; GPUS (CUDA_VISIBLE_DEVICES);
# EXTRA (more trainer flags, default "--ema-decay 0.999", the Canon sweep's).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
: "${B:?set B=<BRSET root>}"
: "${M:?set M=<mBRSET root>}"
STAGES=${STAGES:-fit check train score}
CAM=${CAM:-Canon CR}
HOLD=${HOLD:-NIKON NF5050}
P=${P:-0.5}
STRENGTH=${STRENGTH:-1.0}
CK=${CK:-ck_phone_sim}
OUT=${OUT:-exp_phone_sim}
ARM_OUT=${ARM_OUT:-exp_rr_phonesim}
ARM_CK=${ARM_CK:-ck_rr_phonesim}
BASE_CK=${BASE_CK:-ck_rr_canon}
SIZE=${SIZE:-384}
DEVICE=${DEVICE:-auto}
WORKERS=${WORKERS:-5}
EXTRA=${EXTRA:---ema-decay 0.999}
PY=.venv/bin/python
SEEDS=("$@"); [ ${#SEEDS[@]} -eq 0 ] && SEEDS=(0)
[ -x "$PY" ] || { echo "[fatal] no .venv — see README Quick start"; exit 1; }
for v in B M; do [ -d "${!v}" ] || { echo "[fatal] $v root not found: ${!v}"; exit 1; }; done
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$GPUS
mkdir -p "$OUT" "$CK"
PROFILE="$CK/profile.json"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

if has fit; then
  if [ -f "$PROFILE" ]; then echo "[fit] $PROFILE exists, skipping"
  else
    $PY phone_sim.py fit --device "$DEVICE" --target-root "$M" --source-root "$B" --source-camera "$CAM" \
      --image-size "$SIZE" --num-workers "$WORKERS" --out "$PROFILE"
  fi
fi
[ -f "$PROFILE" ] || { echo "[fatal] no profile at $PROFILE (run the fit stage)"; exit 1; }

if has check; then
  for s in "${SEEDS[@]}"; do
    base="$BASE_CK/seed$s.pt"
    [ -f "$base" ] || { echo "[check] no $base, skipping seed $s"; continue; }
    d="$OUT/check_seed$s"
    [ -f "$d/check.csv" ] && { echo "[check] $d exists, skipping"; continue; }
    $PY style_transfer_xfer.py check --device "$DEVICE" --ckpt "$base" --source-root "$B" --target-root "$M" \
      --methods colour fda_sharp --phone-profile "$PROFILE" --phone-strength "$STRENGTH" \
      --num-workers "$WORKERS" --out "$d"
  done
fi

if has train; then
  # the Canon sweep's own pipeline, one flag different
  B="$B" M="$M" SOURCE_CAMERA="$CAM" HOLDOUT_CAMERA="$HOLD" OUT="$ARM_OUT" CK="$ARM_CK" SIZE="$SIZE" \
    WORKERS="$WORKERS" EXTRA="$EXTRA --phone-sim $PROFILE --phone-sim-p $P --phone-sim-strength $STRENGTH" \
    bash run_retinareach.sh "${SEEDS[@]}"
fi

if has score; then
  cks=()
  for s in "${SEEDS[@]}"; do
    [ -f "$BASE_CK/seed$s.pt" ] && cks+=("canon_only=$BASE_CK/seed$s.pt")
    [ -f "$ARM_CK/seed$s.pt" ] && cks+=("canon_phonesim=$ARM_CK/seed$s.pt")
  done
  [ ${#cks[@]} -gt 0 ] || { echo "[score] no checkpoints in $BASE_CK or $ARM_CK"; exit 1; }
  $PY style_transfer_xfer.py score --device "$DEVICE" --source-root "$B" --target-root "$M" --ckpt "${cks[@]}" \
    --num-workers "$WORKERS" --out "$OUT/score"
fi
echo "[done] $CK/fit_report.txt, $CK/fit_grid.jpg, $OUT/check_seed*/check_summary.txt, $OUT/score/score_summary.txt"

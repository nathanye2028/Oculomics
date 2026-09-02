#!/usr/bin/env bash
# reproduce.sh — the headline BRSET -> mBRSET result (REPORT.md §5.2, §6) end to end.
#
#   B=<BRSET root> M=<mBRSET root> bash reproduce.sh            # 5 seeds, V4-Small @ 384 px
#   B=... M=... SEEDS="0 1 2" bash reproduce.sh                 # fewer seeds
#   B=... M=... STAGE=stats bash reproduce.sh                   # skip training, just re-summarise
#
# Stages (STAGE=all|train|stats|deploy):
#   train   run_kd_xfer.sh: ctrl / teacher / kd per seed, all with --bn-adapt (GPU box, tmux)
#   stats   summarize_xfer.py: paired kd-minus-ctrl and the AdaBN table
#   deploy  evaluate_deploy.py on DEPLOY_CK (val-calibrated operating point, INT8 cost)
#           + export_coreml.py on the Mac (Core ML, ANE latency, real-image verification)
#
# Data: BRSET and mBRSET are PhysioNet-credentialed and are never downloaded here.
# Run outputs: $OUT (JSON + summary.md) and $CK (checkpoints); both are gitignored.
set -euo pipefail
cd "$(dirname "$0")"

: "${B:?set B=<BRSET root: dir with fundus_photos/ + labels.csv>}"
: "${M:?set M=<mBRSET root: dir with images/ + labels_mbrset.csv>}"
SEEDS=${SEEDS:-0 1 2 3 4}
STAGE=${STAGE:-all}
OUT=${OUT:-exp_kd_v4_384}
CK=${CK:-ck_kd_v4_384}
SIZE=${SIZE:-384}
WORKERS=${WORKERS:-5}
STUDENT=${STUDENT:-timm:mobilenetv4_conv_small.e2400_r224_in1k}   # the deployed model
DEPLOY_CK=${DEPLOY_CK:-$CK/kd_seed1.pt}   # REPORT §6: chosen by in-domain val AUROC, not by mBRSET
VERIFY_IMAGES=${VERIFY_IMAGES:-$M/images}  # real fundus images for the Core ML pass/fail check
PY=.venv/bin/python

[ -x "$PY" ] || { echo "[fatal] no .venv — run: /usr/bin/python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"; exit 1; }
$PY check_env.py >/dev/null || { echo "[fatal] check_env.py failed — fix the environment first"; exit 1; }

if [[ "$STAGE" == all || "$STAGE" == train ]]; then
  echo "=== train: $STUDENT @ ${SIZE}px, seeds [$SEEDS] -> $OUT / $CK   $(date) ==="
  # shellcheck disable=SC2086
  B="$B" M="$M" SIZE="$SIZE" WORKERS="$WORKERS" STUDENT="$STUDENT" OUT="$OUT" CK="$CK" \
    bash run_kd_xfer.sh $SEEDS
fi

if [[ "$STAGE" == all || "$STAGE" == stats ]]; then
  echo "=== paired statistics   $(date) ==="
  $PY summarize_xfer.py --dir "$OUT" --treatment kd --control ctrl
fi

if [[ "$STAGE" == all || "$STAGE" == deploy ]]; then
  [ -f "$DEPLOY_CK" ] || { echo "[fatal] $DEPLOY_CK not found (set DEPLOY_CK=...)"; exit 1; }
  echo "=== deployment evaluation: $DEPLOY_CK   $(date) ==="
  $PY evaluate_deploy.py --root "$B" --ckpt "$DEPLOY_CK" --external-root "$M" \
      --target-sens 0.90 --calib sweep
  if [[ "$(uname)" == Darwin ]]; then
    echo "=== Core ML export + ANE benchmark + real-image verification   $(date) ==="
    $PY export_coreml.py --checkpoint "$DEPLOY_CK" --verify-images "$VERIFY_IMAGES"
    echo "NB: the Mac ANE figure is an OPTIMISTIC proxy (bandwidth-bound model);"
    echo "    the iPhone number comes from an Xcode Core ML performance report."
  else
    echo "[skip] export_coreml.py is macOS-only; copy $DEPLOY_CK to the Mac and run:"
    echo "       .venv/bin/python export_coreml.py --checkpoint $DEPLOY_CK --verify-images <mBRSET>/images"
  fi
fi
echo "done=$(date)"

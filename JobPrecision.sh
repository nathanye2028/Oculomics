#!/bin/bash
# JobPrecision.sh — SLURM batch job: does reduced precision destroy the MA channel?
#   Submit from the repo root:   cd ~/Oculomics && sbatch JobPrecision.sh
#   Watch it:                    squeue -u $USER   ;   tail -f slurm-<jobid>.out
#   Cancel:                      scancel <jobid>
#
# RUN THIS BEFORE run_arch_sweep.py. It is ~15 minutes and it can invalidate a
# 15-run sweep, because every recommended sweep command passes --amp and --amp
# is fp16 — the same 10-bit mantissa that already took mobilenetv4_m's MA Dice
# from 0.974 to 0.000 under TF32. GradScaler defends against gradient underflow,
# which is a different failure from mantissa truncation, so AMP is not covered by
# the TF32 fix. It is simply untested.
#
# Cheap because it never touches a real training set: three fixed IDRiD patches,
# memorised. Dice near 1.0 is the only correct answer; anything low is a defect.
#
# IMPORTANT: run this on the SAME GPU type you will run the sweep on. TF32 exists
# only on Ampere and later (A100 / RTX 30xx / H100). On an older card the tf32
# rows collapse to fp32 by construction, the positive control goes silent, and an
# "ok" on the amp row means much less — the script says so in its output.

#SBATCH --job-name=oculomics-precision
#SBATCH --output=slurm-%j.out          # stdout+stderr -> slurm-<jobid>.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4              # a fixed in-memory batch; no dataloader pressure
#SBATCH --mem=16G
#SBATCH --time=01:00:00                # 8 runs x 400 steps at 256px; generous

# --- CLUSTER-SPECIFIC: uncomment + set if your cluster requires them ---
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}  start=$(date)"
nvidia-smi

.venv/bin/python check_env.py                 # fail fast if the GPU/env isn't right

# Record the card: the whole result is conditional on the GPU generation.
.venv/bin/python -c "import torch; print('[gpu]', torch.cuda.get_device_name(0), \
'| capability', torch.cuda.get_device_capability(0), \
'| TF32-capable:', torch.cuda.get_device_capability(0)[0] >= 8)"

# mobilenetv3 is the reference encoder; mobilenetv4_m is the one TF32 broke.
# Add efficientvit_b1 to cover every encoder run_arch_sweep.py trains (12 runs).
.venv/bin/python precision_check.py \
    --encoders mobilenetv3 mobilenetv4_m \
    --out-dir exp_precision

echo "done=$(date)  ->  exp_precision/summary.md"

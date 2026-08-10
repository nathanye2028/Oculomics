#!/bin/bash
# JobFGADR.sh — SLURM batch job: single GCG-U-Net training run on IDRiD + FGADR.
#   Submit from the repo root:   cd ~/Oculomics && sbatch JobFGADR.sh
#   Watch it:                    squeue -u $USER   ;   tail -f slurm-<jobid>.out
#   Cancel:                      scancel <jobid>
#
# Differs from JobSubmit.sh (5-seed GCG-vs-control ablation) by running ONE
# training run over the combined IDRiD + FGADR data. FGADR contributes 1290
# fully-annotated train images vs IDRiD's 54, so it dominates the mixture and
# model selection uses FGADR's own 185-image val split (--val-source auto).
#
# Resources are larger than JobSubmit.sh's: 512px patches with tiled native-res
# evaluation over 1333 images needs more workers and RAM than the IDRiD-only runs.

#SBATCH --job-name=oculomics-fgadr
#SBATCH --output=slurm-%j.out          # stdout+stderr -> slurm-<jobid>.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8              # DataLoader workers (1280px PNG decode is the bottleneck)
#SBATCH --mem=32G
#SBATCH --time=48:00:00                # 200 epochs over 1333 images; raise if the partition allows

# --- CLUSTER-SPECIFIC: uncomment + set if your cluster requires them ---
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}  start=$(date)"
nvidia-smi
mkdir -p results

.venv/bin/python check_env.py                 # fail fast if the GPU/env isn't right
.venv/bin/python fgadr_dataset.py             # fail fast if FGADR data is missing/incomplete

# Expect in the log:  [val=fgadr; test=IDRiD+FGADR]   train=1333 val=185 test=27
.venv/bin/python train_idrid.py \
    --datasets idrid fgadr \
    --patch-size 512 --eval-tiled --amp \
    --epochs 200 --batch-size 4 --num-workers 8 \
    --run-name fgadr_gcg --results-json results/fgadr_gcg.json

echo "done=$(date)  ->  results/fgadr_gcg.json  checkpoints/fgadr_gcg.pt"

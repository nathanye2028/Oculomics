#!/bin/bash
# JobSubmit.sh — SLURM batch job for the GCG-vs-control benchmark.
#   Submit from the repo root:   cd ~/Oculomics && sbatch JobSubmit.sh
#   Watch it:                    squeue -u $USER   ;   tail -f slurm-<jobid>.out
#   Cancel:                      scancel <jobid>
# SLURM runs this on a compute node under the scheduler, so it is immune to
# VPN drops / closing VS Code — no tmux/nohup needed.

#SBATCH --job-name=oculomics-gcg
#SBATCH --output=slurm-%j.out          # stdout+stderr -> slurm-<jobid>.out
#SBATCH --gres=gpu:1                   # 1 GPU (run_experiment runs its subprocesses serially)
#SBATCH --cpus-per-task=4              # for DataLoader workers
#SBATCH --mem=16G
#SBATCH --time=24:00:00                # wall-clock cap; raise if your partition allows

# --- CLUSTER-SPECIFIC: uncomment + set these if your cluster requires them ---
# Check available partitions with `sinfo`; ask your admin for the account name.
##SBATCH --partition=gpu
##SBATCH --account=your_account
# If GPUs need a module (not needed here — .venv ships CUDA torch):
# module load cuda

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"                  # the dir you ran `sbatch` from (= ~/Oculomics)

echo "host=$(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-?}  start=$(date)"
nvidia-smi
.venv/bin/python check_env.py           # fail fast if the GPU/env isn't right

# 5 seeds + pretrained encoder -> enough power for the paired significance test.
# Separate out-dir so earlier results are never clobbered.
.venv/bin/python run_experiment.py \
    --seeds 0 1 2 3 4 \
    --epochs 200 --patch-size 512 --eval-tiled \
    --out-dir experiments_pretrained5 --ckpt-dir checkpoints_pretrained5

echo "done=$(date)  ->  experiments_pretrained5/summary.md"

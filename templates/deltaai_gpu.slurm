#!/bin/bash
# ---------------------------------------------------------------------------
# DeltaAI GPU job template (NVIDIA GH200 Grace Hopper).
# Submit with:  sbatch deltaai_gpu.slurm
#
# One node = 4 GH200 superchips = 4 H100 GPUs / 288 ARM cores / ~440 GB.
# Request in units of one GH200 (72 cores / 110 GB / 1 GPU).
# ---------------------------------------------------------------------------
#SBATCH --account=CHANGEME              # `accounts` on DeltaAI shows valid codes
#SBATCH --partition=ghx4                # ghx4 | ghx4-interactive
#SBATCH --job-name=gh200-job
#SBATCH --time=04:00:00                 # hh:mm:ss (partition max: 48h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1           # each superchip = one socket
#SBATCH --cpus-per-task=72              # one full GH200's worth of ARM cores
#SBATCH --mem=100g                      # <= 110 GB per superchip
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=verbose,closest
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module reset
module load python/miniforge3_pytorch   # ships PyTorch + NCCL + AWS OFI plugin
module list

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi
srun python3 train.py

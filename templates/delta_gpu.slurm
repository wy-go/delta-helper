#!/bin/bash
# ---------------------------------------------------------------------------
# Delta single-node GPU job template.
# Submit with:  sbatch delta_gpu.slurm
#
# GPU partitions (pick one, match --gpus-per-node accordingly):
#   gpuA40x4     — 4x A40, cheapest        (charge 0.5x)
#   gpuA100x4    — 4x A100 40GB, default   (charge 1.0x)
#   gpuA100x8    — 8x A100 80GB            (charge 1.5x)
#   gpuH200x8    — 8x H200                 (charge 3.0x)
#   gpuMI100x8   — 8x AMD MI100            (charge 0.25x)
# Add `-interactive` (1h max) or `-preempt` (checkpoint required!) suffix.
# ---------------------------------------------------------------------------
#SBATCH --account=CHANGEME-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=gpu-job
#SBATCH --time=04:00:00                 # hh:mm:ss (partition max: 48h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # 1 for single-process PyTorch
#SBATCH --cpus-per-task=16              # ~cores-per-GPU on the node
#SBATCH --mem=64g
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest              # pin to CPU nearest the GPU
#SBATCH --constraint="scratch"
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module reset
module load pytorch-conda               # or your own env
module list

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi
srun python3 train.py

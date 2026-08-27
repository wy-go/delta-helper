#!/bin/bash
# ---------------------------------------------------------------------------
# Delta multi-node PyTorch DDP template (torchrun over Slingshot 11).
# Adapted from the sample in Delta's running_jobs docs.
# Submit with:  sbatch delta_gpu_multinode.slurm
# ---------------------------------------------------------------------------
#SBATCH --account=CHANGEME-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --job-name=ddp-job
#SBATCH --time=02:00:00
#SBATCH --nodes=2                       # scale as needed
#SBATCH --ntasks-per-node=1             # torchrun spawns processes per GPU itself
#SBATCH --gpus-per-node=4               # all GPUs on the node
#SBATCH --cpus-per-task=64              # ~all CPUs; torchrun children share
#SBATCH --mem=0                         # whole-node memory
#SBATCH --constraint="scratch"
#SBATCH --exclusive
#SBATCH --output=ddp-%j.log
#SBATCH --error=ddp-%j.err

set -euo pipefail

module reset
module load pytorch-conda
module load aws-ofi-nccl                # NCCL over Slingshot 11
module list

nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
head_node="${nodes[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node ($head_node_ip)"

export LOGLEVEL=INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn

srun torchrun \
    --nnodes "$SLURM_NNODES" \
    --nproc_per_node "$SLURM_GPUS_PER_NODE" \
    --rdzv_id "$RANDOM" \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    train_ddp.py "$@"

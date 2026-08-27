#!/bin/bash
# ---------------------------------------------------------------------------
# Delta CPU job template.
# Submit with:  sbatch delta_cpu.slurm
# ---------------------------------------------------------------------------
#SBATCH --account=CHANGEME-delta-cpu    # `accounts` on Delta shows valid codes
#SBATCH --partition=cpu                 # cpu | cpu-interactive | cpu-preempt
#SBATCH --job-name=cpu-job
#SBATCH --time=04:00:00                 # hh:mm:ss (partition max: 48h)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16              # OMP_NUM_THREADS defaults to this
#SBATCH --mem=32g
#SBATCH --constraint="scratch"          # add &projects, &work, &taiga, &ime as needed
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

module reset
# module load <your modules here>
module list

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
srun ./your_program

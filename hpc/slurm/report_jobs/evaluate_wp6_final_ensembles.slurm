#!/bin/bash
#SBATCH --job-name=ie643_wp6_final
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${PROJECT_ROOT}/Classification"
"${PROJECT_ROOT}/.venv/bin/python" evaluate_wp6_final_ensembles.py \
  --output-dir "${PROJECT_ROOT}/artifacts/results/wp6_final"

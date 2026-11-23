#!/bin/bash
#SBATCH --job-name=ie643_verify
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0

srun "${PROJECT_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/Classification/verify_split.py" \
  --root "${PROJECT_ROOT}/artifacts/data/official_icbhi" \
  --split-file "${PROJECT_ROOT}/dataset/ICBHI_final_database/ICBHI_challenge_train_test.txt" \
  | tee "${PROJECT_ROOT}/artifacts/data/official_icbhi/splits/verify_split.txt"

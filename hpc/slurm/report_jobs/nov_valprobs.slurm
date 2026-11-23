#!/bin/bash
#SBATCH --job-name=nov_valp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
for d in /tmp /dev/shm /var/tmp; do if t=$(mktemp -d "$d/nv.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi; done
cd "${PROJECT_ROOT}/SpecDiff"
srun "${PROJECT_ROOT}/.venv/bin/python" val_probs.py --arms ${ARMS} --out "${PROJECT_ROOT}/artifacts/specdiff/valprobs.npz"

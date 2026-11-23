#!/bin/bash
# Soft labels for synthetic pools from the real-only AST seed ensemble (teacher_probs.py).
# Submit with e.g.:
#   sbatch -p l40 -q l40 --export=ALL,PHASE=sel,POOLS=sel_s50000_w3.5_empirical+sel_s20000_w3.5_empirical,SEEDS="1 2 3 4 5" hpc/slurm/nov_teacher.slurm
# PHASE=sel uses the nov_sel_M0 models (train-minus-val); PHASE=final uses the nov_final_M0 models (full train).
#SBATCH --job-name=nov_teacher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
: "${PHASE:?}" "${POOLS:?}" "${SEEDS:?}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
for d in /tmp /dev/shm /var/tmp; do
  if t=$(mktemp -d "$d/nv.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PY="${PROJECT_ROOT}/.venv/bin/python"
POOL_ARGS=""
for pool in ${POOLS//+/ }; do POOL_ARGS="${POOL_ARGS} ${PROJECT_ROOT}/artifacts/specdiff/pools/${pool}"; done
CKPT_ARGS=""
for seed in ${SEEDS}; do CKPT_ARGS="${CKPT_ARGS} ${PROJECT_ROOT}/artifacts/runs_nov/icbhi_ast_ce_nov_${PHASE}_M0_seed${seed}/last.pth"; done
cd "${PROJECT_ROOT}/SpecDiff"
srun "${PY}" teacher_probs.py --pool_dirs ${POOL_ARGS} --ckpts ${CKPT_ARGS} --tag "${PHASE}M0"

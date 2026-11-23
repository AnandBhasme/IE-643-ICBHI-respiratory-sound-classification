#!/bin/bash
# Generic job used by scripts/reproduce.py --backend slurm. The command and working directory arrive in the environment:
#   STEP_CMD  shell command to run     STEP_CWD  directory to run it in
# Partition, QOS, GPU, time and memory are given on the sbatch command line by reproduce.py (one GPU per GPU job; no --exclusive
# and no --nodelist). Logs go to hpc/logs/ (create it first: mkdir -p hpc/logs).
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -uo pipefail
: "${STEP_CMD:?}" "${STEP_CWD:?}"
source "${SLURM_SUBMIT_DIR}/hpc/env.sh"
echo "job ${SLURM_JOB_ID} on $(hostname) at $(date '+%F %T %Z')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
cd "${STEP_CWD}"
bash -c "${STEP_CMD}"
rc=$?
echo "exit code ${rc} at $(date '+%F %T %Z')"
exit ${rc}

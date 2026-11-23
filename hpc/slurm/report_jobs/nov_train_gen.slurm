#!/bin/bash
# Train the SpecDiff generator. Resumable: resubmit the same command to continue.
# Submit with:
#   sbatch -p <part> -q <qos> --export=ALL,GEN=sel hpc/slurm/nov_train_gen.slurm
#   GEN=sel  -> official train minus validation patients (selection phase)
#   GEN=full -> all official train cycles (final phase)
#   optional STEPS=<n> (default 60000); GTAG=<suffix> and GEN_ARGS="<extra train.py options>" for generator variants;
#   BUDGET=<minutes> clean-stop budget (default 455, keep it below --time)
#SBATCH --job-name=nov_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
: "${GEN:?GEN=sel or full}"
STEPS="${STEPS:-60000}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
# optional GTAG: generator variant suffix (folder artifacts/specdiff/gen_<GEN><GTAG>)
GTAG="${GTAG:-}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
# Short, writable temp dir (a long or unwritable one breaks DataLoader workers: "AF_UNIX path too long").
for d in /tmp /dev/shm /var/tmp; do
  if t=$(mktemp -d "$d/nv.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi
done
echo "TMPDIR=${TMPDIR:-unset} on $(hostname)"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PY="${PROJECT_ROOT}/.venv/bin/python"
DATA="${PROJECT_ROOT}/artifacts/data/official_icbhi"
CACHE="${PROJECT_ROOT}/artifacts/data/fbank_cache"
cd "${PROJECT_ROOT}/SpecDiff"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [[ ! -f "${CACHE}/test.npz" ]]; then
  srun "${PY}" cache_fbank.py --data_folder "${DATA}" --out_dir "${CACHE}"
fi
EXCLUDE=""
if [[ "${GEN}" == "sel" ]]; then
  EXCLUDE="--exclude_manifest ${DATA}/splits/val_manifest.csv"
fi
srun "${PY}" train.py --cache "${CACHE}/train.npz" \
  --out_dir "${PROJECT_ROOT}/artifacts/specdiff/gen_${GEN}${GTAG}" \
  --steps "${STEPS}" --time_budget_min "${BUDGET:-455}" ${EXCLUDE} ${GEN_ARGS:-}

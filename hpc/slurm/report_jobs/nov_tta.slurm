#!/bin/bash
# Test-time re-generation of real evaluation cycles before classification (tta_translate.py).
#   sbatch -p a40 -q a40 --export=ALL,GEN=sel,STEP=50000,REF=val,SEEDS="1 2 3 4 5 6 7" hpc/slurm/nov_tta.slurm
# GEN=sel with REF=val during selection; GEN=full with REF=test only for the final locked evaluation.
#SBATCH --job-name=nov_tta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
: "${GEN:?}" "${STEP:?}" "${REF:?}" "${SEEDS:?}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
# optional GTAG: generator variant suffix (folder artifacts/specdiff/gen_<GEN><GTAG>)
GTAG="${GTAG:-}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
for d in /tmp /dev/shm /var/tmp; do
  if t=$(mktemp -d "$d/nv.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
PY="${PROJECT_ROOT}/.venv/bin/python"
DATA="${PROJECT_ROOT}/artifacts/data/official_icbhi"
CACHE="${PROJECT_ROOT}/artifacts/data/fbank_cache"
if [[ "${REF}" == "val" ]]; then
  [[ "${GEN}" == "sel" ]] || { echo "REF=val requires GEN=sel" >&2; exit 1; }
  PHASE=sel
  REF_ARGS="--train_cache ${CACHE}/train.npz --val_manifest ${DATA}/splits/val_manifest.csv"
else
  [[ "${GEN}" == "full" ]] || { echo "REF=test requires GEN=full" >&2; exit 1; }
  PHASE=final
  REF_ARGS="--test_cache ${CACHE}/test.npz"
fi
CKPT_ARGS=""
for seed in ${SEEDS}; do CKPT_ARGS="${CKPT_ARGS} ${PROJECT_ROOT}/artifacts/runs_nov/icbhi_ast_ce_nov_${PHASE}_M0_seed${seed}/last.pth"; done
cd "${PROJECT_ROOT}/SpecDiff"
srun "${PY}" tta_translate.py --ckpt "${PROJECT_ROOT}/artifacts/specdiff/gen_${GEN}${GTAG}/ema-${STEP}.pt" \
  --ref "${REF}" ${REF_ARGS} --ast_ckpts ${CKPT_ARGS} \
  --out "${PROJECT_ROOT}/artifacts/specdiff/tta/${GEN}${GTAG}_s${STEP}_${REF}.npz"

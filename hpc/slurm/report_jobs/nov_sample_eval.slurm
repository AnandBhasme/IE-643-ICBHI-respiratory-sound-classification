#!/bin/bash
# Sample a synthetic log-mel pool from a SpecDiff checkpoint and run the quality gate.
# Submit with e.g.:
#   sbatch -p dgx -q dgx --export=ALL,GEN=sel,STEP=60000,W=2.0,POLICY=empirical hpc/slurm/nov_sample_eval.slurm
# GEN=sel  -> gate against the validation subset with the train-minus-val probe
# GEN=full -> gate against the official test set (final locked diagnostic only)
# optional SKIP_EVAL=1 (sample only; use for GEN=full before the selection is locked)
# optional SAMPLE_ARGS="<extra sample.py options>" (e.g. --target_cache for POLICY=target)
# optional PER_CLASS, plus-separated (default 2100+2100+2100+2100); TAGSUF appended to the pool folder name
#SBATCH --job-name=nov_pool
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
: "${GEN:?}" "${STEP:?}" "${W:?}" "${POLICY:?}"
PER_CLASS="${PER_CLASS:-2100+2100+2100+2100}"
PER_CLASS="${PER_CLASS//+/,}"
TAGSUF="${TAGSUF:-}"
# optional TRANSLATE_T=<0..1>: partial re-generation of real cycles (see sample.py); POLICY=other changes the stethoscope
TRANSLATE_T="${TRANSLATE_T:-0}"
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
POOL="${PROJECT_ROOT}/artifacts/specdiff/pools/${GEN}${GTAG}_s${STEP}_w${W}_${POLICY}${TAGSUF}"
cd "${PROJECT_ROOT}/SpecDiff"
EXCLUDE=""
if [[ "${GEN}" == "sel" ]]; then
  EXCLUDE="--exclude_manifest ${DATA}/splits/val_manifest.csv"
fi
if [[ ! -f "${POOL}/pool.npy" ]]; then
  srun "${PY}" sample.py --ckpt "${PROJECT_ROOT}/artifacts/specdiff/gen_${GEN}${GTAG}/ema-${STEP}.pt" \
    --cache "${CACHE}/train.npz" ${EXCLUDE} --out_dir "${POOL}" \
    --per_class "${PER_CLASS}" --guidance "${W}" --device_policy "${POLICY}" --batch_size 100 --translate_t "${TRANSLATE_T}" ${SAMPLE_ARGS:-}
fi
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  echo "SKIP_EVAL=1: pool sampled, gate not run"
elif [[ "${GEN}" == "sel" ]]; then
  srun "${PY}" evaluate_synth.py --pool_dir "${POOL}" \
    --probe "${PROJECT_ROOT}/artifacts/runs_nov/icbhi_ast_ce_nov_probe_trainminusval_seed1_bf16/last.pth" \
    --train_cache "${CACHE}/train.npz" --ref val --val_manifest "${DATA}/splits/val_manifest.csv"
else
  srun "${PY}" evaluate_synth.py --pool_dir "${POOL}" \
    --probe "${PROJECT_ROOT}/artifacts/runs/icbhi_ast_ce_wp3_C0_seed1_bf16/last.pth" \
    --train_cache "${CACHE}/train.npz" --ref test --test_cache "${CACHE}/test.npz"
fi

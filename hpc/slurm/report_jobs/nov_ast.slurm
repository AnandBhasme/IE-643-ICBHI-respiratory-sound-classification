#!/bin/bash
# AST fine-tuning (C0 recipe: CE, 20 epochs, BF16) with optional extra training items.
# Submit with e.g.:
#   sbatch -p l40 -q l40 --export=ALL,PHASE=sel,ARM=M3,SEEDS="1 2 3",MIX_KIND=synth,MIX_MODE=balance,POOL=sel_s60000_w2.0_empirical hpc/slurm/nov_ast.slurm
# PHASE=sel   -> train on train-minus-val, evaluate on the validation subset only
# PHASE=final -> train on the full official train set, report on the official test set
# MIX_KIND=none|synth|real_dup ; MIX_MODE as in build_mix.py ; POOL = pool folder name (synth only)
# optional MINP (probe filter threshold, default 0), AFT=1 (adversarial fine-tuning with real/generated domain head)
#SBATCH --job-name=nov_ast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=02:30:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -uo pipefail
: "${PHASE:?}" "${ARM:?}" "${SEEDS:?}" "${MIX_KIND:?}"
MIX_MODE="${MIX_MODE:-}"
POOL="${POOL:-}"
MINP="${MINP:-0}"
AFT="${AFT:-0}"
# optional two-stage training: INIT_ARM=<arm> starts each seed from that arm's last.pth (same phase, same seed);
# EPOCHS / LR override the stage length and learning rate (defaults 20 / 5e-5)
INIT_ARM="${INIT_ARM:-}"
# optional SOFT=1: synthetic rows are labelled by the real-only seed ensemble of the same phase
# (run nov_teacher.slurm on the pool first); SOFT_BETA mixes teacher label and intended class (default 1 = teacher only)
SOFT="${SOFT:-0}"
SOFT_BETA="${SOFT_BETA:-1.0}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-5e-5}"
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
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

if [[ "${PHASE}" == "sel" ]]; then
  REAL_MANIFEST="${DATA}/splits/train_minus_val_manifest.csv"
  PHASE_ARGS="--val_mode holdout --val_manifest ${DATA}/splits/val_manifest.csv"
else
  REAL_MANIFEST="${DATA}/splits/train_manifest.csv"
  PHASE_ARGS="--save_probs"
fi

MIX_ARGS=""
if [[ "${MIX_KIND}" != "none" ]]; then
  MIX_CSV="${PROJECT_ROOT}/artifacts/specdiff/mixes/${PHASE}_${ARM}.csv"
  POOL_ARG=""
  if [[ "${MIX_KIND}" == "synth" ]]; then
    POOL_ARG="--pool_dir ${PROJECT_ROOT}/artifacts/specdiff/pools/${POOL} --min_p_intended ${MINP}"
    if [[ "${SOFT}" == "1" ]]; then POOL_ARG="${POOL_ARG} --teacher_tag ${PHASE}M0"; fi
  fi
  cd "${PROJECT_ROOT}/SpecDiff"
  "${PY}" build_mix.py --kind "${MIX_KIND}" --mode "${MIX_MODE}" --real_manifest "${REAL_MANIFEST}" \
    ${POOL_ARG} --out "${MIX_CSV}" || exit 1
  MIX_ARGS="--extra_manifest ${MIX_CSV}"
  if [[ "${SOFT}" == "1" ]]; then MIX_ARGS="${MIX_ARGS} --soft_extra --soft_beta ${SOFT_BETA}"; fi
fi
AFT_ARGS=""
if [[ "${AFT}" == "1" ]]; then
  AFT_ARGS="--adversarial_ft --meta_mode mixed"
fi

cd "${PROJECT_ROOT}/Classification"
failures=0
for seed in ${SEEDS}; do
  echo "Starting ${PHASE} ${ARM} seed ${seed} at $(date '+%F %T %Z')"
  INIT_ARGS=""
  if [[ -n "${INIT_ARM}" ]]; then
    INIT_CKPT="${PROJECT_ROOT}/artifacts/runs_nov/icbhi_ast_ce_nov_${PHASE}_${INIT_ARM}_seed${seed}/last.pth"
    [[ -f "${INIT_CKPT}" ]] || { echo "missing ${INIT_CKPT}" >&2; failures=$((failures + 1)); continue; }
    INIT_ARGS="--pretrained --pretrained_ckpt ${INIT_CKPT}"
  fi
  if srun "${PY}" main.py \
    --tag "nov_${PHASE}_${ARM}_seed${seed}" \
    --save_dir "${PROJECT_ROOT}/artifacts/runs_nov" \
    --data_folder "${DATA}" \
    --dataset icbhi --seed "${seed}" --train_real_only --class_split lungsound --n_cls 4 \
    --epochs "${EPOCHS}" --batch_size 32 --num_workers 4 --optimizer adam \
    --learning_rate "${LR}" --weight_decay 1e-6 --desired_length 5 \
    --cosine --warmup_epochs 1 --model ast --test_fold official \
    --pad_types repeat --resz 1 --n_mels 128 --from_sl_official \
    --audioset_pretrained --method ce --amp bf16 \
    --print_freq 100 ${PHASE_ARGS} ${MIX_ARGS} ${AFT_ARGS} ${INIT_ARGS}; then
    echo "Completed ${PHASE} ${ARM} seed ${seed} at $(date '+%F %T %Z')"
  else
    failures=$((failures + 1))
    echo "FAILED ${PHASE} ${ARM} seed ${seed}" >&2
  fi
done
(( failures == 0 ))

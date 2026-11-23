#!/bin/bash
# Real-only AST probe trained on train-minus-val (C0 recipe).
# The official test set is not opened by this job.
#SBATCH --job-name=nov_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
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
cd "${PROJECT_ROOT}/Classification"
srun "${PROJECT_ROOT}/.venv/bin/python" main.py \
  --tag "nov_probe_trainminusval_seed1_bf16" \
  --save_dir "${PROJECT_ROOT}/artifacts/runs_nov" \
  --data_folder "${PROJECT_ROOT}/artifacts/data/official_icbhi" \
  --dataset icbhi --seed 1 --train_real_only --class_split lungsound --n_cls 4 \
  --epochs 20 --batch_size 32 --num_workers 4 --optimizer adam \
  --learning_rate 5e-5 --weight_decay 1e-6 --desired_length 5 \
  --cosine --warmup_epochs 1 --model ast --test_fold official \
  --pad_types repeat --resz 1 --n_mels 128 --from_sl_official \
  --audioset_pretrained --method ce --amp bf16 \
  --val_mode holdout \
  --val_manifest "${PROJECT_ROOT}/artifacts/data/official_icbhi/splits/val_manifest.csv" \
  --print_freq 100

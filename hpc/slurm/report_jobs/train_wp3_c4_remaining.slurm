#!/bin/bash
#SBATCH --job-name=ie643_c4_remaining
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -uo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${PROJECT_ROOT}/Classification"

failures=0
for seed in 1 2 3; do
  completed_dir="${PROJECT_ROOT}/artifacts/runs/icbhi_ast_patchmix_cl_wp3_C4_seed${seed}_bf16_retry1"
  if [[ -s "${completed_dir}/last.pth" ]]; then
    echo "C4 seed ${seed} already has a completed retry1 last.pth; skipping at $(date '+%F %T %Z')"
    continue
  fi
  echo "Starting remaining pre-registered C4 seed ${seed}, BF16, target_type=project_block at $(date '+%F %T %Z')"
  if srun "${PROJECT_ROOT}/.venv/bin/python" main.py \
    --tag "wp3_C4_seed${seed}_bf16_retry2" \
    --save_dir "${PROJECT_ROOT}/artifacts/runs" \
    --data_folder "${PROJECT_ROOT}/artifacts/data/official_icbhi" \
    --dataset icbhi --seed "${seed}" --train_real_only --class_split lungsound --n_cls 4 \
    --epochs 20 --batch_size 32 --num_workers 4 --optimizer adam \
    --learning_rate 5e-5 --weight_decay 1e-6 --desired_length 5 \
    --cosine --warmup_epochs 1 --model ast --test_fold official \
    --pad_types repeat --resz 1 --n_mels 128 --from_sl_official \
    --audioset_pretrained --method patchmix_cl --target_type project_block \
    --ema_decay 0.998 --amp bf16 --save_probs --print_freq 100; then
    echo "Completed remaining C4 seed ${seed} at $(date '+%F %T %Z')"
  else
    failures=$((failures + 1))
    echo "FAILED remaining C4 seed ${seed}; continuing to next seed" >&2
  fi
done

echo "Remaining C4 block finished with ${failures} failed seeds at $(date '+%F %T %Z')"
(( failures == 0 ))

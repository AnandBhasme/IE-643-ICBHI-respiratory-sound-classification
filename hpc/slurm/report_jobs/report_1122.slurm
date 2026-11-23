#!/bin/bash
# 22 Nov report diagnostics on saved outputs (no training). TASK=A|B|C.
#SBATCH --job-name=rep1122
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=hpc/logs/%x.%j.out
#SBATCH --error=hpc/logs/%x.%j.err

set -euo pipefail
R="${SLURM_SUBMIT_DIR}"
O="$R/report_extra_1122"
source /lustre-flash/apps/spack/share/spack/setup-env.sh
spack load anaconda3@2023.09-0
export HF_HUB_OFFLINE=1
for d in /tmp /dev/shm /var/tmp; do
  if t=$(mktemp -d "$d/nv.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "job ${SLURM_JOB_ID} task ${TASK} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || true) start $(date)"
cd "$R/SpecDiff"
PY="$R/.venv/bin/python"
case "$TASK" in
  A)
    $PY bias_sweep.py --out "$O/bias_sweep" 2>&1 | tee "$O/bias_sweep/stdout.txt"
    $PY copy_probs_1122.py "$O/probs" 2>&1 | tee -a "$O/bias_sweep/stdout.txt" ;;
  B)
    P="$R/artifacts/specdiff/pools"
    $PY nn_test_copy.py --pools "$P/full_tt_s20000_w2.0_target" "$P/full_s50000_w3.5_empirical" \
      --teacher "$P/full_tt_s20000_w2.0_target/teacher_finalM0.npy" \
      --probe "$R/artifacts/runs/icbhi_ast_ce_wp3_C0_seed1_bf16/last.pth" \
      --train_cache "$R/artifacts/data/fbank_cache/train.npz" --test_cache "$R/artifacts/data/fbank_cache/test.npz" \
      --out "$O/nn_test" 2>&1 | tee "$O/nn_test/stdout.txt" ;;
  C)
    $PY same_probe_1122.py --root "$R" --out "$O/same_probe" 2>&1 | tee "$O/same_probe/stdout.txt" ;;
esac
echo "end $(date)"

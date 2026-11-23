# Sourced by hpc/slurm/step.slurm before every job. Edit the two lines marked EDIT for your cluster.
# On PRAJNA (IIT Bombay) the values below are the ones used for the report.

# EDIT 1: make a Python 3.11 with PyTorch available. Examples:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate ie643
#   module load python/3.11
if [ -f /lustre-flash/apps/spack/share/spack/setup-env.sh ]; then
  source /lustre-flash/apps/spack/share/spack/setup-env.sh
  spack load anaconda3@2023.09-0
fi
# EDIT 2: the virtual environment created with `python -m venv .venv && pip install -r requirements.txt`
if [ -f "${SLURM_SUBMIT_DIR:-$PWD}/.venv/bin/activate" ]; then source "${SLURM_SUBMIT_DIR:-$PWD}/.venv/bin/activate"; fi

export HF_HUB_OFFLINE=1
# short writable temp dir (a long path breaks DataLoader workers: "AF_UNIX path too long")
for d in /tmp /dev/shm /var/tmp; do
  if t=$(mktemp -d "$d/ie643.XXXXXX" 2>/dev/null); then export TMPDIR="$t"; break; fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

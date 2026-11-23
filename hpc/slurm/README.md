# SLURM usage

`scripts/reproduce.py --backend slurm --partition <p> [--qos <q>]` submits one job per step with `--dependency=afterok`, using
`hpc/slurm/step.slurm`. Use `--dry-run` first to see the job graph. Check partitions with `sinfo`; partition and QOS must match
(on PRAJNA the QOS has the same name as the partition: `a40`, `l40`, `dgx`).

Cluster etiquette (PRAJNA rules, followed by these scripts): one GPU per job, no `--exclusive`, no `--nodelist`, honest `--time`,
nothing heavy on the login node. A full reproduction submits about 50 jobs; per-user running/submitted caps apply
(for example 4 running / 5 submitted on `l40`), so on a capped partition submit stage by stage
(`--stages data`, then `--stages m0`, ...) or split seeds with `--seeds`.

Generator training is resumable: if a job hits its `--time_budget_min` or the limit, run the same command again and it continues
from `resume.pt`.

The older single-purpose scripts in this folder (`train_e*.slurm`, `nov_*.slurm`, ...) are the exact jobs used for the report and
contain PRAJNA-specific paths; `step.slurm` is the portable replacement.

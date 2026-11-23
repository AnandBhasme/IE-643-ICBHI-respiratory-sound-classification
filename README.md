# Synthetic respiratory sounds for ICBHI classification — SpecDiff

IE 643 course project (team Gradient Ascent). Four-class lung-sound classification on the ICBHI 2017 database (normal / crackle /
wheeze / both) with an Audio Spectrogram Transformer (AST), and **SpecDiff**: a conditional diffusion model that generates the
classifier's own log-mel input for any class, stethoscope and duration. This repository reproduces the whole pipeline and every
reported metric, on a workstation (default) or on a SLURM cluster, and ships a demo that classifies recordings and generates
synthetic cycles of any of the four classes.

> **Honest summary.** SpecDiff works as a generator (a real-only classifier recognises 89 % of generated wheeze and 83 % of
> generated "both" cycles as the requested class; the earlier DiffWave pipeline reached 6 % and 2 %). On the official test split the
> final arm (T_K3 + prior correction) scores **59.57 ± 0.64 vs 58.46 ± 0.96** for real-only training (10 seeds, paired +1.11,
> p = 0.022, 8/10 seeds). That gain is **statistically indistinguishable from moving the real-only model to the same
> specificity (−0.10 [−0.65, +0.44])**, disappears against a size-matched real-only ensemble, and the main arm uses unlabelled test
> audio (transductive). See [docs/PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) and [docs/PROTOCOL.md](docs/PROTOCOL.md).

| Official test, last epoch, 10 seeds | Single model | Paired vs real only | 10-model ensemble |
|---|---|---|---|
| M0 real only | 58.46 ± 0.96 | – | 60.79 |
| T_K3 + prior correction (transductive) | **59.57 ± 0.64** | +1.11 [+0.20, +2.02], p = 0.022, 8/10 | 60.98 (+0.20, n.s.) |
| M3 + prior correction (no test audio) | 58.83 ± 0.92 | +0.37, p = 0.29 | 60.91 (+0.12, n.s.) |

## 1. Quick start (no data, no GPU needed)

```bash
git clone https://github.com/AnandBhasme/IE-643-ICBHI-respiratory-sound-classification.git
cd IE-643-ICBHI-respiratory-sound-classification
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu   # or the CUDA build
pip install -r requirements.txt                            # torchaudio and torchvision come from the torch line above
python -m pytest tests -q                                  # unit tests, ~1 min on CPU; 2 tests skip without the test audio
python scripts/reproduce.py --stages check evaluate        # every table and figure from the shipped test probabilities, ~1 min
```

`results/saved/probs/` holds the raw last-epoch test probabilities of all 40 final runs (a few MB), so the headline numbers, the
paired statistics, the ensembles and the bias-sweep control are reproduced **without retraining**. Output goes to
`results/reproduced/` (`RESULTS.md`, CSVs, figures, `look9.txt`, `bias_sweep/`).

## 2. Demo: classify, compare and generate

```bash
pip install -r requirements.txt -r requirements-demo.txt
python scripts/download_weights.py          # weights from the GitHub releases weights-v1 and weights-intensive-v1 (≈ 6 GB), SHA-256 checked
python demo/app.py                          # open http://127.0.0.1:7860
```

Options: `python demo/app.py --port 8000`, `--share` for a public link, `--host 0.0.0.0` to reach it from another machine.
Set `IE643_INTENSIVE_DIR` to use intensive-phase checkpoints from another folder instead of `weights/intensive/`.

| Tab | What it does |
|---|---|
| Classify a recording | Upload a WAV cycle or click one of eight preselected official test cycles. Choose a model (final models, intensive candidates, ensembles), a device, and whether to apply the prior correction. The tab shows the probabilities, the input log-mel and the time and memory used. Each parameter name explains itself on hover. |
| Generate and compare | Choose a class, stethoscope, duration, guidance, steps and seed. The tab shows the synthetic log-mel next to the nearest real cycle, the spectral profile, approximate audio, and what the chosen classifier says about both. |
| Generator check | Generates cycles of every class and reports how often the chosen probe recognises the requested class. |
| About | What each model is, its official score, and whether it is installed here. |

**Models** (catalogue in `demo/catalog.py`, scores are the official Score on the official test split, last epoch):

| Model | Official Score | Notes |
|---|---|---|
| T_K3, final model | 59.57 ± 0.64 (10 seeds) | SpecDiff + soft labels, with prior correction. Transductive. |
| M0, real data only | 58.46 ± 0.96 (10 seeds) | Baseline. |
| C4, C2, C1, C0 (one seed each) | 58.74, 58.60, 58.53, 58.14 (3 seeds) | Intensive-phase candidates. |
| Ens-AST, 15 models | 60.39 | Intensive headline ensemble (C0 to C4, seeds 1 to 3). |
| Ens-C1, Ens-C4, Ens-C0 | 60.18, 59.93, 59.57 (3 models each) | Intensive ensembles. |

Ensembles load their members one at a time, so they need little more memory than a single model. Models whose files are missing
are listed as "not installed", with the command that installs them.

**Preselected cycles.** Eight cycles, two per class, are listed in `demo/samples.json` (written by `scripts/make_demo_samples.py`).
Four come from the official test list and four from the training recordings; the demo shows which. They are read from
`artifacts/data/official_icbhi/{test,training}/real/`, which the data stage creates, so the ICBHI recordings are not copied into
the repository.

Command line without the web UI: `python demo/cli.py classify rec.wav [--model T_K3|M0|C4|ENS-AST] [--no-correction]`,
`python demo/cli.py generate --cls wheeze --out out/`, `python demo/cli.py check --n 8 --model M0`.

*Weights.* The files exceed GitHub's 100 MB limit, so they are not in git. They are attached to two releases:
[`weights-v1`](https://github.com/AnandBhasme/IE-643-ICBHI-respiratory-sound-classification/releases/tag/weights-v1) (final models
and generator) and [`weights-intensive-v1`](https://github.com/AnandBhasme/IE-643-ICBHI-respiratory-sound-classification/releases/tag/weights-intensive-v1)
(the 15 intensive-phase checkpoints). `scripts/download_weights.py` reads `weights/weights_manifest.json`, retries failed transfers,
and checks SHA-256 (`--verify-only` checks what is in place, `--release` fetches one release).
*Audio.* The generator outputs spectrograms. The player uses an approximate Griffin-Lim inversion of the log-mel; it is for listening,
not a clinical-quality recording. *Not a medical device.*

## 3. Full reproduction

Data: ICBHI 2017 (about 2 GB zip, free for research, <https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge>) and the
AudioSet-pretrained AST checkpoint (about 345 MB).
```bash
python scripts/download_data.py --icbhi --ast                         # official host; on failure it says what to do
python scripts/download_data.py --zip /path/ICBHI_final_database.zip --ast   # a copy you downloaded yourself
python scripts/download_data.py --verify                              # check everything; non-zero exit on any failure
python scripts/reproduce.py --list                   # all steps and commands
python scripts/reproduce.py --stages all --quick     # tiny end-to-end smoke run (1 seed, 1 epoch, 600 generator steps), ~30 min on a GPU
python scripts/reproduce.py --stages all             # the real thing
```
The split file `ICBHI_challenge_train_test.txt` comes from the official host. If that host cannot be reached, `download_data.py`
rebuilds it from `results/saved/test_order.csv` (the official test cycles behind every reported number). The rebuild is checked: it
must give 539 train and 381 test recordings, and the test recordings must equal those in `test_order.csv`. The script uses the same
recording-name alias as `Classification/prepare_official_split.py`.

**Normal machine (default, `--backend local`)**: steps run one after another, skipping what already exists, so an interrupted run
continues where it stopped. You need one CUDA GPU with ≥ 24 GB (AST training, BF16; use `--amp fp16` on older cards via
`scripts/train_arm.py`).

**HPC (`--backend slurm`)**: one job per step with `--dependency=afterok`, one GPU per GPU job, no `--exclusive`/`--nodelist`:
```bash
mkdir -p hpc/logs && $EDITOR hpc/env.sh                                       # python environment of your cluster
python scripts/reproduce.py --stages all --backend slurm --partition a40 --dry-run   # inspect the job graph
python scripts/reproduce.py --stages all --backend slurm --partition a40             # submit (partition and QOS must match)
```
Cluster caps on submitted jobs apply (see `hpc/slurm/README.md`); submit stage by stage if needed. The exact job files used for
the report are in `hpc/slurm/report_jobs/`.

| Stage | Steps | GPU time (one L40S/A40-class GPU) |
|---|---|---|
| data | split, validation split, log-mel cache | CPU, ~1 h |
| m0 | 10 real-only classifiers | 10 × ~25 min |
| generators | `gen_full` 50k steps, `gen_full_tt` 20k steps (init from the 30k checkpoint) | ~4.5 h + ~1.5 h |
| pools | 12,500 + 8,400 synthetic cycles, quality gate | ~1–2 h |
| teacher | soft labels from M0 seeds 1–5 | ~10 min |
| arms | M1b ×5, M3 ×10, K3 ×5, T_K3 ×10 | 30 × ~25 min |
| evaluate / nn_test | tables, figures, paired statistics, bias sweep, near-copy check | CPU 2 min / GPU 5 min |

Reproduction is deterministic in everything except GPU non-determinism of training, so retrained numbers differ from the
shipped ones by seed noise (the report's seed SD is 0.6–1.0 Score points). `results/saved/` keeps the original probabilities;
a fresh run never mixes with them (`SpecDiff/ensemble_nov.py:load_probs`).

### System requirements

| Use | What you need |
|---|---|
| Shipped tables and statistics (`reproduce.py --stages evaluate`) | Linux, Python 3.11, any CPU, no GPU |
| Unit tests, environment check, data preparation (`--stages check data`) | the same, plus about 5 GB of free disk for the dataset, feature cache and checkpoint |
| Demo: classify a recording | a CPU is enough; a CUDA GPU makes it faster |
| Demo: generate a cycle | a CPU takes minutes per cycle with 50 steps; a GPU takes seconds |
| Training and generation (`m0`, `generators`, `pools`, `teacher`, `arms`, `nn_test`) | one CUDA GPU with **at least 24 GB** of memory. The scripts check this before they start and stop with a message on smaller GPUs. A 4 GB GPU runs out of memory at the recipes' batch size of 32. |
| Full reproduction on a cluster | `--backend slurm` with a partition whose GPUs have at least 24 GB |

Software: Python 3.11, PyTorch 2.0.1 (the CUDA 11.8 build for GPU use), and the packages in `requirements.txt` and
`requirements-demo.txt`. Disk: about 5 GB for data and artifacts after the data stage, and about 0.8 GB for the release weights.

## 4. Repository layout

```
SpecDiff/          generator (model, diffusion, train, sample), quality gate, mixes, soft labels, prior correction, statistics
Classification/    AST trainer (main.py), data preparation, legacy AFT / Patch-Mix code, inference helpers
Diffwave/          the first (failed) pipeline: DiffWave vocoder resynthesis
scripts/           reproduce.py (orchestrator), train_arm.py, download_data.py, download_weights.py, package_weights.py,
                   make_demo_samples.py, make_demo_assets.py, gpu_check.py (refuses GPU steps on hardware that cannot run them)
configs/           arms.json, generators.json: the exact final-phase settings
demo/              app.py (Gradio web UI), engine.py (models, generation, plots), catalog.py (the model list and its scores),
                   samples.json (preselected cycles), cli.py, ABOUT.md, assets/ (real reference log-mels)
hpc/               env.sh, slurm/step.slurm (generic job), slurm/report_jobs/ (jobs used for the report)
weights/           weights_manifest.json and SHA256SUMS are tracked; the checkpoints are downloaded here (not in git)
results/saved/     raw test probabilities of the 40 final runs, test order, training-mix class counts
results/final/     tables, figures and diagnostics of the report
docs/              PROJECT_HISTORY.md (every experiment), PROTOCOL.md, VERIFICATION.md (checks performed), EXPERIMENT_LOG.md,
                   TEST_LOOK_REGISTER.md, NOVELTY_PLAN.md, POSTHOC_DIAGNOSTICS_LOOK10.md, demo_examples/
tests/             CPU unit tests (the demo tests run when the weights and the prepared test cycles are present)
```

## 5. Method in five lines

1. **SpecDiff**: U-Net on 128 × 512 log-mels (AST units), conditioned on class (+ null for guidance), stethoscope and duration;
   v-prediction, cosine schedule, min-SNR weighting, EMA; DDIM 50 steps, classifier-free guidance 2.0–3.5.
2. **Mix**: synthetic cycles top every class up to the normal-class count; synthetic labels are the intended class (M3) or the soft
   output of a real-only teacher ensemble (K3, T_K3).
3. **T_K3** additionally fine-tunes the generator with the unlabelled test cycles (class = null) and draws synthetic stethoscopes from
   the test device histogram (transductive; no test labels).
4. **Prior correction**: `p_c ← p_c · π_real,c / π_mix,c`, no tuned parameter.
5. **Evaluation**: official split, last epoch, Score = (Se + Sp)/2; paired same-seed comparison, bootstrap (2,000 resamples) for ensembles.

## 6. Limitations (please keep them with the numbers)

* Ten looks at the test set are registered in `docs/TEST_LOOK_REGISTER.md`; the final method was chosen after earlier looks failed.
* T_K3 is transductive. M3, which is not, gains +0.37 (n.s.).
* The gain equals an operating-point shift (bias sweep) and vanishes against a size-matched real-only ensemble.
* Wheeze and "both" recall fall; the gain is a specificity gain.
* ICBHI has few patients; the split is recording-level, not patient-disjoint. Last-epoch numbers are not comparable with the
  best-epoch numbers of most papers.

## 7. Credits

Base paper: AFT, arXiv 2311.06480 (check the author list in the PDF). AST: Gong et al. DiffWave: Kong et al. Patch-Mix CL: Bae et al.
ICBHI 2017: Rocha et al. Course: IE 643, IIT Bombay. No licence is set yet; add one before reuse by others. The ICBHI data are not
redistributed here (`demo/assets` holds a few derived log-mel spectrograms for the comparison view).

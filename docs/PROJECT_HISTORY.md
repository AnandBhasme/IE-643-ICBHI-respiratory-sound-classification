# Project history and every experiment (IE 643, team Gradient Ascent)

This file is the short version of `EXPERIMENT_LOG.md` (full, dated lab log), `NOVELTY_PLAN.md` (plan and status) and
`TEST_LOOK_REGISTER.md` (every look at the test set). Numbers are percentages on the **official ICBHI test split** (2,756 cycles)
unless marked *validation*; Score = (Se + Sp) / 2; "last" = last training epoch, "best" = epoch with the highest *test* Score
(optimistic; the base-paper code selects this way).

## 1. Task and constraints

* **Task.** 4-class respiratory-cycle classification on ICBHI 2017 (normal / crackle / wheeze / both), official recording-level split:
  4,142 train / 2,756 test cycles (patients 156 and 218 appear on both sides). Backbone: Audio Spectrogram Transformer (AST,
  AudioSet-pretrained), 5 s repeat-padded cycles, 128-bin Kaldi fbank.
* **Base paper.** AFT (arXiv 2311.06480): DiffWave-resynthesised audio plus adversarial fine-tuning with a real-vs-generated domain head.
* **Feedback from the course.** "Get the synthetic generation working and show a substantial improvement in the final objective."
* **Deliverables.** Poster and ≤ 2-page report, due 22 Nov 2026. Report prose must be human-written.

## 2. Phase 1 — reproduce the base paper (October 2026)

| Run | What | Best / last Score |
|---|---|---|
| E1 | AST + CE, real only (3 seeds: best 60.53 ± 0.62, last 57.85 ± 0.75) | 61.25 / 58.71 (seed 1) |
| E2 | AST + Patch-Mix CL | 59.97 / 57.24 |
| E3 | real + DiffWave v1 synthetic (wheeze, both; N = 1000) + CE | 59.84 / 57.45 |
| E4 | E3 data + AFT | 61.02 / 58.78 |
| E5 | E3 data + Patch-Mix CL + AFT | 60.41 / 55.52 |
| E7 | weighted cross-entropy (class-imbalance control) | 59.89 / 57.81 (3 seeds) |

* **Quality gate of the old generator** (real-only AST probe, % of generated cycles recognised as the intended class):
  DiffWave v1: wheeze 6.0, both 2.4; DiffWave v2 (33.9k steps): 21.7 and 2.5. Cause: a mel vocoder trained for 34k steps (paper: ~1M)
  with 80-bin conditioning, which loses crackle transients and wheeze lines.
* **Plan v3.2 candidates** (C0–C4 recipes, SC1/SC2 scalogram views, 15-checkpoint ensemble "Ens-AST" 60.39 last-epoch Score) and a
  Petmezas CNN-LSTM reproduction. See `NOVELTY_PLAN.md` §6 and `EXPERIMENT_LOG.md` (24–25 Oct).
* Everything here is in `Diffwave/` and `Classification/` (legacy code paths, unchanged).

## 3. Phase 2 — SpecDiff, a generator that works (November 2026)

**Idea.** Skip the vocoder: a class-, stethoscope- and duration-conditional diffusion U-Net (v-prediction, cosine schedule, DDIM
50 steps, classifier-free guidance) generates the classifier's own 128 × 498 log-mel input directly (`SpecDiff/`).

**Quality gate (test reference, G_full 50k steps, guidance 3.5):** intended-class accuracy normal 96.9, crackle 81.4, wheeze 89.1,
both 82.7 (old DiffWave: 6.0 / 2.4 for wheeze / both). Scored with the *same* probe, DiffWave v1 reaches 10.0 / 4.6 and v2 22.5 / 7.5
(`results/final/diagnostics/gate_same_probe.csv`).

**Arms** (all: AST + CE, full official train + added synthetic rows, 20 epochs, BF16):

| Arm | Training mix |
|---|---|
| M0 | real only |
| M1b | real cycles duplicated up to the normal-class count (control without synthetic data) |
| M3 | SpecDiff cycles fill every class up to the normal count (balanced prior) |
| M3f | M3 with a probe filter on synthetic rows |
| M6 | twice the normal count per class |
| K3 | M3 with soft labels from the real-only seed ensemble ("teacher") |
| T_K3 | transductive: generator also trained on the *unlabelled* official-test audio (class = null), synthetic stethoscope mix = test device histogram, soft labels |
| X3_tt | SDEdit translation of training cycles to the test stethoscope mix |
| A3 / A3t | base-paper AFT on the M3 pool / the transductive pool |

**Prior correction (look 8).** Class-balanced training shifts the prior; multiply probabilities by `π_real / π_mix`
(`π_real` = official-train counts 2063 / 1215 / 501 / 363). No tuned parameter. For ensembles the raw probabilities are averaged
first, then weighted (renormalising per model first gives 60.83 instead of 61.22 for T_K3×5; the look-8 definition is kept).

### The ten looks at the test set (full register: `TEST_LOOK_REGISTER.md`)

| # | What | Result |
|---|---|---|
| ctrl | M0, M1b | M0 58.33 ± 1.42 (5 seeds); M1b 57.45 ± 0.83 |
| 1 | M3f (selected on validation), M3, M6 | negative: 56.97, 57.67, 55.12 |
| 2 | fusion of AST with the generator used as a classifier | negative (57.63 vs 58.33); validation 61.4 → test 45.2 |
| 3 | K3, T_K3 | neutral: 57.80, 58.55 |
| 4 | X3_tt SDEdit translation | negative: 56.59 |
| 5 | test-time stethoscope normalisation | neutral: 58.19 |
| 6–7 | A3, A3t (AFT) | negative: 56.42, 57.48 |
| ctrl | M0 seeds 6–10 (size-matched ensemble) | M0×10 ensemble 60.79: earlier ensemble gains were ensemble size |
| 8 | prior correction on saved probabilities of every arm | first positive: T_K3 60.03 ± 0.48 (5 seeds) |
| 9 | replication, seeds 6–10 for T_K3 and M3 | **T_K3 59.57 ± 0.64 vs M0 58.46 ± 0.96, paired +1.11 [+0.20, +2.02], p = 0.022, 8/10 seeds**; ensemble +0.20 n.s.; M3 +0.37 n.s. |
| 10 | post-hoc diagnostics (22 Nov): bias sweep, near-copy distances, same-probe gate | see below |

Validation (12 held-out training patients) did not predict test (`results/final/figures/F8_validation_vs_test.png`).

### Look 10 — what the diagnostics showed

* **Bias sweep** (`diagnostics/matched_summary.csv`): moving a single bias on M0's normal-class score to the same specificity as
  T_K3-corrected gives a Score difference of **−0.10 [−0.65, +0.44], p = 0.67, T_K3 ahead on 4/10 seeds**. M3-corrected: −0.56
  [−1.15, +0.03]. The prior correction therefore acts as an operating-point shift; the +1.11 over *uncorrected* M0 is not evidence of
  better discrimination. At the ensemble level every synthetic arm is below the M0 ensemble at matched specificity.
* **Near-copy check** (`nn_test_summary.csv`): T_K3's synthetic cycles are not near-copies of test cycles (probe-space ratio 1.20,
  1.9% below the 5th percentile of train-to-test distances vs 2.8% for the non-transductive pool). Among the 162 closest samples
  the teacher's soft label agrees with the nearest test label 94% of the time (possible indirect leakage channel; see
  `pseudo_label_check.json`).
* **Same-probe gate**: SpecDiff still beats DiffWave by a wide margin with the same probe.

## 4. What to claim (and not)

* Synthetic generation now **works** (89% / 83% recognised vs 6% / 2%).
* The official Score gain is **small and explained by an operating-point shift**; it does not survive a size-matched real-only
  ensemble or a matched-specificity comparison. It comes from specificity; wheeze and "both" recall drop.
* The main arm is transductive (unlabelled test audio, test device histogram). Nine test looks preceded the final claim.
* Numbers are last-epoch; literature numbers (AST 59.55, AFT 61.79, BTS 63.54, ...) select the best epoch on the test set and are not
  directly comparable.

## 5. Where things are in this repository

| Topic | Code | Outputs |
|---|---|---|
| generator | `SpecDiff/{model,diffusion,train,sample}.py` | `results/final/figures/F1`, F2 |
| quality gate | `SpecDiff/evaluate_synth.py`, `same_probe_1122.py` | `results/final/tables/generator_quality_gate.csv` |
| training mixes, soft labels | `SpecDiff/{build_mix,teacher_probs}.py`, `scripts/train_arm.py` | `results/saved/mix_class_counts.json` |
| prior correction, stats | `SpecDiff/{prior_correct,look9,make_results}.py` | `results/final/tables/`, `results/saved/probs` |
| diagnostics | `SpecDiff/{bias_sweep,nn_test_copy}.py` | `results/final/diagnostics/` |
| failed ideas | `SpecDiff/{diffusion_classify,fuse_dclf,tta_translate}.py`, `Classification/main.py --adversarial_ft` | `results/final/tables/diffusion_classifier_by_step.csv` |

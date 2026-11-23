# Results tables (IE 643, Gradient Ascent, SpecDiff novelty)

Every number here is regenerated from saved outputs by `code/SpecDiff/make_deliverables.py`. The CSVs in this folder are the source of truth. Official ICBHI split, last epoch, Score = (Se + Sp) / 2 in %.

## T1. Headline (look 9): 10 seeds each, official test

| Arm | Single model mean ± SD | Se / Sp | Paired vs M0 (same seed) | 10-model ensemble | Ensemble vs M0×10 (bootstrap 95% CI) |
|---|---|---|---|---|---|
| M0 real only | 58.46 ± 0.96 | 46.9 / 70.0 | – | 60.79 | – |
| **T_K3 + prior correction** (transductive) | **59.57 ± 0.64** | 42.3 / 76.9 | **+1.11 [+0.20, +2.02], t-test p = 0.022, Wilcoxon p = 0.027, 8/10 seeds** | 60.98 | +0.20 [−1.05, +1.41] |
| M3 + prior correction (non-transductive) | 58.83 ± 0.92 | 42.5 / 75.1 | +0.37 [−0.38, +1.12], p = 0.29, 6/10 | 60.91 | +0.12 [−1.29, +1.47] |
| T_K3 raw (no correction) | 58.25 ± 0.59 | – | – | 59.09 | – |
| M3 raw | 57.24 ± 1.15 | – | – | 59.03 | – |

Source: `look9_10seed_summary.csv`, `script_outputs/look9.txt`, `script_outputs/ensembles_raw.txt`.

## T2. All arms, seeds 1-5, raw and prior-corrected (looks 1-8)

| Arm | What it is | Raw single | Corrected single | Raw ens5 | Corrected ens5 | Corrected ens5 vs M0×5 |
|---|---|---|---|---|---|---|
| M0 | real only | 58.33 ± 1.42 | – | 59.78 | – | – |
| M1b | real duplicates balanced (control) | 57.45 ± 0.83 | 58.30 ± 0.67 | 58.48 | 58.75 | −1.03 [−2.42, +0.35] |
| M3 | SpecDiff balance | 57.67 ± 1.25 | 59.17 ± 0.82 | 58.74 | 60.90 | +1.12 [−0.33, +2.57] |
| M3f | + probe filter (look-1 primary) | 56.97 ± 0.63 | 58.54 ± 0.89 | 58.44 | 60.09 | +0.31 [−1.00, +1.63] |
| M6 | 2× normal count per class | 55.12 ± 0.63 | 56.73 ± 0.90 | 55.96 | 58.87 | −0.91 [−2.48, +0.62] |
| K3 | balance + soft labels | 57.80 ± 1.14 | 59.44 ± 1.14 | 58.63 | 60.12 | +0.34 [−0.98, +1.61] |
| T_K3 | transductive + soft labels | 58.55 ± 0.52 | 60.03 ± 0.48 | 58.85 | 61.22 | +1.43 [+0.10, +2.67] |
| X3_tt | SDEdit translation to test devices | 56.59 ± 1.25 | 57.50 ± 1.06 | 57.56 | 59.59 | −0.19 [−1.66, +1.18] |
| A3 | AFT (base paper) | 56.42 ± 0.80 | 57.94 ± 0.67 | 57.42 | 60.18 | +0.40 [−0.97, +1.74] |
| A3t | AFT + transductive pool | 57.48 ± 1.10 | 58.83 ± 1.08 | 58.59 | 61.12 | +1.33 [+0.03, +2.71] |

Source: `test_arm_summary_seeds1-5.csv`, per run in `test_per_run.csv`. A3t was not part of the look-9 replication.

## T3. Per stethoscope and per class, 10-model ensembles (official test)

| | AKG C417L (n=1836) | Meditron (459) | Litt3200 (461) | Recall N / C / W / B |
|---|---|---|---|---|
| M0×10 | 61.8 | 65.4 | 49.7 | 73.2 / 58.1 / 39.7 / 28.0 |
| T_K3×10 corrected | 62.5 | 60.4 | 50.6 | 81.7 / 53.5 / 27.0 / 16.1 |
| M3×10 corrected | 61.4 | 62.8 | 50.9 | 82.9 / 53.5 / 21.0 / 21.0 |

Single-model means over 10 seeds: `figures/F7`. Confusion matrices: `confusion_*.csv`.

## T4. Generator quality gate (real-only AST probe; % of generated samples recognised as the intended class)

| Generator | normal | crackle | wheeze | both | KID N/C/W/B | nn-ratio N/C/W/B |
|---|---|---|---|---|---|---|
| DiffWave v1 (old) | – | – | 6.0 | 2.4 | – | – |
| DiffWave v2 (old) | – | – | 21.7 | 2.5 | – | – |
| SpecDiff G_full 50k, w3.5 (test reference) | 96.9 | 81.4 | 89.1 | 82.7 | 1.64 / 0.83 / 3.74 / 3.25 | 0.56 / 0.74 / 0.48 / 0.64 |
| probe recall on real test cycles | 72.6 | 56.6 | 36.4 | 23.1 | real-train vs test: 1.52 / 2.15 / 4.03 / 5.64 | – |
| SpecDiff G_sel 20k, w2.0 (validation) | 89.9 | 69.9 | 70.8 | 54.0 | 0.94 / 0.87 / 1.46 / 1.47 | 0.87 / 0.96 / 0.78 / 0.87 |
| SpecDiff r3 25k, w2.0 (class-margin, validation) | 97.6 | 67.9 | 97.4 | 46.7 | 4.68 / 3.88 / 7.59 / 4.12 | 1.29 / 1.49 / 0.95 / 1.27 |

All checkpoints and guidance values: `generator_quality_gate.csv`. Old-generator probe = E1 seed 1 (different probe from SpecDiff rows). nn-ratio ≈ 1 means as novel as unseen real data; near 0 means copies.

## T5. Validation screening (12 held-out training patients, 504 cycles)

Selected rows (all 30 arms in `validation_screening_all_arms.txt`):

| Arm | Seeds | Score | Test counterpart (raw single) |
|---|---|---|---|
| M0 | 7 | 70.00 ± 1.42 | 58.33 |
| M1b | 3 | 69.85 ± 1.12 | 57.45 |
| M3_s50k_w3.5 | 3 | 70.83 ± 1.07 | 57.67 |
| M3f_s50k_w3.5 (selected) | 6 | 71.62 ± 1.13 | 56.97 |
| M6_s50k_w3.5 | 7 | 70.96 ± 1.43 | 55.12 |
| K3 | 3 | 71.08 ± 2.75 | 57.80 |
| K3_tv (mirror of T_K3) | 3 | 70.97 ± 1.53 | 58.55 |
| X3_tv (mirror of X3_tt) | 3 | 71.75 ± 1.52 | 56.59 |

Validation prior-correction check: `validation_prior_correction_check.csv`.

## T6. Generator used directly as a classifier (lowest denoising error)

| Generator | 10k | 20k | 25k | 30k | 50k | 90k |
|---|---|---|---|---|---|---|
| G_sel (plain), validation | 43.0 | 44.0 | – | 46.5 | 40.4 | 35.5 |
| r3 (class-margin hinge), validation | 61.2 | 60.8 | 61.4 | 60.7 | – | – |
| r3, official test | – | – | 45.2 | – | – | – |

Source: `diffusion_classifier_by_step.csv`, `figures/F3`.

## T7. Literature anchors (official split, 4-class Score)

AST 59.55; AFT (base paper) 61.79; SG-SCL 61.71; Patch-Mix CL 62.37; MVST 62.50; CycleGuardian 63.26; M2D 63.29; BTS 63.54; distillation 64.39; ADD-AST 65.53. The base-paper code selects the best epoch on the test set (see `logs/earlier_phase/summary_before_novelty.md`: our E1 best 61.25 vs last 58.71). All numbers in this folder are last-epoch, so they are not directly comparable with best-epoch figures.

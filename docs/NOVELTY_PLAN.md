# NOVELTY PLAN: making synthetic generation work (IE 643, team Gradient Ascent)

Written 14 Nov 2026 (IST) for any model or person continuing this work.
Read `handoff.md` first for project history, then `PRAJNA_HPC_AGENT_RULES.md` (binding), then this file.

## 1. Goal and deadline

- Feedback from the intensive-assessment presentation: **get the synthetic generation model working and show a substantial improvement in the final objective.** This is the only focus of the novelty phase.
- Final objective: official ICBHI Score = (Se + Sp)/2 on the official test split (2,756 cycles), ideally with **both** Se and Sp above the real-only control.
- Deliverables (see `IE643 2026 Course Project Novelty Assessment Instructions.pdf`): A1 poster PDF (`GradientAscent_IE643_CourseProject_Poster.pdf`) and a report of at most 2 pages (`GradientAscent_IE643_NoveltyAssessment_report.pdf`), uploaded to the existing Drive folder by **22 Nov 2026, 23:59 IST**. In-person poster session on 23 or 25 Nov.
- Report prose: the course gives zero for AI-generated report text (see `handoff.md` section 14). Supply results, tables, figures and outlines; the user writes the prose.

## 2. Why the old generator failed (from `artifacts/diag/diag_report.md`)

DiffWave was used as a mel-conditioned **vocoder** that re-synthesises real training cycles. It was under-trained (34k steps vs about 1M in the AFT paper) and its 80-bin mel conditioning loses crackle transients and wheeze lines. A real-only AST probe recognised only 6% of generated wheeze and 2% of generated "both" clips as their intended class. Downstream (E3 to E5), Se rose and Sp fell, with no net gain. Only wheeze and "both" were augmented, which also shifts the class prior toward abnormal.

## 3. The new idea ("SpecDiff")

Generate **directly in the classifier's input space**: a conditional diffusion model over the exact 128-bin Kaldi log-mel that AST consumes (498 x 128). No vocoder, so no waveform reconstruction loss, and compute is far smaller.

Design choices and the reason for each:

| Choice | Reason |
|---|---|
| Diffusion on AST log-mel (padded to 128 x 512) | Removes the vocoder, the main detail loss; matches row 2 of the user's own novelty plan (slide 18) |
| Conditioning on **class** (4), **stethoscope** (4) and **cycle duration** | Class gives true class-conditional generation (new samples, not re-syntheses). Device is strongly confounded with class (AKG C417L has 82% of crackle; Meditron is mostly normal). Duration tells the model the repeat-padding period |
| Classifier-free guidance on the **class only**, at fixed device and duration | Amplifies class evidence rather than device evidence |
| "Balanced" device sampling option | Counterfactual class/device combinations, to weaken the stethoscope shortcut (row 5 of slide 18) |
| Generate for **all four classes**, not only minorities | Avoids the prior shift that lowered Sp in E3 to E5 |
| v-prediction, cosine schedule, min-SNR-5 weighting, DDIM 50 steps, EMA 0.9999, BF16 | Standard, stable settings; nothing exotic |
| U-Net base 96, multipliers (1,2,3,4), 2 res blocks per level, attention at 16 x 64, dropout 0.1, gain augmentation | Moderate size for only about 4k training images |

Known risk: with 4,142 training cycles the model may memorise. That is measured (section 5), not assumed away.

## 4. Protocol (keeps the test set clean)

The existing patient-grouped validation split is used for every choice: `artifacts/data/official_icbhi/splits/val_manifest.csv` (504 cycles, 12 patients) and `train_minus_val_manifest.csv` (3,638 cycles).

**Phase A, selection (official test never touched):**
1. `G_sel`: generator trained on train-minus-val.
2. Probe: AST with the C0 recipe trained on train-minus-val (`--val_mode holdout`).
3. Sample pools at a few guidance scales and both device policies; run the quality gate against validation.
4. Downstream screening on validation with the C0 recipe (20 epochs, BF16, about 15 min per run on an L40S), 3 seeds per arm. Planned arms:
   - M0 real only
   - M1 real duplicates, wheeze and both filled to 1000 (non-generative control)
   - M1b real duplicates, all classes balanced to 2063 (step-matched control for M3)
   - M2 synthetic, wheeze and both filled to 1000
   - M3 synthetic, all classes balanced to 2063, empirical devices
   - M4 as M3 with device-balanced pool
   - M5 synthetic, +1x the real count per class (prior preserved)
   - optional: probe-filtered variants; AFT (`--adversarial_ft --meta_mode mixed`) on the best mix
   Validation is small, so differences under about 1.5 points are noise. Pick conservatively and fix the choice in writing before Phase B.

**Phase B, final (locked settings):**
1. `G_full`: same generator settings on all 4,142 training cycles (can train in parallel with `G_sel` because generator hyper-parameters are fixed in advance).
2. Sample the final pool with the selected guidance, device policy and mix.
3. Train **5 seeds each** of: real-only control, matched real-duplicate control, selected synthetic arm(s). Full official train, C0 recipe.
4. Report **last-epoch** official-test Se / Sp / Score, mean ± sample SD, per-class recall, size-matched 5-model ensembles and a paired bootstrap of ensemble differences. Same headline rule as the pre-registered protocol. Report every arm that was run.

Honest framing: an improvement is the hypothesis, not a promise. If the gain is within seed noise, say so, and present the quality-gate numbers as the evidence that the generator itself now works.

## 5. Generator quality gate (`evaluate_synth.py`)

Per class, with a real-only AST probe:
- intended-class accuracy on synthetic samples, next to the probe's recall on real held-out cycles (the realistic ceiling; the old generator scored 6% / 2%);
- KID in probe feature space, synthetic vs held-out real, next to real-train vs held-out real;
- memorisation ratio: nearest-training-neighbour distance of synthetic samples divided by that of real held-out samples (about 1 is as novel as unseen real data, near 0 means copies);
- per-sample probe outputs in `probe.csv` for optional filtering.

Use `--ref val` during selection. `--ref test` only for the final locked diagnostic.

## 6. Code and setup status

Code in `Code/Code/AFT/SpecDiff/` (all executed end to end by the smoke job 385019):

| File | Purpose |
|---|---|
| `model.py` | conditional U-Net, 52.0M parameters. A 2x2 pixel-unshuffle stem was added so it works at 64 x 256 (the first full-resolution version ran out of memory at batch 32 on an L40S). Attention at the two lowest levels |
| `diffusion.py` | schedule, loss, DDIM sampler with class guidance |
| `cache_fbank.py` | AST log-mel cache; done, in `artifacts/data/fbank_cache/{train,test}.npz` |
| `train.py` | generator training; resumable (`resume.pt`), `--time_budget_min`, `--exclude_manifest`; writes `ema-<step>.pt` every 10k steps |
| `sample.py` | writes `pool.npy` (float16, N x 498 x 128, AST units) and `pool.csv`; `--device_policy balanced` is uniform over all four stethoscopes |
| `evaluate_synth.py` | quality gate; writes `quality_<ref>.json` and `probe.csv` |
| `build_mix.py` | builds the `--extra_manifest` CSV. Class lists are plus-separated (`fill:1000:2+3`) because `sbatch --export` splits on commas |
| `collect_nov.py` | summarises runs in `artifacts/runs_nov/` per arm (stdlib only) |

Classifier changes (`Code/Code/AFT/Classification/main.py`, `util/icbhi_diffusion_dataset.py`): `--extra_manifest`, used with `--train_real_only`. Verified working for `real_dup` (M1) and for runs without extras (M0). The `synth` path is verified too. `IE643_repo/Classification/` is the older copy and has **not** been updated.

Job scripts in `hpc/slurm/`: `nov_smoke_specdiff`, `nov_train_gen` (GEN=sel|full, STEPS), `nov_probe`, `nov_sample_eval` (GEN, STEP, W, POLICY), `nov_ast` (PHASE=sel|final, ARM, SEEDS, MIX_KIND, MIX_MODE, POOL, MINP, AFT). New runs go to `artifacts/runs_nov/`, generators to `artifacts/specdiff/gen_{sel,full}/`, pools to `artifacts/specdiff/pools/`, mixes to `artifacts/specdiff/mixes/`.

Measured so far:
- Speed on a shared L40S: training about 2.5 it/s at batch 32 (60k steps is about 7 h); sampling 0.55 s per sample at 50 steps with guidance.
- Smoke test only (600 training steps, 50 samples per class, probe = C0 seed 1 trained on full train): intended-class accuracy 60 / 24 / 56 / 22% (normal / crackle / wheeze / both). Not a result, but already far above the old generator's 6% / 2%.
- Device shift found in the cache: train devices AKG 2510, LittC2SE 594, Meditron 997, Litt3200 41; **test** devices AKG 1836, LittC2SE 0, Meditron 459, Litt3200 461. Litt3200 is rare in training and common in test. This is filename metadata, not labels, but it was seen, so disclose it if the device-balanced arm is reported.

Jobs: see section 6a for the live state.

Bugs already fixed (do not reintroduce): nearest upsampling via `F.interpolate` is not implemented for BF16 in PyTorch 2.0.1 (use `repeat_interleave`); job scripts create a short temp dir with `mktemp -d` under /tmp, /dev/shm or /var/tmp and export it as TMPDIR, or DataLoader workers can fail with `AF_UNIX path too long`; commas inside `--export` values are split by sbatch. Node `cn45-l40` once failed with `No CUDA GPUs are available` (node problem, just resubmit).

## 6a. Results so far (updated 15 Nov 2026, 04:20 IST)

Generators: G_sel (job 385029) and G_full (385030) on `dgx`, about 80k and 77k of 100k steps, loss about 0.018. Both stop at the time budget around 01:05 to 01:20 on 15 Nov, short of 100k; resume with `sbatch -p dgx -q dgx --export=ALL,GEN=sel,STEPS=100000 hpc/slurm/nov_train_gen.slurm` (same for `GEN=full`). Checkpoints `ema-<step>.pt` every 10k.

Quality gate on validation (probe recall on real held-out cycles N/C/W/B = 74.5 / 77.9 / 66.1 / 12.5%; KID real-train vs held-out 1.36 / 1.41 / 2.41 / 5.27):

| Checkpoint, guidance | Intended-class acc N/C/W/B | KID N/C/W/B | nn ratio N/C/W/B |
|---|---|---|---|
| 10k, w2.0 | 83 / 61 / 61 / 49 | 0.86 / 0.98 / 1.61 / 1.37 | 1.09 / 1.14 / 0.93 / 0.98 |
| 20k, w1.3 | 83 / 61 / 48 / 39 | mean 1.10 | 0.92 to 0.96 |
| 20k, w2.0 | 90 / 70 / 71 / 54 | 0.94 / 0.87 / 1.46 / 1.47 | 0.87 / 0.96 / 0.78 / 0.87 |
| 20k, w3.5 | 95 / 83 / 88 / 73 | 1.39 / 0.88 / 2.52 / 2.74 | 0.79 / 0.86 / 0.63 / 0.76 |
| 50k, w3.5 | 97 / 81 / 88 / 76 | 1.95 / 0.81 / 2.32 / 2.65 | 0.68 / 0.82 / 0.58 / 0.70 |

The old generator scored 6% (wheeze) and 2% (both). Stronger guidance gives clearer class identity and less diversity. Generated crackle on non-AKG devices is weak (Meditron 45% at 50k).

Validation screening (504 cycles, last epoch, mean over seeds, sample std):

| Arm | Seeds | Se | Sp | Score | Recall N/C/W/B |
|---|---|---|---|---|---|
| M0 real only | 3 | 64.76 | 74.65 | 69.70 +- 1.57 | 74.6 / 78.1 / 57.7 / 25.0 |
| M1 real_dup fill:1000:2+3 | 3 | 61.77 | 75.80 | 68.79 +- 1.61 | 75.8 / 76.5 / 53.0 / 19.2 |
| M1b real_dup balance | 3 | 60.68 | 79.02 | 69.85 +- 1.12 | 79.0 / 72.7 / 54.8 / 24.2 |
| M2_w2.0 synth fill:1000:2+3 (20k pool) | 3 | 64.22 | 75.93 | 70.08 +- 1.38 | 75.9 / 74.0 / 66.7 / 24.2 |
| M3_w2.0 synth balance (20k pool) | 3 | 65.71 | 74.90 | 70.31 +- 1.22 | 74.9 / 75.4 / 62.5 / 34.2 |
| M3_w3.5 synth balance (20k pool) | 3 | 65.99 | 75.16 | 70.57 +- 1.31 | 75.2 / 75.8 / 66.1 / 29.2 |
| M5_w2.0 synth mult:1 | 3 | 62.86 | 76.58 | 69.72 +- 0.80 | 76.6 / 72.9 / 57.7 / 32.5 |
| M5_w3.5 synth mult:1 | 3 | 62.59 | 76.45 | 69.52 +- 0.82 | 76.4 / 73.6 / 60.7 / 24.2 |

Reading: synthetic balancing is the best arm, +0.4 to +0.9 Score over real-only, with wheeze and "both" recall clearly up. With a seed std near 1.5 this is **not yet a demonstrated gain**. Doubling every class (M5) does not help. Duplicating real cycles does not raise minority recall.

Second screening round (finished 15 Nov, about 04:00; full table: `collect_nov.py --runs_dir artifacts/runs_nov --phase sel`):

| Arm | Seeds | Se | Sp | Score |
|---|---|---|---|---|
| M0 real only | 7 | 64.55 | 75.46 | 70.00 +- 1.42 |
| M3 balance, 50k pool, w3.5 | 3 | 65.99 | 75.68 | 70.83 +- 1.07 |
| M3 balance, 80k pool, w2.0 | 3 | 67.48 | 74.52 | 71.00 +- 2.36 |
| **M3f balance, 50k pool, w3.5, probe filter MINP=0.6** | 6 | 66.46 | 76.77 | **71.62 +- 1.13** |
| M3f8, MINP=0.8 | 2 | 65.71 | 75.68 | 70.69 |
| M6 `fill:3608:0+1+2+3`, 50k pool | 7 | 65.19 | 76.72 | 70.96 +- 1.43 |
| two-stage M3_w3.5 then 5 epochs real (lr 1e-5) | 3 | 65.58 | 74.13 | 69.85 +- 1.45 |
| two-stage control M0 then 5 epochs real | 3 | 64.08 | 75.42 | 69.75 +- 1.44 |
| X3_other: stethoscope translation t=0.5, balance | 3 | 65.03 | 74.52 | 69.78 +- 1.26 |
| X5_other: translation, mult:1 | 3 | 63.27 | 72.97 | 68.12 +- 0.62 |
| X3_same: same-device re-generation, balance | 2 | 61.22 | 78.76 | 69.99 |

Stethoscope translation (`sample.py --translate_t`, `POLICY=other`, job variable `TRANSLATE_T`) noises a real cycle to diffusion time t and denoises it under another device. Its KID is lower than pure generation (0.74 / 0.62 / 1.56 / 2.50) but it does not help the classifier. The 80k checkpoint is not better than 50k on the gate (w3.5: acc 98 / 78 / 85 / 77), so the generators were left at about 90k steps.

**Selection locked 15 Nov 04:10 (see `Code/Code/AFT/EXPERIMENT_LOG.md`):** primary arm M3f; pre-declared ablations M3 (no filter) and M6. Phase B jobs (official test, seeds 1 to 5, `PHASE=final`): controls M0 (386078, 386079) and M1b (386080, 386081); pool gate/probe for `full_s50000_w3.5_empirical` (386247); M3f (386248, 386249); M3 (386250, 386251); M6 (386252, 386253, `fill:4126:0+1+2+3`). Summarise with `collect_nov.py --runs_dir artifacts/runs_nov --phase final`. Probabilities are saved per run (`--save_probs`) for ensembles and the paired bootstrap.

Literature anchors for the poster (official split, 4-class Score): AST 59.55, AFT 61.79, SG-SCL 61.71, Patch-Mix 62.37, MVST 62.50, M2D 63.29, CycleGuardian 63.26, BTS 63.54, distillation 64.39, ADD-AST 65.53. The SAM-AST preprint's 68.10 uses binary sensitivity; its confusion matrix gives official Se 49.53, Sp 67.89, Score about 58.7. C2GA (VQ-VAE + autoregressive Transformer, class-only conditioning, curated 3-class subset, CNN14/ResNet18) reports F1 gains of +1.35 / +2.20. A realistic "substantial" gain here is about +1.5 to +2.5 Score.

## 6b. Phase B result and what followed (updated 15 Nov 2026, 23:00 IST)

**First locked attempt is negative.** Official test, 5 seeds, Se / Sp / Score: M0 real only 47.27 / 69.39 / 58.33 +- 1.42; M1b 57.45; M3 57.67; M3f (selected) 45.39 / 68.56 / 56.97 +- 0.63; M6 55.12. Seed ensembles: M0 59.78, M3 58.74, M3f 58.44. The +1.6 validation gain did not transfer. This must be reported.

Second attempt (post hoc, validation only so far; details in `EXPERIMENT_LOG.md`):
- Soft labels from the real-only seed ensemble (`teacher_probs.py`, `nov_teacher.slurm`, `SOFT=1` / `SOFT_BETA` in `nov_ast.slurm`, `--soft_extra` in `main.py`): K arms reach 70.9 to 71.8 on validation, the same size of gain as before. Not run on test.
- Generator as classifier (`diffusion_classify.py`, `nov_dclf.slurm`): validation Score only 35 to 46.5 (best at 30k steps, worse with longer training); adds nothing to AST. This is the clearest explanation of the negative result and a cheap generalisation measure for the generator.
- Test-time re-generation / stethoscope normalisation (`tta_translate.py`, `nov_tta.slurm`): no benefit.
- Regularised generators `gen_sel_r1/_r2/_r3` (job variable `GTAG`, extra `train.py` options via `GEN_ARGS`: `--dropout`, `--shift_aug`, `--base`, `--cls_margin_w`) are training; compare them with `nov_dclf.slurm` (`GTAG=_r1`, `STEP=...`) before sampling pools.

## 6c. Status 18 Nov 2026

- r1 stopped at 19.8k steps (time budget), r2 finished 30k, r3 stopped at 28.6k. K6_s50k_w1.5 on validation: 70.26 +- 1.82 (no better).
- Generator-as-classifier jobs submitted 18 Nov for r1 10k/15k, r2 10k/20k/30k, r3 10k/20k/25k (jobs 402449 to 402456); outputs `artifacts/specdiff/dclf/sel_r*_s*_val.npz`, score in the last line of `hpc/logs/nov_dclf.<job>.out`. Baseline to beat: 46.53 (G_sel 30k).

- 18 Nov results (details in `EXPERIMENT_LOG.md`): class-margin generator r3 is a good classifier on validation (61.4 at 25k) but a poor sampler (KID 4 to 8; augmentation 68.8 to 69.0 < M0 70.0). Locked fusion AST + diffusion classifier: +1.8 on validation single models, but **negative on test** (ensemble 59.78 -> 57.9; diffusion classifier alone 45 on test). Transductive generator (test audio unlabelled) did not change that. r3 training jobs cancelled.
- Running (third test look, pre-declared): `T_K3` (transductive pool `full_tt_s20000_w2.0_target`, soft labels) and ablation `K3` (non-transductive pool, soft labels), 5 seeds each on test; validation mirror `K3_tv`.

## 6d. Test looks 3 to 9 (updated 18 Nov 2026, evening; full record in `EXPERIMENT_LOG.md`)

All numbers are official test, last epoch, Score = (Se+Sp)/2. Every look is listed; none is dropped.

| Look | Arm | Single model (mean ± SD) | 5-model ensemble |
|---|---|---|---|
| 1-2 | M0 real only (seeds 1-5) | 58.33 ± 1.42 | 59.78 |
| 1-2 | M3f / M3 / M6 / M1b | 56.97 / 57.67 / 55.12 / 57.45 | 58.44 / 58.74 / 55.96 / 58.48 |
| 3 | K3 soft labels | 57.80 ± 1.14 | 58.63 |
| 3 | T_K3 transductive pool + soft labels | 58.55 ± 0.52 | 58.85 |
| 4 | test-time normalisation | 58.19 | - |
| 5 | X3_tt SDEdit translation to test devices | 56.59 ± 1.25 | 57.56 |
| 6-7 | A3 / A3t AFT (base-paper adversarial FT) | 56.42 / 57.48 | 57.42 / 58.59 |
| control | M0 seeds 1-10 | 58.46 ± 0.96 | 60.79 (10 models) |
| **8** | **prior correction p·π_real/π_mix**: M3 / T_K3 / K3 / M1b control | 59.17 / **60.03 ± 0.48** / 59.44 / 58.30 | 60.90 / **61.22** / 60.12 / 58.75 |

- Mechanism: class-balanced synthetic data raises Se, lowers Sp; 57% of test cycles are normal. Logit adjustment removes the training-prior shift with no tunable parameter. The real-duplicate control gains nothing from it, so the gain needs the generated samples.
- T_K3 corrected vs M0: paired per seed +1.50 / +2.24 / +2.35 / +3.29 / −0.89; ensemble bootstrap vs M0×5 +1.42 [+0.10, +2.67]; vs M0×10 only +0.43 [−0.89, +1.71].
- Validation check (no test data): synthetic arms beat M0 per model already without correction (70.6-71.9 vs 69.94); the correction helps the M3 family (+0.2 to +1.2) and is neutral/slightly negative for the soft-label K family.
- **Look 9 (declared, 10 seeds each, corrected):** T_K3 59.57 ± 0.64 vs M0 58.46 ± 0.96, paired +1.11 [+0.20, +2.02], p = 0.022, 8/10 seeds; T_K3×10 ensemble 60.98 vs M0×10 60.79 (+0.20, n.s.). M3 58.83, paired +0.37 (p = 0.29), ensemble 60.91. Report the 10-seed numbers; the 5-seed +1.70 was optimistic.
- Caveats to state: 9 looks at test; T_K3 is transductive; gain comes from Sp, wheeze and both recall drop.

## 7. Next steps, in order

1. Run `nov_dclf.slurm` on the checkpoints of `gen_sel_r1/_r2/_r3`. Only if one is clearly better than 46.5 on validation: sample a pool from it (`nov_sample_eval.slurm` with `GTAG`), label it (`nov_teacher.slurm`), screen balance and large-volume arms with at least 5 seeds.
2. A second test run is justified only for a validation gain clearly larger than the +1.6 that already failed (aim for +2.5 or more over 5+ seeds). It must be reported as a second look next to the first attempt.
3. Deliverables regardless of outcome: gate table (old generator 6% / 2% vs new), validation and test tables, per-class recall, generator-as-classifier curve over training steps, real vs synthetic log-mel grid, paired bootstrap of M3f vs M0 on test.
4. Sync code to `IE643_repo/` and GitHub; poster content and report outline for the user (results, tables, figures, outlines only; no ghostwritten report prose).
5. Not done and needing the user's decision: training the generator on unlabelled test audio (transductive; departs from the clean-test protocol).

## 8. Cluster facts checked on 14 Nov 2026

- Submit from the project root; logs go to `hpc/logs/`. Environment in job scripts: `source /lustre-flash/apps/spack/share/spack/setup-env.sh; spack load anaconda3@2023.09-0`, interpreter `.venv/bin/python` (Python 3.11.5, PyTorch 2.0.1+cu118, timm 0.4.5). Compute nodes are offline (`HF_HUB_OFFLINE=1`). `diffusers` is not installed, which is why the U-Net is hand-written.
- Limits: `l40` max 2 days, 4 running / 5 submitted jobs per user; `a40` max 4 days, 3 running / 6 submitted, 2 GPUs per job; `dgx` max 6 days, 4 running / 5 submitted; `interactive` 4 h, 2 jobs.
- QOS-wide submit caps (all users together): `l40` 70, `dgx` 90, `a40` 100. When a cap is full, sbatch fails with `QOSGrpSubmitJobsLimit`; retry later or use another partition. Short jobs on `l40` and `interactive` started within minutes once accepted.
- Queues were long: `--test-only` estimated about 8 h on `l40`, about 5.5 h on `dgx`, under 1 h on `interactive`. `a40` refused with `QOSGrpSubmitJobsLimit` (the QOS-wide cap was full) and `debug` refused with `AssocGrpSubmitJobsLimit`. Keep `--time` tight so jobs backfill.
- The login node is very slow: `import torch` did not finish in 150 s. Do no testing there; use jobs.
- One GPU per job, no `--exclusive`, no `--nodelist`. Confirm with the user before large arrays.

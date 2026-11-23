# Evaluation protocol and what counts as a result

**Data.** ICBHI 2017 official recording-level split, cycles cut with the annotation files, resampled to 16 kHz, repeat-padded or cut
to 5 s: 4,142 training and 2,756 test cycles (`Classification/prepare_official_split.py`). Patients 156 and 218 occur on both sides, as
in the official protocol.

**Metric.** Sp = recall of normal; Se = recall over the three abnormal classes pooled; Score = (Se + Sp) / 2 (official ICBHI
definition, not the cycle-micro average). Reported at the **last** training epoch (no test-selected epoch), so the numbers are
lower than best-epoch numbers in the literature (our E1: best 61.25, last 58.71).

**Seeds and pairing.** Arms are compared with the same training seed (paired t-test and Wilcoxon on the per-seed differences).
Ensembles average probabilities; ensemble differences use a paired bootstrap of the test cycles (2,000 resamples, seed 20261118)
against a **size-matched** real-only ensemble.

**Prior correction.** `p_c ← p_c · π_real,c / π_mix,c`, with `π_mix` the class counts of the training mix (real + added rows) and
`π_real` the official-train counts. Weights are applied to each model's probabilities for single-model scores and to the averaged
probabilities for ensembles (the look-8 definition; renormalising before averaging changes T_K3×5 from 61.22 to 60.83).

**Test-set use.** The official test set was opened ten times during the novelty phase (`TEST_LOOK_REGISTER.md`): nine to evaluate arms or
corrections and one for post-hoc diagnostics. "Declared" means the arm, setting and comparison were written in `EXPERIMENT_LOG.md`
before the number existed. Validation for selection used 12 patient-disjoint training patients (504 cycles), which did not predict test
performance (`results/final/figures/F8_validation_vs_test.png`).

**Transductive use.** T_K3's generator was fine-tuned on the 2,756 *unlabelled* test cycles (class = null) and its synthetic stethoscope
mix copies the test device histogram. No test labels enter training. M3 and K3 do not use test audio.

**Controls that bound the claim.**
1. M1b (real duplicates, same balancing, no synthetic data) + correction: 58.30 — correction alone does not help.
2. M0 × 10 ensemble 60.79 vs T_K3 × 10 ensemble 60.98: no ensemble gain.
3. Bias sweep: M0 with one normal-class bias set to T_K3's specificity: Δ = −0.10 [−0.65, +0.44] (`results/final/diagnostics/`).
4. Near-copy check: T_K3's synthetic cycles are not closer to test cycles than real training cycles are (probe-space ratio 1.20).

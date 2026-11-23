# report_extra_1122 — post-hoc diagnostics on saved outputs (22 Nov 2026)

Task prompt: `prajna_task/PROMPT_FOR_PRAJNA_AGENT.md`. Nothing was trained. No file in `artifacts/` was changed. Nothing was selected or tuned on the test set. This is logged as look 10 in `deliverables/logs/TEST_LOOK_REGISTER.md`.

## What ran

| Task | SLURM job | Partition / node / GPU | Wall time | Status |
|---|---|---|---|---|
| A bias sweep + probs copy | 409355 | dgx / cn15-dgx / 1 x A100-SXM4-80GB | 00:00:35 | COMPLETED |
| B near-copy distances | 409356 | dgx / cn15-dgx / 1 x A100-SXM4-80GB | 00:04:18 | COMPLETED |
| C same-probe gate | 409357 | dgx / cn15-dgx / 1 x A100-SXM4-80GB | 00:03:27 | COMPLETED |

The job script is `scripts/report_1122.slurm`, also at `hpc/slurm/report_1122.slurm`. Its environment is copied from `nov_probe.slurm`: spack anaconda plus `.venv`, `HF_HUB_OFFLINE=1`. SLURM logs are in `hpc/logs/rep1122_{A,B,C}.<jobid>.{out,err}`. The `.err` files contain only library warnings (CuDNN nvrtc workaround and similar).

## Deviations and fixes

1. **Task A ran as a 1-GPU batch job, not a CPU `srun`.** PRAJNA has no CPU-only partition: `sinfo` lists dgx, dgx-mpi, a40, l40, interactive and debug. Submitting to l40 returned `QOSMaxSubmitJobPerUserLimit` with no jobs of mine in the queue. I first submitted to a40 (jobs 409350–409352). Their estimated starts were 18:17 to 19:08, so I cancelled those three jobs of mine and resubmitted on dgx, which had a free node.
2. **`bias_sweep.py` bug fix (one change; definitions untouched).**
   - **Bug:** `ensemble_nov.run_dir` falls back to a glob `*nov_final_{arm}_seed{s}`. For K3 seeds 6–10, which don't exist, that pattern matches `icbhi_ast_ce_nov_final_T_K3_seed{s}`. As shipped, "K3_corrected" would therefore have included five T_K3 runs.
   - **Fix:** `load()` now accepts a run only when the directory name is exactly `icbhi_ast_ce_nov_final_{arm}_seed{s}` or `nov_final_{arm}_seed{s}`.
   - **Check:** after the fix, K3 and M1b have 5 seeds each, and M0, T_K3 and M3 have 10 each (`meta.json`).
3. **`nn_test_copy.py` ran unchanged.**
4. **New helper scripts:**
   - `copy_probs_1122.py` (Task A step 2) asserts that each copied run directory has the exact expected name.
   - `same_probe_1122.py` (Task C).
5. **Reading `matched_summary.csv`:**
   - M1b_corrected and K3_corrected have only seeds 1–5. Their rows labelled `seeds=1-10` therefore have n = 5 and are identical to their `1-5` rows.
   - For these two arms the `ens5` rows are compared with the 10-model M0 ensemble curve, which is how the script defines it.
   - I didn't change either behaviour.
6. **Task C preprocessing.** The DiffWave wavs were converted two ways, and both are reported:
   - `cache_fbank`: repeat-pad or crop to 5 s with `cut_pad_sample_torchaudio` (`pad_types=repeat`), then `generate_fbank`. This is the `cache_fbank.py` path used for the prepared real cycles.
   - `orig_gate`: `inference/predict.preprocess`, which computes the fbank on the raw clip and then bilinearly resizes it to 498 × 128.

   Checked against the earlier gate numbers (E1 seed 1 `best.pth`), the probes match the earlier gates as follows:
   - **v1:** `cache_fbank` reproduces the earlier 6.01 % (wheeze) and 2.35 % (both) exactly.
   - **v2:** `orig_gate` reproduces the earlier 21.67 % and 2.5 % exactly.
   - **C0 probe on the SpecDiff pool:** it reproduces the earlier 89.15 % and 82.68 % exactly.
   - **E1 real-test recall:** wheeze recall is 28.57 %, the same as before.
7. **Clip sources:**
   - v1 (20k steps): `artifacts/synth/v1/raw/class{2,3}`, 499 and 637 clips.
   - v2 (33,868 steps): `artifacts/diag/v2/synth_pool/class{2,3}`, 120 and 120 clips.

   The `test_vocoder` clips (re-synthesised test cycles) were not scored.

## Task A — bias sweep (`bias_sweep/`)

**Sanity check:** this passed. Max |difference| from the published per-seed Scores was 0.00 for both M0_raw (10 seeds) and T_K3_corrected (10 seeds).

| Comparison | Seeds | n | Mean diff | 95% CI | p_t | p_Wilcoxon | Wins |
|---|---|---|---|---|---|---|---|
| T_K3_corrected − M0_at_matched_Sp | 1–10 | 10 | −0.10 | [−0.65, +0.44] | 0.674 | 0.492 | 4 |
| T_K3_corrected − M0_at_matched_Sp | 1–5 | 5 | +0.01 | [−1.19, +1.20] | 0.989 | 0.813 | 2 |
| T_K3_corrected − M0_at_matched_Sp | 6–10 | 5 | −0.22 | [−0.95, +0.52] | 0.460 | 0.438 | 2 |
| M3_corrected − M0_at_matched_Sp | 1–10 | 10 | −0.56 | [−1.15, +0.03] | 0.060 | 0.063 | 3 |
| M3_corrected − M0_at_matched_Sp | 1–5 | 5 | −0.71 | [−1.50, +0.08] | 0.066 | 0.125 | 1 |
| M3_corrected − M0_at_matched_Sp | 6–10 | 5 | −0.41 | [−1.69, +0.87] | 0.428 | 0.375 | 2 |
| M1b_corrected − M0_at_matched_Sp | 1–5 | 5 | −1.51 | [−2.82, −0.19] | 0.033 | 0.063 | 0 |
| K3_corrected − M0_at_matched_Sp | 1–5 | 5 | −0.13 | [−1.68, +1.43] | 0.831 | 1.000 | 2 |
| T_K3_corrected − M0_oracle_Score | 1–10 | 10 | −1.12 | [−1.83, −0.41] | 0.006 | 0.010 | 1 |

**Ensemble rows** (10-model M0 ensemble curve unless stated):

| Arm ensemble | Se / Sp / Score | M0 ensemble at matched Sp | M0 ensemble oracle |
|---|---|---|---|
| T_K3_corrected ens10 | 40.27 / 81.70 / 60.98 | −0.76 | −0.88 |
| M3_corrected ens10 | 38.91 / 82.90 / 60.91 | −0.72 | −0.96 |
| M1b_corrected ens5 | 34.15 / 83.34 / 58.75 | −2.80 | −3.11 |
| K3_corrected ens5 | 38.74 / 81.51 / 60.12 | −1.57 | −1.74 |

**Where the rest is:** comparisons at matched Se, and against M0's own raw Score, are in `matched_summary.csv`. Per-seed values are in `matched_per_seed.csv`.

**Raw probabilities:** `probs/` holds 40 raw (uncorrected) `probs_test_last.npy` copies (M0, T_K3 and M3 seeds 1–10; M1b and K3 seeds 1–5). `test_order.csv` has 2,756 rows, in `test_labels.test_order()` order.

## Task B — near-copy distances to test (`nn_test/`)

These are cosine nearest-neighbour distances. The probe is C0 seed 1 `last.pth`. The train → test baseline median is 0.25197 in probe space and 0.03814 in log-mel space.

| Pool | Space | ratio_test | near_copy_p1 | near_copy_p5 |
|---|---|---|---|---|
| T_K3 pool `full_tt_s20000_w2.0_target` (8,400) | probe | 1.203 | 0.0051 | 0.0193 |
| T_K3 pool | log-mel (2×2 pooled) | 0.866 | 0.0002 | 0.0323 |
| Control `full_s50000_w3.5_empirical` (12,500) | probe | 1.157 | 0.0060 | 0.0278 |
| Control | log-mel (2×2 pooled) | 0.976 | 0.0007 | 0.0206 |

Values by class are in `nn_test_summary.csv`. The teacher file `teacher_finalM0.npy` has shape (8,400, 4) and is row-aligned with `pool.csv`; the script asserts this.

`pseudo_label_check.json`, T_K3 pool, probe space:
```
nearest_test_label_matches_intended              0.4269
n_close_p5                                       162
close_nearest_test_label_matches_intended        0.7654
teacher_argmax_matches_intended                  0.6892
teacher_argmax_matches_nearest_test_label        0.5304
close_teacher_argmax_matches_nearest_test_label  0.9444
```
`closest_pairs.npz` holds the 6 closest pairs of T_K3 synthetic and test log-mels, in float16.

## Task C — same-probe gate (`same_probe/gate_same_probe.csv`)

Intended-class accuracy, %. Real test recall of each probe:
- **C0:** wheeze 36.36, both 23.08.
- **E1:** wheeze 28.57, both 16.78.

| Generator | Preprocessing | C0 wheeze | C0 both | E1 wheeze | E1 both |
|---|---|---|---|---|---|
| DiffWave v1 20k | cache_fbank | 10.02 | 4.55 | 6.01 | 2.35 |
| DiffWave v1 20k | orig_gate | 20.04 | 7.06 | 9.02 | 3.77 |
| DiffWave v2 33,868 | cache_fbank | 22.50 | 7.50 | 22.50 | 5.00 |
| DiffWave v2 33,868 | orig_gate | 33.33 | 10.00 | 21.67 | 2.50 |
| SpecDiff full_s50000_w3.5 | (cached log-mel) | 89.15 | 82.68 | 88.93 | 73.13 |

The SpecDiff normal and crackle rows for both probes are in the CSV.

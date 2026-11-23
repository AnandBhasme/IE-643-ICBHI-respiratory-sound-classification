# Register of every look at the official ICBHI test set (novelty phase)

Official split, 2,756 test cycles, last training epoch, Score = (Se + Sp) / 2. Compared with M0 (real only, AST + CE).
"Declared" means the arm, setting and comparison were written in `EXPERIMENT_LOG.md` before the test number existed.
Nothing tested is left out of this list.

| # | Date (IST) | What was tested | Declared before? | Transductive? | Result (single-model mean ± SD; 5-model ensemble) | Verdict |
|---|---|---|---|---|---|---|
| control | 15 Nov 03:15 | M0 real only, M1b real duplicates balanced (no synthetic data) | yes | no | M0 58.33 ± 1.42 / 59.78; M1b 57.45 ± 0.83 / 58.48 | reference |
| 1 | 15 Nov 04:10 lock | M3f (primary, chosen on validation +1.6), M3, M6: SpecDiff class balancing | yes (validation lock) | no | M3f 56.97 ± 0.63 / 58.44; M3 57.67 ± 1.25 / 58.74; M6 55.12 ± 0.63 / 55.96 | negative |
| 2 | 18 Nov | Fusion of AST with the generator used as a classifier (r3 25k, k = 25), plain and transductive generator | yes | r3t yes | fused single 57.63 / 57.63; ensemble 57.93 / 57.75 (vs 59.78); generator alone 45.2 / 45.0 | negative |
| 3 | 18 Nov | K3 soft labels (non-transductive); T_K3 transductive pool + test-device mix + soft labels | yes | T_K3 yes | K3 57.80 ± 1.14 / 58.63; T_K3 58.55 ± 0.52 / 58.85 | neutral |
| 4 | 18 Nov | X3_tt: SDEdit translation of train cycles to the test stethoscope mix | yes (approved by user) | yes | 56.59 ± 1.25 / 57.56 | negative |
| 5 | 18 Nov | Test-time stethoscope normalisation (regenerate test cycles, average) | yes, primary t0 0.2 / AKG | yes | 58.19 (M0 58.33); secondary grid 58.22-58.50, not selected | neutral |
| 6 | 18 Nov | A3: base-paper AFT (real-vs-generated domain head) + M3 pool | yes | no | 56.42 ± 0.80 / 57.42 | negative |
| 7 | 18 Nov | A3t: AFT + transductive pool | yes | yes | 57.48 ± 1.10 / 58.59 | negative |
| control | 18 Nov | M0 seeds 6-10 (size-matched real-only ensemble) | yes | no | M0 10 seeds 58.46 ± 0.96; M0×10 ensemble 60.79 | showed earlier ensemble gains were ensemble size |
| 8 | 18 Nov 17:15 | Prior correction p_c · π_real,c / π_mix,c on saved probabilities of every arm (no parameter) | yes | as per arm | T_K3 60.03 ± 0.48 / 61.22; M3 59.17 ± 0.82 / 60.90; K3 59.44 / 60.12; M1b control 58.30 / 58.75; others in `results/test_arm_summary_seeds1-5.csv` | first positive |
| 9 | 18 Nov 17:40 | Replication: T_K3 and M3 seeds 6-10, same fixed correction, size-matched vs M0 10 seeds | yes (script dry-run on seeds 1-5 first) | T_K3 yes | T_K3 59.57 ± 0.64, paired +1.11 [+0.20, +2.02], p = 0.022, 8/10; ensemble 60.98 vs 60.79 (n.s.). M3 58.83, paired +0.37 (p = 0.29) | per-model gain holds, ensemble gain does not |
| 10 | 22 Nov | Look 10 (22 Nov): post-hoc diagnostics on saved outputs (bias sweep on M0, near-copy distances to test, same-probe gate). No model, threshold or arm was selected from them. | yes (task prompt) | uses test audio/labels for scoring only | see `report_extra_1122/README.md` | diagnostic |

## Disclosures that go with these numbers

- Nine looks plus two control batches. The final claim (look 9) was declared before its five new seeds were trained, but the method (look 8) was chosen after looks 1-7 had failed.
- T_K3's generator saw the unlabelled official-test audio (no labels), and its synthetic stethoscope mix copies the test device histogram.
- The prior correction's only inputs are class counts of the training mix; no test labels, no tuned parameter.
- Ensemble-combination sensitivity: weights applied per model then renormalised gives T_K3×5 60.83 instead of 61.22; the look-8 definition was kept.
- The 5-seed estimate (+1.70) shrank to +1.11 with 10 seeds; report the 10-seed number.
- Validation (12 training patients) did not predict test (`figures/F8`); on validation the correction helps the M3 family and not the soft-label K family.

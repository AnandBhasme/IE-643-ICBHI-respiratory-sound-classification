# Experiment log

## 2026-10-24 08:06 IST — Phase 0 reconnaissance and Phase 1 preparation

- **Job:** none; no SLURM work was submitted.
- **Partition/QOS:** none used. Live controller query confirmed `debug/debug`, `l40/l40`, `a40/a40`, and `dgx/dgx` as available choices with limits recorded in `RECON.md`.
- **Config/data:** inspected project code and the official split file presence; did not process audio or modify the existing split data. Added `Classification/prepare_official_split.py` and `Classification/verify_split.py` for official recording-level splitting and acceptance checks.
- **Outcome:** both new scripts passed a Python syntax compilation check. No split output or experiment metrics exist yet.
- **Deviation/blocker:** the plan requires split processing and verification via SLURM. `$SCRATCH` is unset and `/scratch/$USER` is absent, so the writable project output path needs confirmation. PyTorch is not installed in the available Python 3.11.5 environment; the plan requires user confirmation before creating a venv/installing dependencies. No GPU jobs were run.

## 2026-10-24 08:20 IST — Workspace environment and first SLURM jobs

- **User decisions:** use the project workspace for code and artifacts; prefer `l40/l40`; create a Python 3.11 venv and install dependencies.
- **Storage/layout:** created `hpc/{slurm,configs,logs}`, `artifacts/{data,splits,runs,diffwave,synth,results}`, root and AFT `.gitignore` files, `hpc/README.md`, dependency recipe and lock file. The venv is `.venv` under the project workspace. Workspace filesystem is Lustre home; numeric quota limits show 0/unlimited, with 116.2 GiB already used at inspection. No `/scratch/$USER` path exists.
- **Environment:** installed Python 3.11.5 stack `torch 2.0.1+cu118`, `torchvision 0.15.2+cu118`, `torchaudio 2.0.2+cu118`, `timm 0.4.5`, NumPy 1.26.4 and project requirements. `pip check` passed. CUDA is unavailable on the login node, as expected; GPU validation is via SLURM.
- **Scheduler:** L40 `sbatch --test-only` predicts a start for the split-preparation job. Debug test-only returned `AssocGrpSubmitJobsLimit`; L40 test-only succeeded without an explicit account. Current L40 jobs do not exceed one GPU task at a time.
- **Jobs:** 362876 `ie643_split` is running on L40 to create `artifacts/data/official_icbhi`; 362877 `ie643_verify` waits on 362876 via `afterok`; 362878 `ie643_smoke` failed in 49 seconds because its script could not import the repo `models` package. Fixed its Python import path and resubmitted as 362880. No old data or checkpoints were deleted or overwritten.

## 2026-10-24 08:24 IST — Official split accepted; baseline preparation

- **Jobs:** 362876 completed on `l40/l40` in 00:04:58. 362877 completed on `l40/l40` in 00:00:05.
- **Split acceptance:** 4,142 train / 2,756 test cycles; classes train `{0:2063,1:1215,2:501,3:363}`, test `{0:1579,1:649,2:385,3:143}`; 539/381 recordings; zero cycle filename overlap; shared patients exactly 156 and 218. Saved output to `artifacts/splits/verify_split.txt`. The split is recording-level.
- **Checks/code:** the two required ICBHI metric tests passed (always-normal gives Sp=1, Se=0, Score=0.5; perfect gives 1/1/1). Updated classification evaluation to use the shared confusion-matrix metric helper, log per-epoch class recalls/confusions and binary sensitivity to `metrics.csv`, save `config.json` with SLURM job ID, and keep only one `best.pth` file instead of a new 1 GB file for every improving epoch.
- **Next jobs:** 362880 is the corrected one-GPU AST smoke check, pending resources on L40. `train_e1.slurm` is ready for 50-epoch E1 and passed L40 `sbatch --test-only`, with estimated start 2026-10-24 09:41 IST; it has not been submitted while the AST smoke check is pending.
- **Deviation:** artifacts are in workspace `artifacts/` per user direction, not `/scratch`; the cycle WAV dataset uses about 1.1 GiB. The first smoke attempt 362878 failed on Python import path and was fixed before resubmission; no model training has run yet.

## 2026-10-24 08:23 IST — Queue baseline runs behind smoke check

- **Jobs:** 362880 AST smoke is pending on `l40/l40` with reason `Resources`; `scontrol` estimates 09:39 IST. Submitted 362883 E1 and 362885 E2 with `afterok:362880`, each one GPU, 8 CPUs, 48 GiB RAM, 12-hour limit. They will not start if the smoke check fails. Test-only estimates were 09:41 for E1 and 11:40 for E2.
- **E1:** CE, seed 1, 50 epochs, batch 32, LR 5e-5, weight decay 1e-6, 5-second cycles, AudioSet AST, EMA beta 0.5.
- **E2:** Patch-Mix CL with the same data and optimizer config, temperature 0.06, alpha 0.2, mix beta 1.0. Static inspection found the parser's default `target_type=project1_project2block` has no matching implementation branch and would leave `proj1` unset. The script explicitly uses the repository-supported `project_block` path, which detaches the original projected anchor and trains the mixed projection.
- **Status:** no classifier training has started. Baseline jobs remain dependency-held until smoke passes.

## 2026-10-24 08:27 IST — DiffWave preparation and classifier queue update

- **Jobs:** 362880 AST smoke remains pending for L40 resources. E1 362883 and E2 362885 remain dependency-held (`afterok:362880`). DiffWave train-only dataset job 362890 is running on `cn45-l40`; 362891 waits via `afterok:362890`.
- **DiffWave data:** 362890 is generating class-organized 4-second, 16 kHz train WAVs under `artifacts/data/diffwave/`; output was 184 MiB at the last check. No job error output so far.
- **Resource plan:** one GPU per job; DiffWave training starts after CPU preprocessing finishes. Workspace artifacts are retained as requested.
- **No model results yet:** neither the AST smoke nor E1/E2 classifier training nor DiffWave training has completed. Continue with the smoke gate before interpreting any training metrics.

## 2026-10-24 08:31 IST — Preflight checks and queue estimate

- **Checks:** Python syntax compilation passed for the classifier metrics/training and DiffWave changes; both metric unit tests passed again; all SLURM scripts passed `bash -n`.
- **Queue:** AST smoke 362880 remains pending for resources with controller estimate 09:39 IST. The paired classifier runs are dependency-held, not policy-blocked. L40 has 52 idle CPUs but only six nodes (shared GPU scheduling); A40 is available as an alternate, but the user asked to prefer L40. We remain on that preference pending the plan's 60–90 minute fallback threshold.
- **DiffWave data:** 362890 remains active; `artifacts/data/diffwave/` grew from 184 MiB to 434 MiB in about a minute. No errors reported.
- **Outstanding plan check:** the AST smoke is a forward/backward model-stack check. A separate 1–2 epoch classifier timing smoke has not yet run; use the first short classifier run to measure epoch time and memory before scheduling follow-on seeds.

## 2026-10-24 08:34 IST — Classifier smoke check constrained by submit cap

- **Plan check:** the queued E1/E2 jobs are the 50-epoch official-split runs; a distinct 1–2 epoch classifier timing smoke has not run.
- **Scheduler constraint:** `sbatch --test-only hpc/slurm/smoke_classifier.slurm` was rejected with `QOSMaxSubmitJobPerUserLimit` while five jobs were already submitted. Making room would require canceling pending E1, but automatic approval review rejected that cancellation as not explicitly authorized. No cancellation or replacement occurred; jobs 362880/362883/362885/362890/362891 remain as previously submitted.
- **Deviation:** keep the existing E1/E2 jobs and record per-epoch time/memory from their logs/metrics instead of canceling an approved full baseline to make room. The smoke script is prepared but not submitted. This is an automatic-review constraint, not an HPC partition/QOS mismatch.
- **Validation:** syntax compilation, `bash -n`, and the two metric unit tests pass.

## 2026-10-24 08:38 IST — AST GPU smoke passed; baseline runs started

- **Smoke job:** 362880 completed on `l40/l40` in 00:02:12 (GPU node `cn42-l40`, device NVIDIA L40S). Offline imports and AudioSet AST weights loaded; Kaldi fbank shape `(98, 128)`; AST forward/backward returned four logits; peak allocated CUDA memory 1.017 GiB. Exit 0. A cuDNN warning noted missing `nvrtc.so` workaround support, but the operation completed successfully.
- **Baselines:** after the smoke passed, E1 job 362883 (CE) and E2 job 362885 (Patch-Mix CL) both started on `cn42-l40`; full 50-epoch runs, one GPU each, seed 1, settings in their SLURM scripts. The separate classifier epoch-timing smoke remains omitted because the QOS submit cap prevented adding a job without canceling a baseline; this deviation is recorded above.
- **DiffWave preparation:** the training-only cycle stage completed and reported 4,142 clips. Spectrogram preprocessing remains active in job 362890; the output tree measured 633 MiB at the last check. DiffWave training job 362891 remains afterok-dependent on preprocessing.

## 2026-10-24 08:42 IST — First E1 epoch recorded

- **E1 job 362883:** epoch 1 completed in 45.93 s (train portion); 4,142 train / 2,756 test cycles loaded. Best-epoch-by-test-Score so far: Sp 48.32%, exact-class abnormal Se 59.30%, ICBHI Score 53.81%; accuracy 53.01%; binary abnormal sensitivity 79.69%. This is an interim epoch-1 result, not a final/best-over-50-epochs claim. Checkpoint `best.pth` was written (about 1.0 GiB); `metrics.csv` includes class recalls, confusion matrix, and epoch duration.
- **E2 job 362885:** training started successfully; first epoch progress is visible, no error reported yet.
- **DiffWave job 362890:** still preprocessing the 4,142 train cycles into spectrograms; no error output. Training job 362891 remains dependency-held.

## 2026-10-24 08:47 IST — Baseline learning curves and DiffWave preprocessing

- **E1 job 362883:** epochs 1–4 completed. Epoch 1 score 53.81 (Se 59.30, Sp 48.32); epoch 2 score 55.99 (Se 49.79, Sp 62.19); epoch 3 score 56.15 (Se 46.81, Sp 65.48); epoch 4 score 53.30 (Se 52.08, Sp 54.53). Train portion is about 38–46 sec/epoch. These fluctuate and are interim only.
- **E2 job 362885:** epochs 1–2 completed. Epoch 1 score 49.81 (Se 23.87, Sp 75.74); epoch 2 score 51.66 (Se 53.36, Sp 49.97), roughly 73–82 sec/epoch train portion. Interim only.
- **DiffWave job 362890:** spectrogram preprocessing progress reached 100%; job is still finalizing at the last queue check. Job 362891 stays afterok-dependent.

## 2026-10-24 08:50 IST — DiffWave training data ready

- **Job 362890:** completed successfully on `l40/l40` in 00:11:03 (CPU preprocessing, 8 workers), exit 0. Confirmed 4,142 train WAV files and 4,142 `.npy` spectrograms; no test cycle was included. Output size 844 MiB under `artifacts/data/diffwave/`.
- **DiffWave training 362891:** now pending for L40 resources (`Resources`), afterok dependency satisfied. It requests one GPU, 12 hours, max 20,000 optimizer steps, checkpoint every 5,000. No DiffWave model step has run yet.
- **E1/E2:** both seed-1 baseline jobs remain running. E1 job 362883 and E2 job 362885 have each used about six minutes elapsed at the last accounting check.

## 2026-10-24 08:53 IST — Interim baseline progress

- **E1 362883:** epoch 10 reached; best so far epoch 8, score 61.25 (Se 42.82%, Sp 79.67%); epoch 10 score 57.67. Per-epoch test fluctuations continue.
- **E2 362885:** epoch 6 reached; best so far epoch 5, score 58.08 (Se 46.30%, Sp 69.85%); epoch 6 score 57.80.
- **DiffWave 362891:** still pending for L40 resources. Controller estimate remains 09:39 IST at last check. Both baseline jobs are healthy and not near their 12-hour limits.

## 2026-10-24 08:55 IST — Early learning curves

- **E1 362883:** reached epoch 15. The highest test-selected score so far remains 61.25 at epoch 8 (Se 42.82%, Sp 79.67%); epoch 15 is 59.99 (Se 40.19%, Sp 79.80%). Train accuracy is above 90%, while test accuracy is about 63%, showing an early train/test gap. This is interim.
- **E2 362885:** reached epoch 8. Best so far remains epoch 5, Score 58.08 (Se 46.30%, Sp 69.85%); epoch 8 is 55.33 (Se 51.32%, Sp 59.34%). Interim; report test-selected best only with the selection caveat and compare last epoch separately when the job completes.
- **DiffWave 362891:** pending resources with the last recorded controller estimate 09:39 IST. No queue-policy error.

## 2026-10-24 09:00 IST — Inference implementation and queued validation

- **Implementation:** added offline `Classification/inference/predict.py`. It reproduces the training fbank settings and normalization, accepts a cycle WAV or array, loads AST encoder plus separately trained classifier head from a repo checkpoint, and returns four class probabilities.
- **Validation job:** created `hpc/slurm/test_inference.slurm` and a three-clip smoke runner; it checks GPU and CPU predictions on three official test cycles and records peak GPU allocation. Script syntax and Python compilation passed; L40 test-only estimated a start at 11:38. Submitted job 362900 with `afterok:362883`, so it cannot read the changing checkpoint before E1 finishes. It adds one queued task under the QOS submit cap.
- **Current inference is not yet validated:** wait for job 362900 and inspect numerical output before presenting it as ready.

## 2026-10-24 09:04 IST — DiffWave training started

- **Job 362891:** started on `l40/l40`, node `cn44-l40`; one GPU, 8 CPUs, 48 GiB, 12-hour cap. It points to the official-train-only 4,142-cycle WAV/spec dataset and is configured for 20,000 steps, saving every 5,000 and on SIGTERM. No checkpoint has appeared yet; training is still initializing/working through the first steps. Job output has only the NumPy deprecation warning, no error.
- **Baselines:** E1 362883 at epoch 18 and E2 362885 at epoch 10 at the last check; no failures. Inference smoke 362900 remains held until successful E1 completion.

## 2026-10-24 09:10 IST — Three GPU jobs active

- **E1 362883:** epoch 22, best-by-test-Score remains epoch 8 at 61.25 (Se 42.82%, Sp 79.67%); epoch 22 score 58.69.
- **E2 362885:** epoch 13, best so far epoch 11 at 59.27 (Se 48.94%, Sp 69.60%); epoch 13 score 54.84.
- **DiffWave 362891:** active on `cn44-l40`, about 4 minutes elapsed. First run directory and args exist, no periodic checkpoint yet (first is at step 5,000). Scheduler shows active CPU and memory use (about 5.1 GiB RSS); no error output. Checkpoint and throughput remain unverified until a save/log update appears.
- **Concurrency:** E1, E2, and DiffWave are the three active/queued GPU requests; inference job 362900 waits on E1.

## 2026-10-24 09:40 IST — E1 completed; inference validation passed

- **E1 job 362883:** completed all 50 epochs on `l40/l40`, exit 0, elapsed 00:40:43, host MaxRSS 3.60 GiB. Best epoch by test Score was 8: Se 42.82%, Sp 79.67%, Score 61.25%. Last epoch 50: Se 47.07%, Sp 70.36%, Score 58.71%, accuracy 60.41%. Best and last are reported separately; best uses test-set selection.
- **Inference job 362900:** completed on `l40/l40`, exit 0 in 00:03:14. Ran three official test cycles using both CPU and NVIDIA L40S. Per-file CPU/GPU probabilities agree to about 1e-4 or better; peak GPU allocation 0.526 GiB. This validates the new offline predictor on this checkpoint. The first three sorted test clips all happen to be normal class clips, so this check covers preprocessing/loading but is not a class coverage evaluation.
- **E2 job 362885:** still running, 34 epochs completed at the last metrics read. Current best so far epoch 19: Se 40.53%, Sp 79.42%, Score 59.97%; last logged epoch 34: Se 48.94%, Sp 63.20%, Score 56.07%. Values may change before the 50-epoch run ends.
- **DiffWave 362891:** still running; event log shows training loss/gradient scalars through step 150 at an earlier check, no checkpoint saved yet (first planned save at step 5,000). Current elapsed around 32 minutes. Output reports a cuDNN `nvrtc.so` workaround warning; training continues.
- **Results artifacts:** added `Classification/collect_results.py`; current `artifacts/results/results.csv` summarizes available metrics and is refreshed from per-run CSVs. E2 remains an in-progress row; regenerate after completion.

- **Timestamp correction:** the preceding section header used 09:40 IST in error; the client-local time at that update was about 09:20 IST. Job states, metrics, and elapsed accounting in the preceding entry are accurate.

## 2026-10-24 09:22 IST — Resume status after interruption

- **Completed:** official split builder/verifier (4142/2756), AST GPU stack smoke, E1 full 50 epochs (job 362883), and inference validation (job 362900).
- **E1 final:** best epoch 8 by test Score: Se 42.82%, Sp 79.67%, Score 61.25%; last epoch 50: Se 47.07%, Sp 70.36%, Score 58.71%; job elapsed 00:40:43, host MaxRSS 3.60 GiB.
- **Inference:** 3 test cycles ran on CPU and L40S; CPU/GPU probabilities agree; GPU peak allocation 0.526 GiB; exit 0 in 00:03:14. All three selected files are normal-class recordings, so the smoke checks inference plumbing rather than class coverage.
- **E2:** job 362885 remains running; metrics currently have 34 logged epochs; current best is epoch 19, Se 40.53%, Sp 79.42%, Score 59.97%; last logged row (epoch 34) was Se 48.94%, Sp 63.20%, Score 56.07%. Wait for job completion to report final last epoch.
- **DiffWave:** job 362891 remains active on L40. TensorBoard event data has advanced to step 1,750, with roughly 50 seconds per 50 steps (~1 sec/step); the first 5,000-step checkpoint is still pending. At this rate, 20,000 steps should fit within its 12-hour limit, subject to variability.
- **Summary file:** `artifacts/results/results.csv` currently has E1 and in-progress E2 rows. Rerun `Classification/collect_results.py` after E2 completes.

## 2026-10-24 09:37 IST — Synthetic and E3 dependency chain queued

- **E2 362885:** epoch 45/50 at the latest check; last logged test score 56.74 (Se 46.47%, Sp 67.00%). Expected a few more minutes at its recent pace.
- **DiffWave 362891:** running 49 minutes; previous event-log rate was about one optimizer step per second. No checkpoint output yet; periodic saves begin at step 5,000.
- **Queued jobs:** 362928 `ie643_synth_v1` (generate 499 class-2 and 637 class-3 clips using fast six-step sampling), afterok DiffWave; 362930 `ie643_mix_data`, afterok generation; 362932 E3 AST+CE, N=1,000 minority-only real+synthetic, seed1, afterok data assembly. Each new script passed syntax checks; generation, data assembly, and E3 each had L40 `sbatch --test-only` validation where available. The current run log for E2/DiffWave remains the source of job progress.
- **Generation manifest/mix logic:** added `prepare_synth_mix.py`, which verifies the official train manifest, requires sufficient class-specific synthetic files, and constructs `training/real_with_gen` using symlinks plus explicit used-item and synthetic manifests. It refuses to overwrite an existing mixed dataset.

## 2026-10-24 11:04 IST — DiffWave training audit (job 362891)

- Confirmed the job runs the repository's conditional DiffWave noise-prediction objective with 50 diffusion steps, batch 16, Adam at 2e-4, and a fixed seed. The active config points to 4,142 generated WAV clips and their 4,142 mel sidecars; these are built from the official training manifest only, with classes separated in `label0`–`label3` directories. No test clips are in the training input.
- Checkpointing is step-based every 5,000 updates, with resume support restoring model, optimizer, scaler, and step. The L40 job is active; the 5,000-step checkpoint exists and TensorBoard loss/gradient scalars are finite through step 7,500 at the audit check.
- Deviations/risks: this run is capped at 20,000 steps, below the plan's 100k–150k target. Also, `crop_mel_frames=62` means each optimizer update trains on about 0.99 s paired audio/mel crops from 4 s source clips; inference later conditions on the full 4 s mel. This inherits the repository's crop setting and may limit long-context quality. Do not treat generated samples as validated until the planned audio/mel/classifier quality gate is run. If quality is inadequate, continue from the latest checkpoint with corrected settings where possible rather than restarting weights from scratch.

## 2026-10-24 14:48 IST — DiffWave and E3 complete; E4/QC/E5 queued

- **DiffWave 362891:** completed on `l40/l40`, exit 0, elapsed 05:52:25. Reached 20,000 updates; `weights-5000.pt`, `weights-10000.pt`, `weights-15000.pt`, and `weights-20000.pt` are present, and `weights.pt` points to step 20,000. Final logged L1 noise-prediction loss was 0.07475. This remains a 20k reduced-compute run, with one-second training crops and four-second inference conditioning as noted in the earlier audit.
- **Synthetic v1 generation 362928:** completed on L40 in 00:09:12, exit 0; produced 499 class-2 and 637 class-3 WAVs (1,136 total), conditioned on official-train mel files using the step-20k checkpoint. Warning: PyTorch reported its cuDNN `nvrtc.so` workaround; generation completed successfully.
- **Mix assembly 362930:** completed in 00:00:11, exit 0. Training pool counts are class0=2063, class1=1215, class2=1000, class3=1000 (499 generated class2 and 637 generated class3).
- **E3 362932:** completed on L40 in 00:54:40, exit 0, 50 epochs. Best test-selected result: Sp=71.75%, Se=47.92%, Score=59.84%; last epoch: Sp=68.08%, Se=46.81%, Score=57.45%. This did not beat E1's best Score=61.25%; test-based best-epoch selection applies.
- **Synthetic QC:** added `Classification/synth_qc.py` and `hpc/slurm/synth_qc.slurm` to compare real conditioning sources and generated classes 2/3 for waveform finite/clipping/RMS/peak, E1 intended-class prediction fraction, and mel grids. Job 363477 is running on L40. No QC outcome is claimed yet; class 1 has no synthetic pool in v1.
- **E4:** added an L40 script for AFT CE on the N=1000 minority mix; job 363474 is pending resources.
- **E5:** added the combined Patch-Mix CL + adversarial branch, separate `lambda_dis`, and a training domain-accuracy metric; `main.py` and QC script passed Python syntax compilation. Job 363482 is queued after successful QC and E4. E5 applies the discriminator to the un-mixed forward pass, as allowed in the plan.

## 2026-10-24 14:52 IST — Synthetic QC outcome and scheduling adjustment

- **QC job 363477:** completed on L40 in 00:03:49, exit 0. Audio is finite and has no near-full-scale clipping, but the planned classifier-consistency gate fails badly: E1 intended-class accuracy is 6.01% for 499 wheeze samples vs 97.67% on 300 real conditioning clips, and 2.35% for 637 both samples vs 93.31% on 299 real conditioning clips. Synthetic RMS means are 0.078 (wheeze) and 0.091 (both), below corresponding real means 0.180 and 0.213. Class 1 has no v1 synthetic pool. QC figures and `qc_summary.csv` are under `artifacts/synth/v1/qc/`.
- The one-run E4/E5 comparisons remain queued to document behavior with this weak v1 pool, per the fallback instruction; these results must not be described as evidence of good synthetic quality.
- **Scheduling:** reduced pending E4 and E5 walltimes to 2h using observed E3 runtime 00:54:40. Updated L40 test-only estimate for E4/E5 is 2026-10-24 21:34 local; E5 remains after successful E4 and QC. This is to improve backfill placement without over-requesting time.

## 2026-10-24 16:52 IST — E4/E5 moved to A40

- User requested using A40 rather than waiting on L40. Verified `a40` QOS is allowed and found one free A40 GPU with enough CPUs on `cn27-a40`; other currently visible A40 nodes had all GPUs allocated.
- Updated pending jobs 363474 (E4) and 363482 (dependent E5) from `l40/l40` to `a40/a40`. Reduced their walltime requests to 01:30 based on E3's measured 00:54:40 and current test-only start estimate. E4 scheduler start estimate is 2026-10-24 17:51 IST (latest test-only: 18:00 IST); E5 starts only after E4 and QC succeed. E4 remains pending for resources at the latest check.
- Updated `hpc/slurm/train_e4.slurm` and `train_e5.slurm` to reflect A40 partition/QOS and 90-minute walltime.

## 2026-10-24 19:20 IST — E4 complete; E5 retry submitted

- **E4 363474:** completed on A40 in 01:24:14, exit 0. Best test-selected metrics: Se=47.49%, Sp=74.54%, Score=61.02%. Last epoch: Se=49.11%, Sp=68.46%, Score=58.78%. `train_domain_acc` was logged each epoch.
- **E5 first attempt 363482:** failed on A40 at the end of the first training iteration. Root cause: EMA state list for adversarial runs is `[model, class_head, domain_head, projector]`, while the generic EMA update incorrectly selected index 2 (domain head) for the projector. No epochs/checkpoint were produced.
- Fixed the EMA selection to use index 3 for adversarial runs and index 2 otherwise. E5 retry uses a distinct run tag, A40/a40, 4 CPUs, 4 DataLoader workers, batch 32, and a 90-minute limit. Python syntax compilation passed.
- **E5 retry 363578:** submitted and confirmed running on `cn23-a40` at 19:19 IST. Await first epoch and continue monitoring.

## 2026-10-24 21:16 IST — E5 retry1 timed out; fresh full run submitted

- **E5 retry1 363578:** SLURM state `TIMEOUT`, elapsed 01:30:05. It completed epochs 1–33 of 50 before the 90-minute limit. Best test-selected result was at epoch 23: Se=45.45%, Sp=75.55%, Score=60.50%; epoch 33 Score=55.98%. This remains a partial run.
- A faithful checkpoint resume is not possible from `best.pth`: the checkpoint stores the AST, class head, and optimizer, but not the E5 projector or domain-discriminator weights; the optimizer state cannot reconstruct those missing model parameters. Do not describe retry1 as resumed or complete.
- Submitted a clean 50-epoch E5 seed-1 retry2, same official data/synthetic v1/N=1000/config, on the preferred `l40/l40` partition with 4 CPUs, 4 workers, batch 32, and a 02:30:00 measured walltime budget. `sbatch --test-only` estimated immediate start on `cn42-l40`; actual job **363673** started on `cn42-l40` at 21:16 IST. Its output is under `hpc/logs/ie643_E5_r2.363673.out`.
- This is the third E5 attempt only in the sense of scheduler jobs: the first failed on the EMA index bug, retry1 was truncated by its wall limit, and retry2 is a fresh full run. Preserve all run directories and logs.

## 2026-10-24 22:53 IST — E5 retry2 completed; packaging started

- **E5 retry2 363673:** completed 50/50 epochs on `l40/l40`, exit 0, elapsed 01:32:23. Best test-selected epoch 25: Se=40.27%, Sp=80.56%, Score=60.41%. Last epoch 50: Se=49.62%, Sp=61.43%, Score=55.52%. The best result does not improve both Se and Sp over E1 and has a lower Score than E1.
- Refreshed `artifacts/results/results.csv` from all completed run metrics. Added `artifacts/results/summary.md`, including the E1–E5 best/last comparisons, split protocol, synthetic-QC failure, and limits.
- Packaging job **363758** is running on L40 to create state-dict-only E1/E2/E4 checkpoints, plots, and a model card. Inference smoke job **363760** is dependency-held until packaging finishes; it will exercise three official test cycles on GPU and CPU using the slim E1 checkpoint.
- Copied the current results table and summary into `Code/Code/AFT/results/` as requested in the plan. Submitted baseline repeats E1 seed 2 (job **363770**) and E2 seeds 2/3 (jobs **363768**, **363769**), each dependency-gated on inference job 363760. The E1 seed-3 submission was rejected by `QOSMaxSubmitJobPerUserLimit`; retry after one active submission slot frees. Peak planned classifier GPU concurrency remains four.
- **Packaging 363758:** completed successfully in 00:02:23. Wrote slim state-dict checkpoints `artifacts/final_models/E1.pth` (335 MiB), `E2.pth` (335 MiB), and `best_synthetic_E4.pth` (333 MiB), plus the model card and first figure set.
- **Inference 363760:** completed successfully in 00:00:59 on L40S. Predicted three official test cycles on both GPU and CPU; CPU/GPU probabilities closely agreed and peak allocated GPU memory was 0.526 GiB. Inference was run on the slim E1 checkpoint.
- E1 seed 3 (job **363771**) was submitted after a QOS submission slot freed; E2 seeds 2 and 3 began on `cn42-l40` and `cn45-l40`. E1 seeds 2 and 3 remain queued. Current classifier GPU-job count is four at most.
- E1 seed jobs' first train epochs took about 90 s, but epoch 2 took about 38.5 s, close to E1 seed 1's steady pace. SLURM denied increasing the running jobs' time limit; based on the steady epoch time they remain expected to finish within 60 minutes. E2 seed-3 epoch 1 took 97 s and is progressing under its 01:45:00 allocation.

## 2026-10-24 23:15 IST — Detailed progress snapshot

- Added root `progress.md` with the verified official split, preprocessing/environment, DiffWave and synthetic-QC numbers, full seed-1 best/last metrics and class recalls, E5 retry history, artifacts, limitations, and live seed-2/3 status.
- At snapshot, E1 seeds 2/3 (jobs 363770/363771) had 14/50 epochs each; E2 seeds 2/3 (363768/363769) had 8/50 and 10/50. All four were running on L40. Latest observed steady train times were about 38.4–38.6 sec/epoch for E1 and 73.2–73.5 sec/epoch for E2.
- Current interim test-selected scores: E1 seed2 60.15 (best epoch 4), seed3 60.21 (epoch 7); E2 seed2 59.35 (epoch 6), seed3 58.37 (epoch 7). Do not treat these as final until the 50-epoch runs end.

## 2026-10-24 23:18 IST — Baseline repeats advancing

- Read-only scheduler check confirms E1 seeds 2/3 (363770/363771) and E2 seeds 2/3 (363768/363769) are all still RUNNING on L40; no jobs were changed.
- E1 jobs have each trained through epoch 19, with validation through epoch 18. Seed 2's latest validation is Se=44.94%, Sp=68.27%, Score=56.61%; its best remains epoch 4 (39.93%, 80.37%, 60.15%). Seed 3's latest is Se=55.31%, Sp=56.43%, Score=55.87%; best remains epoch 7 (47.83%, 72.58%, 60.21%). Recent train epochs took 38.25–38.41 s.
- E2 seed 2 has trained/validated through epoch 11 and currently has a new best at epoch 11: Se=48.68%, Sp=71.37%, Score=60.03%. E2 seed 3 has trained through epoch 13 and validated through epoch 12; latest Score is 54.50%, while its best remains epoch 7 (Score 58.37%). Recent train epochs took 73.2–73.3 s.
- At the observed epoch pace, remaining-time estimates are ~20 min for each E1 job and ~46–48 min for E2. The matching job wall limits remain 60 min (E1) and 1:45 (E2). These are projections, not guarantees; continue monitoring before marking completion.
- The detailed report at workspace root, `progress.md`, now includes this 23:18 snapshot and remains the authoritative running status report.

## 2026-10-24 23:55 IST — Day-2 continuation P0 and E1 seed repeats

- Read `Plan_v2_Continuation.md`, the full `PRAJNA_HPC_AGENT_RULES.md`, this experiment log, `progress.md`, and the original `plan.md` (the user clarified that “HPC training prompt” means `plan.md`). The separately named `HPC_Training_Prompt.md` does not exist; `plan.md` is the original project plan.
- Jobs 363770 and 363771 (E1 seeds 2/3, L40) completed 50/50 epochs successfully. Seed2 best epoch4: Se=39.93%, Sp=80.37%, Score=60.15%; last epoch50: Se=46.13%, Sp=68.65%, Score=57.39%, accuracy=59.03%. Seed3 best epoch7: Se=47.83%, Sp=72.58%, Score=60.21%; last: Se=48.51%, Sp=66.37%, Score=57.44%, accuracy=58.74%.
- Completed three-seed E1 mean ± sample SD: best Se=43.53±4.00%, Sp=77.54±4.31%, Score=60.53±0.62%; last Se=47.24±1.20%, Sp=68.46±2.00%, Score=57.85±0.75%. Seed1/2/3 best Scores=61.25/60.15/60.21; last Scores=58.71/57.39/57.44.
- P0 audit recomputed all confusion-matrix-derived measures over 417 logged epochs in 10 metrics files. Every matrix totals 2,756 test cycles with class supports [1,579, 649, 385, 143], and each stored Se/Sp/Score/accuracy/per-class recall/binary sensitivity agrees within 0.00002 percentage points. `artifacts/results/results.csv` was refreshed and synced to the project results copies. The E4 last Score issue is already corrected to 58.78%.
- E2 seeds2/3 remain running: jobs 363768/363769 on `cn42-l40`/`cn45-l40`; at 23:55 they were at 38/50 and 40 train / 39 validated epochs, elapsed 58:07 / 58:06. Interim best Scores were 60.74% (seed2, epoch27) and 59.09% (seed3, epoch38); last validated Scores 57.06% and 56.99%.
- Queued P1 diagnostic job **363821** and P2 DiffWave profile job **363823**, both L40 and dependency-gated on completion (any terminal state) of E2 jobs 363768/363769. Their `sbatch --test-only` estimates validated the scripts. At most four project GPU jobs are active/queued. No job has been stopped or modified.
- P1 adds `/diag/` exclusion assertion to the training dataset and writes test-derived diagnostic audio only under `artifacts/diag/v1/`. It compares raw and source-RMS-matched v1 samples, fast-vs-full sampling on training mels, and held-out vocoder diagnostics on 100 cycles per abnormal class. E1 seed1 lacks epoch-50 model weights (only best.pth was saved); this limitation is recorded in output metadata and the best checkpoint is used for classifier probes.
- Repository inspection found DiffWave already preloads the 4,142 training WAVs and mels into memory and enables cuDNN benchmark; P2 profile tests worker count/persistence, batch32 and BF16 instead of repeating that existing optimization. Added BF16 support and explicit checkpoint-path resume support for later segments. Full 50-step inference had an off-by-one timestep map; full sampling now maps directly to all 50 trained timesteps.
- Added E7 real-only imbalance modes `weighted_ce` and `balanced_sampler`, plus `hpc/slurm/train_e7_seed.slurm`. Weighted CE uses the prescribed normalized square-root inverse-frequency weights. Added compact `last.pth` saving for future training runs; older E1 seed1 and currently-running E2 repeats still only have their best checkpoints.
- `Plan_v2_Continuation.md` now drives priorities. Remaining order is E2 completion, P1/P2 results, E7 seeds, then any GPU slots for P4/P5. Deadline remains 25 October 2026 23:59 IST.

## 2026-10-25 00:01 IST — E2 repeats nearing completion

- Read-only `squeue`/`sacct` confirms E2 jobs 363768 and 363769 are RUNNING on `cn42-l40` and `cn45-l40`, elapsed 01:04:33 and 01:04:32, with no failure state. Their logs have progressed to epoch 43 (seed2) and epoch 44 training (seed3); 42 and 43 epochs respectively have validation metrics.
- Interim seed2 best remains epoch 27 (Se 38.06%, Sp 83.41%, Score 60.74%); latest validated epoch 43 is 46.47/69.73/58.10. Interim seed3 best remains epoch 38 (42.31/75.87/59.09); latest validated epoch 43 is 45.79/66.75/56.27. Values may still change through epoch 50.
- At the observed ~73 s/epoch, estimated time to finish is 7–9 minutes. P1 diagnostic job 363821 and P2 profile job 363823 remain dependency-held until both E2 jobs terminate. No job was changed or stopped.

## 2026-10-25 00:09 IST — E2 seed 3 complete

- E2 seed3 job **363769** completed 50/50 epochs on `cn45-l40`, exit 0, elapsed **01:12:23**. Best test-selected epoch 38: Se=42.31%, Sp=75.87%, Score=59.09%. Last epoch 50: Se=45.71%, Sp=67.70%, Score=56.71%.
- E2 seed2 job **363768** is still running on `cn42-l40`, elapsed 01:12:06 at the check; epoch 49 is validated and epoch 50 is in progress. Its interim best remains epoch 27: Se=38.06%, Sp=83.41%, Score=60.74%; epoch49 is 45.71/70.99/58.35. No final status is assigned until job termination.
- Refreshed `artifacts/results/results.csv` and copies in `Code/Code/AFT/results/` and `IE643_repo/results/`. It currently has 10 run rows; E2 seed3 is complete and E2 seed2 is explicitly partial (`epochs_logged=49`). Updated and synced `summary.md` with seed3's final metrics and seed2's in-progress status. Full three-seed E2 aggregation awaits seed2 completion.

## 2026-10-25 00:12 IST — E2 repeats complete; three-seed baseline aggregates

- E2 seed2 job **363768** completed 50/50 epochs on `cn42-l40`, exit 0, elapsed **01:14:19**. Best epoch27: Se=38.06%, Sp=83.41%, Score=60.74%; last epoch50: Se=45.54%, Sp=70.93%, Score=58.24%.
- E2 seed3 job **363769** completed 50/50 epochs on `cn45-l40`, exit 0, elapsed **01:12:23**. Best epoch38: Se=42.31%, Sp=75.87%, Score=59.09%; last epoch50: Se=45.71%, Sp=67.70%, Score=56.71%.
- E2 over seeds1–3, mean ± sample SD: best Se=40.30±2.13%, Sp=79.57±3.77%, Score=59.93±0.82%; last Se=45.57±0.13%, Sp=69.22±1.62%, Score=57.39±0.78%. E1's corresponding best Score mean is 60.53±0.62%; E2 is descriptively 0.60 percentage points lower. Best epochs are selected on the test set.
- Refreshed `results.csv` and `summary.md` under `artifacts/results/`, `Code/Code/AFT/results/`, and `IE643_repo/results/`. Exact E1/E2 repeat runtimes are recorded in `results.csv`.
- Confusion-matrix audit covered **484 nonempty epoch rows** in ten run metrics files (including 34 epochs from the incomplete E5 retry1). Each matrix has expected test support [1579,649,385,143], total 2756; stored Se, Sp, Score, accuracy, class recalls, and binary sensitivity all agree with recomputation within 0.00002 percentage points.
- P1 job **363821** (diagnostics) started on `cn45-l40`, P2 job **363823** (DiffWave profile) on `cn41-l40`, and E7a weighted-CE seed1 job **363852** on `cn42-l40`. All were running at the 00:12 check; three project GPU jobs are active. E7 was validated with `sbatch --test-only` (estimate 00:32:52, cn41-l40) and submitted with an `afterany:363768` dependency. No running job was stopped or changed.

## 2026-10-25 00:18 IST — P1 diagnostics complete; profile and E7 retries

- P1 diagnostic job **363821** completed on L40 in **00:05:01**, exit 0. Gain matching raised E1 probe intended-class accuracy from 6.01% to 10.62% for 499 wheeze samples and from 2.35% to 4.71% for 637 both samples. These are still below E1 real-test recall (best/last metrics 28.57/37.92% for wheeze, 16.78/19.58% for both); RMS matching alone is insufficient.
- On 50 official-train mel conditions per class, fast/full sampling intended-class accuracy was wheeze 6%/2%, both 2%/0%; full sampling did not improve results. Fast sampling was chosen based only on training-source diagnostics.
- Held-out diagnostic vocoding on 100 test cycles/class: prediction agreement was 55% (wheeze) and 40% (both), while E1 probe correctness fell from 28%/17% on original audio to 1%/0% on generated audio. Mean log-mel L1 was 0.11079±0.05795 and 0.11008±0.04668. All test-derived generated audio remains under `artifacts/diag/v1/`, excluded from training. Full factual report is `artifacts/diag/v1/diag_report.md`.
- First DiffWave profile job **363823** failed before training steps in 00:05:08: `ConditionalDataset` has no `.dataset` attribute. It spent 192.01 seconds loading 12,426 discovered examples into RAM. Fixed the DataLoader to receive the `ConditionalDataset` instance directly; the retry output directory is `artifacts/diag/diffwave_profile/retry1/`. Profile retry **363860** was submitted on L40 and is pending resources; test-only estimated start 01:17 IST.
- First E7a seed1 job **363852** failed before any epoch/checkpoint in 00:04:00: `json.dump` encountered a `SpecAugment` object created during loader setup. Updated metadata serialization to omit this runtime-only object. Kept the failed run directory/job log intact; retry tag is distinct (`..._official_retry1`). E7a retry **363858** started on `cn45-l40` at 00:17 IST, with the same planned weighted-CE config.
- At 00:18 IST, E7 retry is running and the DiffWave profile retry is pending. `du -sh artifacts` reports **14G**. No jobs were cancelled or modified while running.
- Follow-up data-loader audit: `find` reports **4,142 unique WAV files**, while the old `glob('**/**/*.wav')` yields **12,426 entries** (each path repeated three times). The original 20k-step run used this same loader pattern; it sampled the same official-train clips with triple index frequency, without changing the empirical data distribution. Updated `ConditionalDataset` to use `**/*.wav` and unique sorted paths for future jobs. The active profile process had already loaded the old list before this patch and was left untouched; its 192.01-second load time is not representative of the deduplicated loader startup.

## 2026-10-25 00:43 IST — DiffWave profile selected; continuation queued

- Successful profile job **363860** completed on L40 in 00:10:02, exit 0. It loaded the in-memory dataset (using the pre-patch threefold glob list) and ran three 200-step comparisons. Throughput results: baseline workers8/batch16/FP32/nonpersistent 1.476 total and 3.042 steady steps/s; workers2/batch16/FP32/persistent 3.031 total and 3.036 steady steps/s; workers2/batch32/BF16/persistent 2.206 total and 2.589 steady steps/s. All losses finite; last-20 mean losses 0.06988, 0.06964, and 0.07049, respectively; peak CUDA memory 8.266–8.277 GiB. Sampled GPU utilization across this job averaged 34.1%, max 100%. Selected workers2/batch16/FP32/persistent (highest total throughput).
- Added `artifacts/diag/diffwave_profile/retry1/profile_report.md` and retained `profile_results.csv` and sampled utilization CSV. The dataset loader now removes duplicate path entries (old glob yielded each of 4,142 unique training WAVs three times). The profile process had already loaded those repeated paths; the model-training continuation will load the patched unique list.
- Submitted DiffWave continuation job **363888** to preferred `l40/l40`: 4-hour segment; resume from step-20k checkpoint; batch16, workers2, FP32, persistent workers; target 200,000 steps; save every 5,000 and signal for checkpoint 120 seconds before walltime. At 00:43 it was pending resources (test-only estimate 01:36). Original 20k run speed ~0.95 steps/s; selected profile steady speed ~3.04 steps/s. Continue only until 12:00 IST per plan, subject to queue/start time.
- E7a weighted CE seed1 retry job **363858** is running on `cn45-l40`; at 00:43 elapsed 00:26:14, epoch30 in progress. E7 seeds2/3 jobs **363866/363867** are pending afterok on successful seed1. Profile/P1 jobs are complete. Artifact storage: 15G at the latest shell check.
- Follow-up scheduler-signal review: changed the segment script for future submissions from `B:TERM@120` (batch-shell signal only) to `TERM@120`, so Python's installed handler can save on pre-timeout warning. Job 363888 was already submitted with the former directive; it remains pending and was not changed or stopped. It still saves periodically every 5,000 steps, so the current queued segment may lose at most the work since its last periodic save if time-limited. Subsequent segments use the corrected directive.
- 00:48 IST follow-up: E7a seed1 job 363858 has reached 35/50 validated epochs in 00:30:38; current best remains epoch 27, Se=41.72%, Sp=78.53%, Score=60.12%. DiffWave segment 363888 remains pending; `squeue --start` currently estimates a 01:17:11 start and 05:17 end for its requested four-hour segment. E7 seed2/3 jobs remain afterok dependencies. No jobs changed.
- 00:49 IST status: E7a weighted CE seed1 job 363858 remains RUNNING on `cn45-l40`, elapsed 00:32:01, 37 epochs validated. Interim best epoch27: 41.72/78.53/60.12; latest epoch37: 49.28/62.57/55.92. DiffWave job 363888 start estimate remains 01:17:11 and approximate segment end 05:17. E7 seeds2/3 remain dependency-pending. No jobs changed.
- 00:52 IST status: DiffWave segment job **363888** actually started on `cn45-l40` at 00:51:40 despite the prior later `squeue --start` estimate; 4-hour scheduled wall end is about 04:51:40. It is resuming from step 20k using batch16/workers2/FP32. E7a seed1 job **363858** is at 41/50 validated epochs after 00:35:18, best epoch27 remains Score60.12; latest epoch41 Score57.71. E7 seeds2/3 remain afterok pending. All jobs are left untouched.

- 00:54 IST verified status/ETA snapshot: E7a weighted-CE seed1 retry 363858 is at 44/50 validated epochs after 00:37:18; best epoch27 remains Se=41.72%, Sp=78.53%, Score=60.12%, latest epoch44 is 45.96/70.30/58.13. At the observed pace, estimate roughly 4–5 minutes plus finalization, around 00:59–01:02. E7 seeds2/3 (363866/363867) are still dependency-gated; scheduler supplies no start estimate, so a reasonable conditional estimate is roughly 35–45 minutes each after they start, with completion around 01:45–01:55 only if they launch promptly. DiffWave segment 363888 started 00:51:40 on cn45-l40 with a four-hour wall limit (scheduled end ~04:51:40); at 00:54:29 it had elapsed 00:02:49 but no training progress log yet. The 00:55 live scheduler query failed because this session could not connect to the SLURM controller; the estimates above are based on the latest successful observation, not a fresh state. No jobs were altered.

- 01:04 IST update: live SLURM access restored. Job 363866 (E7 seed2) is RUNNING on cn45-l40 (elapsed 00:06:06 at 01:03); job 363867 (E7 seed3) is PENDING due to resources. DiffWave 363888 is RUNNING on cn45-l40 (elapsed 00:13:34 at 01:04), four-hour segment end ~04:51:40. SLURM reports L40 `MaxSubmitJobs=5`, `MaxJobs=4`; a four-task array failed `sbatch --test-only` with `QOSMaxSubmitJobPerUserLimit`, so no array was submitted. To stay under the submit limit and four-GPU cap, prepared and submitted job 363897, which depends `afterany` on both E7 seeds and runs E3/E4/E5 v1 repeats for seeds 2 and 3 sequentially on one L40 GPU (four workers), in order E3 seeds2/3, E4 seeds2/3, E5 seeds2/3. It has a 9-hour walltime; at 01:04 it is PENDING (Dependency). No running jobs were stopped or changed. Artifacts tree size is 16G.

- 03:49 IST live status: E7 seeds2/3 jobs 363866/363867 completed successfully in 00:39:26/00:45:56. E3 v1 seed2/3 in sequential job 363897 completed successfully in 00:48:22/00:47:53; best Scores 59.29/60.04, last Scores 55.53/56.34. E4 seed2 is in progress at epoch10/50; interim best epoch7 48.94/74.79/61.87. E4 seed3 and E5 seeds2/3 remain in the same 9-hour job, estimated total completion ~08:30–09:30. Refreshed results.csv to include 16 run summaries and synced results/summary to Code/Code/AFT and IE643_repo. E3 3-seed mean±sample SD: best Se/Sp/Score=44.12±5.51/75.32±6.02/59.72±0.38%; last=47.95±1.00/64.94±2.90/56.44±0.96%. E7 3-seed mean±sample SD: best=43.30±1.40/76.48±2.81/59.89±0.99%; last=45.31±1.79/70.30±2.59/57.81±0.53%. DiffWave 363888 is RUNNING (2:57 at 03:49); step30k checkpoint was saved 03:43:59, corresponding to ~0.97 steps/s from 20k. At this rate segment wall end 04:51:40 projects roughly 34k, materially below the short profile's 3.04 steps/s. Per plan, do not queue more DiffWave segments until discussing options with user; current job remains untouched. `artifacts/` is 21G.

## 2026-10-25 05:04 IST — Plan v3 WP-A/WP0/WP1 and Petmezas launch

- Refreshed the dual-metric audit from the current run snapshot: **17 run summaries, 819 epoch confusion matrices, four reference rows**. The latest E4 seed3 partial epochs are included; subsequent changes in the live metrics file will be picked up in the next refresh. `results.csv` and `metric_audit.md` were synced to `Code/Code/AFT/results/` and `IE643_repo/results/`; `handoff/METRIC_NOTE.md` is synced to the code-only handoff copy.
- WP-A reference values (official / Petmezas-style micro, percent): Petmezas published official = micro 47.37/82.46/64.92/73.69; published 10-fold = 52.78/84.26/68.52/76.39; Figure-5 pooled matrix re-expressed = official 22.36/73.81/48.08, micro 49.52/83.17/66.35/74.76; always-normal = official 0/100/50, micro 57.29/85.76/71.53/78.65.
- WP0 read-only scheduler snapshot at 05:04 IST: job **363897** is running on cn45-l40, elapsed 03:04:34. Sequential steps 363897.0/.1/.2 (E3 seeds2/3 and E4 seed2) completed in 00:48:22/00:47:53/00:49:27. Step .3 (E4 seed3) is running, elapsed 00:38:49; E5 seeds2/3 follow and remain intact per user decision. Current conditional completion estimate is around 08:00–09:00 IST if prior per-run pace holds.
- DiffWave job **363888** reached step **33,868**. Its Python step ended successfully, but the batch was cancelled at its 4-hour wall limit (sacct batch exit 0:15). The final checkpoint `artifacts/diffwave/runs/condition_icbhi_bs_16_sr_16000_len_4_label_0_seed_1_cc_False_step_50_tag_v2_b16_w2_fp32/weights-33868.pt` was written at 04:49. The GPU allocation is already released; no successor segment was queued. Limited WP7 QC is pending a free GPU slot and will not continue model training.
- WP1 validation split: deterministic 5,000-candidate patient-grouped search (seed 0), with patients 156/218 excluded from validation. The result has 504 validation cycles (12.168% of 4,142), 12 validation patients, and class counts normal/crackle/wheeze/both = 259/149/56/40. Train-minus-validation has 3,638 cycles and 67 patients; patient sets are disjoint, filenames do not overlap test, and patients 156/218 stay in the training side. Files are under `artifacts/data/official_icbhi/splits/`.
- Added reusable `Classification/prepare_validation_split.py`; AST dataset/main now have holdout-manifest filtering, an `eval_split` marker, a `--save_probs` option, and post-training official-test evaluation for holdout-mode runs. Default legacy behavior is preserved. Added the plan-informed CNN-LSTM/focal implementation and L40 array script. Syntax compilation and an input/output shape smoke check passed; architecture assumptions are written to each config. Paper-silent choices: 256-point Hann STFT, hop 144, log-power per-image min-max replicated to RGB, repeat-sample padding, 3x3 same convolutions, Adam, batch32, up to100 epochs, validation-loss patience10.
- Submitted the three-seed Petmezas array as job **363959** after `sbatch --test-only` estimated L40 start at 05:03:15. It started at about 05:03 on cn42-l40; seeds1–3 are running. Job 363897 plus the three array tasks equals the 4-GPU cap. No other GPU work will be started until capacity frees.
- Current `main.py` option audit: existing cosine LR, warmup, `ma_update`/`ma_beta`, CE/Patch-Mix losses, adversarial adaptation, `imbalance`, and `meta_mode`; added `val_mode`, `val_manifest`, and `save_probs`. Remaining v3 screening controls (including proper layer-wise LR decay, stochastic depth, per-cycle RMS, location/device features and scalogram views) still need implementation before those arms can be run.
- Storage: `artifacts/` = 22 GiB at 05:04 IST. No job was cancelled, held, modified, or reordered.

## 2026-10-25 05:19 IST — Plan v3.2 lean protocol supersedes validation workflow

- Read the updated `plan_v3.md`. It now explicitly requires full official training data (4,142 cycles), no validation split or screening, predeclared candidates/ensembles, and last-epoch metrics as the headline. The earlier v3.1 validation manifests remain archived and unused for all new experiments.
- Marked Petmezas array 363959 as a **pre-change pilot**. All three tasks completed under v3.1 using 3,638 training cycles and validation-CE selection. Its existing outputs and values remain in `progress.md`, but are not claimed as the v3.2 WP2 reproduction.
- Precision job 363963 failed before its first training epoch: `UnboundLocalError` for `lamb` in the ordinary CE path. It never reached BF16. Patched `main.py` to initialize the neutral CE path's per-batch coefficient and to save final official-test probabilities/confusion for non-holdout runs using `--save_probs`.
- Registered `artifacts/results/PREREG.md` at 05:15:59 IST before any new v3.2 test results. It fixes Petmezas fixed-40-epoch/full-train runs, the C0–C4 full-train AST list, FP32 as the initial common precision, optional complete SC1/SC2, equal-weight ensemble membership, and the disclosed demo-selection rule. The registration notes that it is prospective and does not backdate the earlier pilot.
- Added `Classification/petmezas_lean.py` and `hpc/slurm/train_petmezas_lean.slurm`. It trains all 4,142 cycles for 40 fixed epochs, does not construct a validation loader or early-stop, logs both metric families on official test each epoch for reporting, and stores last-epoch output plus a clearly secondary test-selected best-Score output. Paper-silent preprocessing/architecture choices are recorded in each output config.
- Added `hpc/slurm/profile_ast_precision_lean.slurm` and `hpc/slurm/train_ast_lean_serial.slurm`; marked the old validation screen script obsolete. The precision comparison is restricted to throughput and finite/stable training loss; test metrics do not choose precision. The AST script runs seed 1 C0–C4, then seed 2, then seed 3, with full train and FP32.
- L40 checks: `scontrol show partition l40` confirmed `AllowQos=l40` and `MaxTime=2-00:00:00`. Precision `sbatch --test-only` estimate was 05:18 on cn42-l40. Full-train Petmezas array estimate was 05:18:22, dependent on precision completion.
- Submitted precision job **363966** on L40 and Petmezas 3-task array **363968_[1-3%3]**, dependent on 363966. At 05:19, squeue showed 363897 RUNNING on cn45-l40 (elapsed 03:19:17), 363966 RUNNING on cn42-l40 (elapsed 00:43), and 363968 array PENDING (Dependency). Two active GPU jobs; four-GPU cap preserved. Job 363897 remains unmodified; its 05:13:12 log shows E4 seed3 completed and E5 seed2 started. E5 seed3 remains to run.
- `artifacts/` measured **24 GiB**. About **18h40m** remained before 25 Oct 23:59 IST. No v3.2 AST candidate job has been submitted yet; queue it after dependencies and a test-only estimate.

## 2026-10-25 05:22 IST — v3.2 jobs progressing; reporting pipeline aligned

- At 05:21, job 363966 was RUNNING on cn42-l40 at elapsed 00:03:39, with FP32 epoch 2/3 complete. Its first two full-train epochs took 75.42 and 73.11 seconds. Logged official-test values are reporting-only and will not be used for the precision decision. BF16 has not started yet.
- Job 363968 array tasks 1–3 remain pending on `afterok:363966`; job 363897 remains RUNNING on cn45-l40. The 05:22 log tail for 363897 showed E5 seed2 at epoch 5/50. Two project GPU allocations are active; no job was interrupted or changed.
- `artifacts/` is 25 GiB. About 18h38m remains before the 23:59 IST cutoff.
- Updated the full-train Petmezas logger to emit the same confusion-matrix/metric columns as AST `metrics.csv`, including both metric families, so existing result collection can aggregate it. Updated the result collector to skip legacy nonstandard pilot logs and mark any standard-schema run with a validation-cycle config as pre-v3.2. Updated metric-audit report language to state the v3.2 full-train/no-validation protocol.
- Synced the current scripts, `main.py`, `progress.md`, experiment log, PREREG, and handoff status into `IE643_repo/`. No v3.2 AST candidate run is queued yet.

## 2026-10-25 05:24 IST — Handoff refresh utility and queue-cap check

- Created `make_handoff.sh` for run collection, confusion-matrix metric recomputation, the audit/results refresh, and synchronization of code, results, progress, experiment log, PREREG, and handoff status to `IE643_repo/`. It is not executed during the current partial precision run; it will be run after that job completes.
- Added `handoff/FACTS.md` with split counts, class supports, legacy multi-seed scores, metric references, Petmezas v3.1 pilot status, DiffWave checkpoint, v3.2 config and active job state.
- At 05:24, squeue showed 363897 RUNNING (cn45-l40, elapsed 03:24:27), 363966 RUNNING (cn42-l40, elapsed 00:05:53; FP32 completed and BF16 began), and three tasks of array 363968 PENDING on dependency. `artifacts/` measured 25 GiB at 05:21.
- Attempted a `sbatch --test-only` estimate for the seven-hour serial C0–C4 job, with dependencies on the legacy repeat, precision check and Petmezas array. SLURM returned `QOSMaxSubmitJobPerUserLimit`; no candidate job was submitted. The queue currently has two active allocations and three dependency-pending array tasks, within the four-running-GPU cap but at the submit limit. Retry after the array completes.
- Synced `make_handoff.sh`, `FACTS.md`, the revised `petmezas_lean.py`, metric collector/audit helper, v3.2 SLURM scripts, progress and status into the code-only copy. No dataset, audio or checkpoints were copied.

## 2026-10-25 05:27 IST — Precision decision and candidate launch queued

- Precision job **363966** completed on cn42-l40 in **00:08:22**, exit 0. It trained three full-train epochs in each mode. FP32 epoch times 75.42/73.11/73.22 s; BF16 48.03/42.01/42.02 s. Mean speedup is **1.68×**. Training losses: FP32 0.949/0.758/0.681; BF16 0.949/0.757/0.680. Train accuracy differs by at most 0.34 points. The precision decision uses only training time/loss/accuracy; official-test logs were excluded.
- Following the conditional precision rule in v3.2, changed the entire pre-registered C0–C4 set to BF16, including matched control C0. `PREREG.md` remains unchanged; the transparent decision is recorded in `artifacts/results/PREREG_DECISIONS.md` before any C0–C4 test result. Updated the serial candidate script to use BF16 for all 15 runs.
- Full-train Petmezas array **363968_[1-3%3]** started on cn42-l40 after the precision job. At 05:27:22 its tasks had elapsed 00:00:42–00:00:43. Job 363897 remains running on cn45-l40; its E5 seed2 repeat continues.
- The C0–C4 seven-hour serial job **363980** passed `sbatch --test-only` and was submitted with dependencies afterany:363897, afterok:363966, afterany:363968. `scontrol` confirms it waits for 363897 and all tasks in 363968; four project GPU allocations are active (363897 and three array tasks), and the 4-GPU cap is maintained.
- `artifacts/` measured **27 GiB** at 05:27. About **18h32m** remained before the inherited 23:59 IST deadline.

## 2026-10-25 05:29 IST — v3.2 Petmezas full-train reproduction complete

- Petmezas array **363968** completed all three 40-epoch seeds successfully on L40. All runs used 4,142 official-train cycles; each task took 00:02:30–00:02:32. There was no validation split or early stopping.
- Headline last-epoch means ± sample SD, official Se/Sp/Score: **22.49±4.05 / 64.51±5.23 / 43.50±0.66%**. Petmezas-style micro Se/Sp/Score/Acc: **46.57±1.30 / 82.19±0.43 / 64.38±0.87 / 73.28±0.65%**.
- Per-seed last official Se/Sp/Score: seed1 26.34/59.15/42.74; seed2 18.27/69.60/43.93; seed3 22.85/64.79/43.82. Test-selected best epochs (secondary only) were 2/2/1 and produced official mean 0.17±0.22 / 99.85±0.16 / 50.01±0.06. These secondary predictions were mostly all-normal. Last-epoch official Score is 6.50 points below always-normal Score 50.00; last micro Score 64.38 is below always-normal micro Score 71.53.
- This is an approximate paper reproduction; all paper-silent assumptions are in `artifacts/runs/petmezas_lean_seed{1,2,3}/config.json`. The earlier v3.1 pilot remains separately documented and is not conflated with this full-train run.
- `make_handoff.sh` was run after completion. It collected 23 standard-schema run summaries and recomputed 970 epoch confusion matrices, appending four reference rows. It skipped the three nonstandard v3.1 pilot CSVs with an explicit notice. Updated results, audit and handoff files were synced to `IE643_repo/`.
- At 05:29, squeue showed only job **363897** active (E5 seed2/3 sequence); candidate job **363980** pending on 363897. Petmezas array tasks 363968_1/_2/_3 completed in 00:02:30–00:02:32. `artifacts/` measured 27 GiB at 05:27.

## 2026-10-25 05:32 IST — Always-green handoff refresh after WP2

- Ran `make_handoff.sh` after full-train Petmezas completion. It collected 23 run summaries and audited 970 epoch confusion matrices; the four published/reference rows were retained. The three v3.1 nonstandard Petmezas pilot CSVs were skipped and remain documented separately. Results, metric audit, progress, log, facts/status, registration files and code were synced into `IE643_repo/`.
- At 05:32, read-only squeue showed 363897 RUNNING on cn45-l40 (elapsed 03:32:37) and 363980 PENDING on dependency 363897. Log tail: E5 seed2 epoch 11/50; seed3 remains next. Precision and Petmezas jobs are complete; one project GPU is active. `artifacts/` is 27 GiB. About 18h27m remain to deadline.

## 2026-10-25 05:34 IST — Results refresh and dependency status

- Ran `make_handoff.sh` again after the project log/status updates. It now reports 23 run summaries, **971** epoch confusion matrices, and 4 reference rows. The extra logged epoch is a new completed E5 seed2 epoch from job 363897. It synced the updated handoff and code-only repo.
- At 05:34, `squeue` showed job 363897 RUNNING on cn45-l40 (elapsed 03:34:53), job 363980 PENDING on its completion, and no other active project jobs. `artifacts/` measured 27 GiB. About 18h25m remain before the 23:59 IST deadline.

## 2026-10-25 05:43 IST — v3.2 CWT parity review and WP4 dependency chain

- Re-read current plan v3.2, v2 inherited scheduling instructions, and PRAJNA HPC rules. Confirmed the pre-registration specifies FP32 for SC1/SC2 even though WP3 candidates use BF16 after the measured WP1 diagnostic. Corrected the WP4 script to use FP32 and fixed the WP3 script's misleading startup precision message. `PREREG.md` remains unchanged.
- Reviewed the CWT cache and inference paths against the AST dataset preprocessing. Updated `predict.py` so mel input is computed on the original cycle and resized to the same (498,128) training grid; CWT input uses repeat-pad/crop to five seconds without a fade, matching the cache convention. Added a five-clip cache/on-the-fly maximum absolute error check (<0.01 dB) to the compute-node cache job. SC2 parity/backward check remains in the GPU smoke script.
- At 05:43, job 363897 was RUNNING on cn45-l40, elapsed 03:43:33; E5 seed2 was around epoch 18/50 and seed3 was queued within the same sequential job. No running job was modified.
- Job 363980 remains pending after 363897 and runs all C0–C4, seeds 1–3, BF16, serially (7-hour limit). Submitted job 363983 for SC2 smoke + full CWT cache (1-hour limit) with `afterok:363980`; submitted job 363985 for the complete SC1/SC2 FP32 block (3-hour limit) with `afterok:363983`. These are sequential dependencies, preserve the one-GPU-at-a-time workload and stay within the four-job submission cap. Job 363985 exits without starting WP4 if less than 3h15m remain before the 23:59 IST cutoff, to avoid a partial WP4 block.
- Test-only cache estimate was 06:51; the test-only SC estimate was 07:01 under a hypothetical cache dependency. These estimates do not include the actual upstream run duration and are not start-time promises.
- `artifacts/` was 27 GiB. Latest results refresh before this entry: 23 run summaries, 971 epoch rows, and 4 reference rows. Refresh the audit/handoff after the current update.

## 2026-10-25 05:44 IST — Dual-metric audit and handoff refresh

- Ran `make_handoff.sh` after the CWT implementation and queue update. It produced 23 run summaries, 978 epoch confusion matrices, and four reference rows. It skipped the three legacy nonstandard-schema Petmezas pilot files as intended. Results/audit and the code-only repository/handoff copies were refreshed at 05:44:40 IST.
- At 05:43 scheduler snapshot: 363897 RUNNING; 363980, 363983 and 363985 dependency-pending. `artifacts/` measured 27 GiB. Current latest-result headline remains the completed full-train Petmezas result; AST candidates have not started yet.
- Follow-up note: because SLURM captured the AST batch script when 363980 was submitted, its pending copy will still print an inaccurate `FP32` startup label. Its per-candidate command passes `--amp bf16`; the actual config records the selected precision. The workspace script is corrected for any later submission, and the pending job was left untouched.

## 2026-10-25 07:48 IST — Status check and rolling audit refresh

- At 07:47, job 363897 was RUNNING on cn45-l40, elapsed 05:48:00. E4 seed3 and E5 seed2 completed; E5 seed3 was around epoch 38/50. Recent epoch time is about 93 s; rough remaining work is 15–20 minutes if uninterrupted. E5 seed2 test-selected best Score 61.46 at epoch 7 (secondary only), last Score 56.03; E5 seed3 epoch-38 Score is 56.62 and is not final.
- `scontrol show job` showed 363980 pending only on dependency 363897, with start time unknown. Jobs 363983 and 363985 remain chained on successful completion. No jobs were modified.
- Ran `make_handoff.sh`; it now reports 24 run summaries, 1,048 epoch rows, and four reference rows. `artifacts/` is 28 GiB. 16h11m remain to the 23:59 IST cutoff.

## 2026-10-25 12:38 IST — WP3 C4 failure and recovery

- Job 363980 FAILED with exit 1 after 03:51:50. It completed C0–C3 for seeds 1–3 (12 runs) but all C4 seeds failed before epoch 1 with `UnboundLocalError: proj1`. Cause: parser default `target_type=project1_project2block` is unsupported in the `patchmix_cl` branch; `proj1` is assigned only for four listed target types. Existing E2's config uses `project_block`.
- Updated the workspace AST serial script to pass `--target_type project_block` on C4. Created and syntax-checked `hpc/slurm/train_wp3_c4_recovery.slurm` to rerun all three C4 seeds with full train, 20 epochs, BF16, EMA 0.998, and explicit supported target type. Job 364154 was submitted after L40 test-only estimate of 16:25; actual `scontrol` start is unknown and state is PENDING (Resources).
- Repaired cache job 363983 dependency from the failed 363980 to `afterok:364154`. SC training 363985 still depends on 363983. The corrected chain is C4 recovery → cache/SC2 smoke → full SC1/SC2. No CWT candidate has run.
- Latest `make_handoff.sh` refresh: 36 run summaries, 1,300 epoch rows, and 4 references. Results contain C0–C3 for all three seeds; failed C4 attempts generated no metrics rows. `artifacts/` is 44 GiB. 11h21m remain before cutoff; if the recovery test-only estimate holds, the entire chain could finish around 22:00, but later starts may trigger the WP4 all-or-nothing cutoff guard.

## 2026-10-25 12:40 IST — C4 recovery allocation started

- Job 364154 started on cn45-l40 at 12:40:17. Its Python task began seed1 at 12:40:24 using the corrected supported `target_type=project_block`. The job runs seeds1–3 sequentially, BF16, 20 epochs each. Rough completion estimate 13:35–14:00 based on prior candidate timing.
- Cache job 363983 is intentionally waiting for `afterok:364154`; WP4 job 363985 waits for 363983. One project GPU is active. `artifacts/` remains 44 GiB.

## 2026-10-25 13:08 IST — WP7 DiffWave v2 diagnostic submitted

- The latest DiffWave checkpoint is `weights-33868.pt` (step 33,868) in the v2 run directory. Added a diagnostic-only pool generator and v2 QC runner. These generate 120 clips per class from official-training conditioning mels; compare raw/source-RMS-matched probe accuracy and fast/full sampler accuracy on training-source conditions; compute held-out prediction agreement and mel reconstruction errors; save two mel grids; and write a report comparing v1 at 20,000 steps with v2 at 33,868 steps.
- Output audio, including audio vocoded from official-test cycles, is restricted to `artifacts/diag/`; no training manifests are modified. No DiffWave successor training is included.
- Job 364177 was submitted to A40/QOS A40 after read-only verification of partition policy and live availability. `sbatch --test-only` estimated start 13:07:44 on cn33-a40; actual job started at 13:08:14. Request: one GPU, four CPUs, 32 GiB RAM, two hours.
- At 13:09:24, job 364177 was RUNNING and stdout/stderr had not flushed yet. C4 recovery 364154 was also RUNNING on cn45-l40; 363983 and 363985 were PENDING on dependencies. Four project GPU jobs are submitted/running, within the cap. `artifacts/` measured 45 GiB.

## 2026-10-25 13:18 IST — WP7 DiffWave v2 diagnostic completed

- Job 364177 completed on cn33-a40 in 00:10:17 (exit 0). It generated 120 clips each for wheeze and both using official-training mels and the DiffWave step-33,868 checkpoint; diagnostic WAVs and test-derived audio are isolated under `artifacts/diag/` and absent from training manifests.
- Pool probe raw/RMS-matched accuracy: wheeze 21.67%/22.50%; both 2.50%/5.00%. Sampler probe on 50 train conditions/class: fast/full was wheeze 16%/14%, both 4%/2%; fast-six-step selected based only on training conditions.
- Held-out prediction agreement (100/class) was 57% for wheeze and 40% for both, versus v1 55%/40%. E1 original/vocoder correctness: 29%/5% wheeze and 17%/5% both. Mean mel L1 (SD): 0.1177 (0.0490) and 0.1180 (0.0412). Both agreements remain below 80%; plan decision is no further DiffWave training segment.
- Report `artifacts/diag/diag_report.md`; two mel grids and CSVs under `artifacts/diag/v2/`. `artifacts/` was 45 GiB at 13:19.
- C4 job 364154 remained active on L40: seed1 completed 13:17:59 and seed2 began. Its 90-minute limit ends at 14:10:17; seed3 may need a follow-up if the current allocation expires. Cache 363983 and SC1/SC2 363985 remain dependent and pending.

## 2026-10-25 14:04 IST — WP6 paired bootstrap for ready ensembles

- **Job:** 364220, `ie643_wp6_bootstrap`, CPU-only allocation on `a40/a40`, 2 CPUs, no GPU; completed in 00:00:02 with exit 0. C4 job 364154 was left running and unchanged.
- **Analysis:** 1,000 paired bootstrap resamples over the 2,756 official test cycles; fixed RNG seed 20261025. Inputs are the complete pre-registered Ens-C0, Ens-C1, Ens-MP and the full-train three-seed Petmezas runs. AST probabilities were aligned to `test_manifest.csv` order. Results are uncertainty summaries only; no model, test threshold, or ensemble weight was selected.
- **Output:** `artifacts/results/wp6_partial/wp6_preliminary_paired_bootstrap.csv`, 24 intervals: official and Petmezas-style micro Se/Sp/Score differences for four predeclared comparisons.
- **Selected official intervals (percentage points):** Ens-C1 − Ens-C0: Se +0.51 [−0.33,+1.30], Sp +0.70 [+0.06,+1.29], Score +0.60 [+0.07,+1.06]. Ens-MP − Ens-C0: Se −3.74 [−5.45,−2.21], Sp +3.42 [+1.90,+5.00], Score −0.16 [−1.30,+0.96]. All rows, including intervals crossing zero, are retained.
- **Queue at 14:02:** 364154 RUNNING on cn45-l40; 364216 waits afterany:364154; 363983 waits afterok:364216; 363985 waits afterok:363983. The C4 log at 14:04 showed seed2 epoch 16/20 with a 14:10:17 allocation end. The continuation skips seeds with a completed retry1 `last.pth`. `artifacts/` was 46 GiB. About 9h55m remained to the 23:59 IST cutoff.
# 25 October 2026 — final v3.2 WP4/WP6 audit

- WP4 recovery jobs 364828 (SC1 seed3) and 364829 (SC2 seed3) completed the registered 20-epoch FP32 runs from scratch. The timed-out SC1 attempt is excluded as incomplete. WP4 has 3/3 seeds for both views.
- WP6 CPU job 364880 evaluated all registered equal-weight ensembles and 1,000 paired test-cycle resamples (72 intervals). The official Score was 59.57 (Ens-C0), 60.18 (Ens-C1), 59.93 (Ens-C4), 59.36 (Ens-MS), 59.41 (Ens-MP), 60.39 (Ens-AST), and 59.82 (Ens-ALL). TTA was not run under the time-short priority order.
- The three C3 `probs_test_last.npy` files are byte-identical to the corresponding C1 files. Inspection found that `util.misc.adjust_learning_rate` overwrites every optimizer group's LR each epoch with the same scalar, undoing C3's intended LLRD/head multipliers. Report C3 as a failed implementation of its registered recipe; preserve its output and disclose that registered Ens-AST includes duplicated C1 probability members.
- Highest single candidate by three-seed mean last-epoch official Score is C4 at 58.74 ± 0.86. Ens-AST is 60.39, exceeding it by 1.65, so the registered demo selection rule selects Ens-AST. The selection effect and 15-member/approximately 5-GB footprint are disclosed in `handoff/FACTS.md` and `artifacts/results/summary.md`.
- GPU job 364883 is running `predict_ensemble` over four official test examples (one per class), streaming the 15 AST checkpoints one at a time. Log: `hpc/logs/ie643_ens_infer.364883.out`.

# 15 November 2026 04:10 IST — Novelty phase (SpecDiff): selection locked before Phase B

- **Selection data:** validation subset only (504 cycles, 12 patients held out of official train). Generator `G_sel` and the probe were trained on train-minus-val. Last-epoch official Score, mean ± sample std over seeds.
- **Baseline:** M0 real only, 7 seeds: Se 64.55 / Sp 75.46 / Score 70.00 ± 1.42.
- **Selected primary arm:** `M3f` = class balancing with synthetic log-mels (every class topped up to the normal-class count), pool from the 50k-step EMA checkpoint, guidance 3.5, empirical device policy, kept only if the real-only probe gives the intended class probability ≥ 0.6. 6 seeds: Se 66.46 / Sp 76.77 / Score 71.62 ± 1.13 (+1.62 over M0; Welch standard error about 0.71).
- **Pre-declared secondary arms (ablations, all to be reported):** `M3` (same without the probe filter; validation 70.83 ± 1.07, 3 seeds) and `M6` (every class topped up to twice the normal-class count, unfiltered; validation 70.96 ± 1.43, 7 seeds).
- **Not selected (validation Score):** M1 real_dup fill 68.79; M1b real_dup balance 69.85; M2 synth fill wheeze+both 70.08; M3 20k w2.0 70.31; M3 20k w3.5 70.57; M3 80k w2.0 71.00 ± 2.36; M5 mult:1 69.72 / 69.52; two-stage M3→real 69.85 (control M0→real 69.75); stethoscope translation t=0.5: X3_other balance 69.78, X5_other mult:1 68.12, X3_same 69.99 (2 seeds). Still running at lock time and not used for the decision: M3f8 (filter 0.8; 70.69 on 2 seeds), X3f_other (69.22 on 1 seed), X3_same seed 3.
- **Phase B (official test, last epoch, seeds 1–5 each):** controls `final_M0` (real only) and `final_M1b` (real_dup balance) were submitted at about 03:15, before this lock; they contain no synthetic data and their test numbers were not used for the selection. Synthetic arms use the pool `full_s50000_w3.5_empirical` from `G_full` (all official train cycles), sampled before the lock without any evaluation. The filter probe for Phase B is the earlier C0 seed-1 model trained on the full official train set. The `--ref test` gate statistics produced by the same job are a diagnostic only.
- **Disclosure:** at lock time 4 of 5 `final_M0` seeds and 2 of 5 `final_M1b` seeds had finished and their test summary lines had been printed by `collect_nov.py` (M0 58.55, M1b 56.95). No synthetic-arm test number existed.

# 15 November 2026 18:45 IST — Novelty phase: Phase B result (first locked attempt) — negative

- **Official test, last epoch, mean ± sample std (Se / Sp / Score):** M0 real only (5 seeds) 47.27 / 69.39 / 58.33 ± 1.42; M1b real_dup balance (4) 42.93 / 71.71 / 57.32 ± 0.89; M3 synth balance (5) 44.60 / 70.74 / 57.67 ± 1.25; **M3f primary (4) 45.56 / 68.56 / 57.06 ± 0.69**; M6 (4) 46.33 / 64.31 / 55.32 ± 0.52. Seed 3 of M1b, M3f and M6 did not finish (one held launch failure, two time-outs) and was resubmitted as jobs 390000–390002.
- **Seed ensembles (average probability):** M0 59.78; M1b 58.22; M3 58.74; M3f 58.96; M6 56.13.
- **Conclusion:** the validation gain of the selected arm (+1.6) did not transfer to the test set; every synthetic arm is at or below the real-only control. Synthetic data raises "both" recall on test (25.6 → 30–31) but lowers wheeze recall (37.8 → 31–35). This result is final for the first attempt and must be reported.
- **Per-stethoscope diagnostic (seed mean Se/Sp/Score):** M0 AKG 49.7/70.5/60.1, Meditron 34.5/91.3/62.9, Litt3200 44.7/45.0/44.9; M3f AKG 47.7/69.7/58.7, Meditron 33.6/91.9/62.7, Litt3200 44.1/42.7/43.4.
- **Second attempt (declared post hoc, after seeing the test numbers above):** synthetic samples are labelled by the real-only seed ensemble (soft labels, `teacher_probs.py`, `--soft_extra`) instead of by the class they were conditioned on. Motivation: the probe agrees with the intended class for only 76–88% of abnormal synthetic samples, so hard intended labels inject label noise. Screening on validation only (arms K3, K3f, K6, K3b0.5; teacher = nov_sel_M0 seeds 1–7). Any test run of a K arm is a second look at the test set and must be reported as such next to the first attempt.

# 15 November 2026 23:00 IST — Novelty phase: second-attempt screening and generator diagnostics (validation only)

- **Phase B completed with 5 seeds per arm (official test, Se / Sp / Score):** M0 47.27 / 69.39 / 58.33 ± 1.42; M1b 42.53 / 72.37 / 57.45 ± 0.83; M3 44.60 / 70.74 / 57.67 ± 1.25; M3f 45.39 / 68.56 / 56.97 ± 0.63; M6 46.20 / 64.04 / 55.12 ± 0.63. Five-seed ensembles: M0 59.78, M1b 58.48, M3 58.74, M3f 58.44, M6 55.96; M0+M3 (10 models) 60.13, M0+M3f 59.57. Job 386080 (held after a launch failure, seeds 1–2 already done) was cancelled and its seed 3 rerun as 390000.
- **Soft-label arms (synthetic rows labelled by the nov_sel_M0 7-seed ensemble; validation Score, 3 seeds):** K3 71.08 ± 2.75; K3f 70.98 ± 3.26; K3b0.5 71.69 ± 1.12; K6 71.79 ± 2.15; K3_s20k_w2.0 70.88 ± 0.51; K3_s80k_w2.0 71.69 ± 1.07. Same size as the first attempt's validation gain (M3f 71.62), which did not transfer to test. No K arm has been run on test.
- **Generator as classifier (`diffusion_classify.py`; class with the lowest denoising error; validation, G_sel):** Score 42.99 (10k steps), 44.03 (20k), 46.53 (30k), 40.37 (50k), 35.52 (90k); AST single model 69.94. Adding the standardised score to AST log-probabilities at any tried weight gives 68.0–69.95. The generator's class conditioning generalises poorly to unseen patients and degrades with longer training.
- **Test-time re-generation (`tta_translate.py`; null class, validation, mean of 7 M0 models, baseline 69.94):** original+regenerated at t0 0.2 / 0.35 / 0.5: to AKG 70.18 / 69.51 / 68.66; same device 70.74 / 69.51 / 68.60. Regenerated only: 68.73 → 60.02 (AKG), 69.25 → 61.07 (same). No benefit from stethoscope normalisation.
- **Guidance-1.5 pool (50k checkpoint):** intended-class acc 93 / 60 / 62 / 43, KID 1.32 / 1.35 / 1.26 / 1.65, nn ratio 0.71 / 0.84 / 0.69 / 0.79. Arm K6_s50k_w1.5 running (390867).
- **Regularised generator variants on the selection split (30k steps, EMA 0.999, checkpoints every 5k, time-shift augmentation):** `gen_sel_r1` dropout 0.3 (391404); `gen_sel_r2` base 64, dropout 0.2 (391405); `gen_sel_r3` dropout 0.3 plus class-margin hinge (391406). To be compared by the generator-as-classifier validation Score before any pool is sampled.

# 18 November 2026 — Novelty phase: class-margin generator, transductive generators (user decision: keep experimenting)

- **User decision (18 Nov):** continue experimenting for a substantial gain; using the unlabelled official-test audio for the generator (transductive) is allowed. Any such run departs from the clean-test protocol and must be reported as transductive.
- **Generator as classifier, validation (G_sel variants, 7 M0 AST models for fusion):** r1 (dropout 0.3) 41.57 at 10k, 39.94 at 15k; r2 (base 64, dropout 0.2) 39.15 at 10k; **r3 (dropout 0.3 + class-margin hinge) 60.78 at 20k, 61.42 at 25k** (Se 36.7, Sp 86.1). Previous best (G_sel) 46.53. r2 20k/30k cancelled.
- **Fusion with AST (`fuse_dclf.py`, r3 25k, validation):** fused = log p_AST + w·log_softmax(−relerr/τ); only w/τ matters. Single AST models 69.94 → 71.7 (w/τ 25–50); 7-model ensemble 72.51 → 73.74 (w/τ 25–50).
- **Code:** `train.py` gains `--unlabelled_cache/--unlabelled_include/--unlabelled_frac` (rows trained with the null class; their labels are never read) and `--init_ckpt`; `sample.py` gains `--device_policy target --target_cache` (stethoscopes drawn from the target set's device histogram); `nov_sample_eval.slurm` passes `SAMPLE_ARGS`.
- **Jobs:** `gen_sel_tv` (402466; G_sel 30k fine-tuned 20k steps with validation audio unlabelled) and `gen_full_tt` (402467; G_full 30k fine-tuned with test audio unlabelled); `gen_sel_r3` continued to 40k (402484); `gen_sel_r3v` (402485; r3 recipe + validation audio unlabelled); `gen_full_r3` (402486) and `gen_full_r3t` (402487; r3 recipe + test audio unlabelled), 40k steps each; pools `sel_r3_s25000_w{2.0,3.5}_empirical` with gate (402488, 402489).

## 18 Nov 2026 — validation results and lock for the generative-classifier fusion (before any test look)

- **r3 pools as augmentation hurt (validation, 3 seeds):** K3_r3w2 68.75 ± 0.68, M3_r3w2 68.99 ± 0.93 (M0 70.00). r3 gate: intended acc N/C/W/B 97.6/67.9/97.4/46.7 (w2.0) but KID 4.7/3.9/7.6/4.1 (G_sel ~1) and nn ratio 0.95–1.49: the class-margin generator is discriminative but less realistic. No r3 pool goes to test.
- **r3 as classifier, validation:** 61.2 (10k), 60.8 (20k), 61.4 (25k), 60.7 (30k), 58.2 (35k), 59.4 (40k). `sel_tv` (G_sel 30k + 20k transductive steps on validation audio, no margin): 49.7 (G_sel 30k: 46.5).
- **Fusion:** fused = log p_AST + log_softmax(−k · relerr), relerr = mean_t err_c / mean_t err_null. At k = 25 every r3 checkpoint 20k–40k gives single AST models +1.6 to +2.2 (69.94 → 71.6–72.1) and the 7-model ensemble +0.2 to +1.2. Checkpoint averaging and noise-level subsets add nothing.
- **LOCKED for the test look (second look at the test set, declared now):** generator checkpoint 25k of `gen_full_r3` (plain) and of `gen_full_r3t` (transductive: unlabelled official-test audio in training), k = 25, all 24 noise levels, fused with each of the 5 `final_M0` real-only models and with their 5-model ensemble. Report both generators, single-model mean ± SD and ensemble, plus per-device scores. No other setting will be tried on test.

## 18 Nov 2026 — locked fusion test look: negative

- **Official test, locked settings (25k, k = 25), 5 final_M0 models:** diffusion classifier alone 45.20 (gen_full_r3) / 44.96 (gen_full_r3t, transductive); validation was 61.42. AST single 58.33 ± 1.42 → fused 57.63 ± 1.01 (r3) / 57.63 ± 0.96 (r3t); 5-model ensemble 59.78 → 57.93 / 57.75; paired bootstrap of ensemble difference −1.83 [−3.01, −0.56] / −2.02 [−3.19, −0.81]. Per device (AST ens → fused, r3): AKG 61.8 → 60.8, Meditron 63.9 → 63.7, Litt3200 45.9 → 43.4.
- **Reading:** the generator's class knowledge does not generalise from training to test patients; the validation subset (12 training patients) is not a reliable proxy for the official test split. Seeing the unlabelled test audio did not change this.

## 18 Nov 2026 — pre-declared transductive augmentation arms (third look at the test set)

- **Pool:** `full_tt_s20000_w2.0_target` from `gen_full_tt` (G_full 30k + 20k steps with the unlabelled official-test audio as null-class rows), guidance 2.0, stethoscopes drawn from the official-test device histogram (AKG 0.67, Meditron 0.17, Litt3200 0.17; labels not read), durations from training cycles of the class.
- **Arms (official test, 5 seeds, last epoch):** `T_K3` = class balancing with this pool, soft labels from the 5 final_M0 models; ablation `K3` = same with the non-transductive pool `full_s50000_w3.5_empirical`. Compared with final_M0. Both arms are reported whatever the outcome; no further test arms of this family.
- **Validation mirror (not decisive):** `K3_tv` = balancing with `sel_tv_s20000_w3.5_target`, soft labels from nov_sel_M0.

## 18 Nov 2026 — pre-declared looks 4 and 5 at the test set (approved by the user)

- **Look 4, transductive stethoscope translation (`X3_tt`):** pool `full_tt_s20000_w1.0_target_tr0.5` = official-train cycles noised to t = 0.5 and denoised by `gen_full_tt` under their own class and duration and a stethoscope drawn from the official-test device histogram (guidance 1.0). Hard labels = source labels. Class balancing (`balance`), 5 seeds on test, last epoch, vs final_M0. Validation mirror `X3_tv` (pool from `gen_sel_tv`, target = validation device histogram), 3 seeds, not decisive.
- **Look 5, test-time stethoscope normalisation:** `tta_translate.py` with `gen_full_tt` 20k on the official test cycles (labels not used for generation), 5 final_M0 models. **Primary setting:** t0 = 0.2, policy akg, prediction = mean of original probability and the mean of the 2 regenerated variants, per model and for the 5-model ensemble. The script's other grid points (t0 0.35/0.5, policy same) are computed and disclosed as secondary only.

## 18 Nov 2026 12:45 IST — results of looks 3–5 (all negative)

Official test, last epoch, Score per seed (official_test_final.json). final_M0: 59.23 / 57.78 / 57.43 / 56.88 / 60.33 (58.33 ± 1.42).
- **Look 3:** `K3` (soft labels, non-transductive pool) 57.27 / 56.92 / 59.00 / 56.73 / 59.06 = 57.80 ± 1.14. `T_K3` (transductive target-device pool, soft labels) seeds 1/4/5: 59.26 / 58.82 / 58.39 = 58.83 ± 0.44; paired with M0 on the same seeds +0.03 / +1.94 / −1.94 (mean ≈ 0). Seeds 2 and 3 timed out on a slow node (seed 1 took 3 h 20 min) and were resubmitted (403106, 403107).
- **Look 4:** `X3_tt` (transductive stethoscope translation, t = 0.5) 55.66 / 56.72 / 56.33 / 55.57 / 58.65 = 56.59 ± 1.25.
- **Look 5:** test-time normalisation with gen_full_tt, primary t0 0.2 / akg, original+regenerated: mean single-model 58.19 (original 58.33). Secondary grid 58.22–58.50 (same-device t0 0.35: 58.50, best of six, not selected).
- **Validation mirrors:** K3_tv 70.97 ± 1.53, X3_tv 71.75 ± 1.52 (M0 70.00) — again a validation gain that does not appear on test.
- Note: `probs_test_last.npy` rows are not in `test_manifest.csv` order; align by filename before computing ensembles.

## 18 Nov 2026 — user asked to keep trying (looks 6–7 and ensemble-diversity analysis, pre-declared)

- **Look 6, `A3`:** AFT recipe of the base paper (`--adversarial_ft --meta_mode mixed`, real-vs-generated domain head) + class balancing with `full_s50000_w3.5_empirical` (hard labels), 5 seeds, official test, last epoch.
- **Look 7, `A3t`:** the same with the transductive target-device pool `full_tt_s20000_w2.0_target`.
- **Ensemble diversity (declared analysis):** final_M0 seeds 6–10 are trained so that 10-model real-only ensembles can be compared at equal size with "5 M0 + 5 synthetic-arm" ensembles (arms M3, M3f, M6, K3, T_K3, X3_tt, A3, A3t), all reported. Probabilities aligned by filename.
- **AFT bug fixed (18 Nov 16:00):** in `main.py` the gradient-reversal weight `lamb` was only computed inside the `--ma_update` branch, so `--adversarial_ft` without `--ma_update` crashed in `ReverseLayerF.backward` (alpha = None). The same DANN schedule is now applied whenever `--adversarial_ft` is set. First A3/A3t submissions (403451–403456) failed before epoch 1 and were resubmitted (403496–403501).
- **T_K3 complete (5 seeds):** 59.26 / 58.38 / 57.90 / 58.82 / 58.39 = 58.55 ± 0.52 (M0 58.33 ± 1.42).
- **Ensembles (filename-aligned, `ensemble_nov.py`):** M0×5 59.78; M3 58.74; M3f 58.44; M6 55.96; K3 58.63; T_K3 58.85; X3_tt 57.56; M1b 58.48. 10-model M0+M3 60.13 (+0.35 [−0.51, +1.22]), M0+T_K3 60.14 (+0.36), M0+K3 59.74, M0+M1b 59.43, M0+X3_tt 59.32. Size-matched M0×10 control pending.

## 18 Nov 2026 17:15 IST — size-matched control, AFT partial, and pre-declared prior correction (look 8)

- **final_M0 seeds 6–10:** 58.28 / 58.56 / 58.56 / 58.81 / 58.70; 10 seeds 58.46 ± 0.96. **M0×10 ensemble 60.79**, above every 5 M0 + 5 synthetic ensemble (M0+M3 60.13, M0+T_K3 60.14): the earlier 10-model gain was ensemble size, not synthetic diversity.
- **A3 (AFT + balance) seeds 1/3/5:** 56.09 / 56.60 / 56.71 (Se 48.6, Sp 64.4); 3-model ensemble 57.70.
- **Look 8, declared before computing:** parameter-free prior correction (logit adjustment) of saved last-epoch test probabilities for every arm whose training class prior differs from the real one: p_c ← p_c · π_real,c / π_mix,c, renormalised, with π from the training class counts of the mix CSV (intended class for synthetic rows). Applied to M1b (real-duplicate control), M3, M3f, M6, K3, T_K3, X3_tt, A3, A3t; M0 unchanged. All arms reported, single-model mean and 5-model ensemble.
- **Look 8 result (`prior_correct.py`; single-model mean ± SD over seeds 1–5, 5-model ensemble):** M0 58.33 ± 1.42 / 59.78 (unchanged); M1b 57.45 → 58.30 ± 0.67 / 58.75; M3 57.67 → 59.17 ± 0.82 / 60.90; M3f 56.97 → 58.54 ± 0.89 / 60.09; M6 55.12 → 56.73 / 58.87; K3 57.80 → 59.44 ± 1.14 / 60.12; **T_K3 58.55 → 60.03 ± 0.48 / 61.22**; X3_tt 56.59 → 57.50 / 59.59; A3 (3 seeds) 56.47 → 58.20 / 59.45.
- **T_K3 corrected vs M0:** per-seed paired +1.50 / +2.24 / +2.35 / +3.29 / −0.89 (4 of 5); ensemble paired bootstrap vs M0×5 +1.42 [+0.10, +2.67], P(>0) 0.98; vs M0×10 +0.42 [−0.89, +1.71]. Per device (ensemble): AKG 62.7, Meditron 59.9, Litt3200 51.2 (M0×5: 61.8 / 63.9 / 45.9). The gain comes from specificity (Sp 78.0 vs 69.4 single-model); recall of wheeze and both falls (ensemble 26.5 / 17.5 vs 37.7 / 25.9).
- **Control:** the same correction on the real-duplicate balancing control M1b gives no gain (58.30 single, 58.75 ensemble; −2.03 vs M0×10), so the gain needs the synthetic samples, not the correction alone.
- **Caveat for reporting:** this is the eighth look at the official test set and T_K3 uses unlabelled test audio (transductive). All looks are listed in this log.
- **A3 / A3t complete (5 seeds each, AFT fix applied):** raw A3 56.42 ± 0.80 (ens 57.42), A3t 57.48 ± 1.10 (ens 58.59); prior-corrected A3 57.94 ± 0.67 (ens 60.18), A3t 58.83 ± 1.08 (ens 61.12). 10-model ensembles M0+A3 59.46, M0+A3t 59.54 (M0×10 60.79). AFT does not beat plain balancing.
- **Validation check of the look-8 correction (`val_prior_correct.py`, sel arms, real prior = train minus val counts 1804/1066/445/323; no test data):** per-seed mean raw → corrected: M0 69.94 (unchanged); M1b 69.85 → 69.70; M1 68.79 → 69.17; M3_s50k_w3.5 70.83 → 72.07; M3f_s50k_w3.5 71.62 → 71.85; M3_w3.5 70.57 → 71.13; M3_w2.0 70.51 → 71.22; M6 70.96 → 70.99; K3 71.08 → 70.66; K6 71.79 → 71.42; K3_tv 70.97 → 70.22; X3_tv 71.88 → 71.68; K3_s20k_w2.0 70.88 → 70.88. Reading: on validation every synthetic arm already beats M0 per model without correction; the correction helps the M3 (hard-label) family and is neutral to slightly negative for the soft-label K family and for the real-duplicate control. Validation therefore supports the correction for M3, not specifically for T_K3.
- **Look 9, declared before training (18 Nov ~17:40 IST):** size-matched replication. Train final seeds 6–10 of `T_K3` (primary) and `M3` (non-transductive secondary) with unchanged recipes. Report, with the look-8 correction fixed as defined (no new parameters): single-model mean ± SD over seeds 1–10 vs M0 seeds 1–10 (58.46 ± 0.96), paired per seed; 10-model corrected ensembles vs M0×10 (60.79) with paired bootstrap. Both arms reported whatever the outcome.
- **Look 9 jobs:** T_K3 seeds 6–10 = 403773–403777; M3 seeds 6–10 = 403779 (s6), 403783 (s7, dgx after l40 submit cap), 403780–403782 (s8–10). Analysis script `SpecDiff/look9.py` written and dry-run on seeds 1–5 before any new result; it reproduces look 8 exactly (ensemble = class weights applied to the averaged raw probabilities, as in `prior_correct.py`).
- **Sensitivity disclosed:** applying the weights per model and renormalising before averaging instead gives T_K3×5 ensemble 60.83 (vs 61.22) and M3×5 60.56 (vs 60.90). The look-8 definition is kept for look 9; the alternative is not used for any decision.
- **Look 9 result (`look9.py`, all 10 jobs completed 40-55 min each):** prior-corrected, official test, seeds 1-10.
  - **T_K3:** single 59.57 ± 0.64 (Se 42.3, Sp 76.9) vs M0 58.46 ± 0.96; paired per seed +1.50 / +2.24 / +2.35 / +3.29 / −0.89 / +1.31 / +0.72 / −0.06 / +0.21 / +0.45 → mean +1.11, 95% CI [+0.20, +2.02], t-test p = 0.022, Wilcoxon p = 0.027, 8 of 10 wins. New seeds 6-10 alone: +0.53 (regression from the first five). T_K3×10 ensemble 60.98 vs M0×10 60.79: +0.20 [−1.05, +1.41], P(>0) 0.64 (not significant). Ensemble per device AKG 62.5 / Meditron 60.4 / Litt3200 50.6; recall N/C/W/B 81.7 / 53.5 / 27.0 / 16.1. Raw (uncorrected) T_K3 10 seeds 58.25 ± 0.59.
  - **M3 (non-transductive):** single 58.83 ± 0.92; paired +0.37, CI [−0.38, +1.12], p = 0.29, 6 of 10 wins; ensemble 60.91 vs 60.79 (+0.12 [−1.29, +1.47]). Raw 57.24 ± 1.15.
  - **Reading:** with the fixed correction, transductive synthetic balancing gives a small but statistically supported per-model gain (+1.1 Score, 8/10 seeds), plus lower seed variance. It does not improve over a size-matched real-only ensemble. The non-transductive arm is not significant. The look-8 per-model estimate (+1.70) was optimistic; the 10-seed estimate (+1.11) is the one to report.

## 18 Nov 2026 evening — deliverables folder built (no new test looks)

- `deliverables/` at the project root: results tables (CSV + `results/RESULTS.md`), figures F1-F8 (PNG 300 dpi + PDF), `logs/TEST_LOOK_REGISTER.md`, copies of this log, NOVELTY_PLAN, all novelty-phase SLURM logs, final-run configs/test json (65 runs), generator/pool configs, code snapshot with diffs of the 18 Nov changes, poster and report outlines (no prose).
- Built by `SpecDiff/make_deliverables.py` from saved probabilities and json only; it reproduces the look-8 and look-9 numbers exactly. User decided (18 Nov) to stop experimenting and build deliverables.

## 22 Nov 2026 16:35-16:50 IST — look 10: post-hoc diagnostics on saved outputs (no training)
Look 10 (22 Nov): post-hoc diagnostics on saved outputs (bias sweep on M0, near-copy distances to test, same-probe gate). No model, threshold or arm was selected from them. Jobs 409355-409357 (dgx). Outputs and every deviation, including a fix to the bias_sweep K3 loader bug: `report_extra_1122/README.md`.

# IE 643 ICBHI respiratory sound classification

Code-only project snapshot. It excludes the ICBHI audio, generated audio, pretrained AST weights, trained checkpoints, run logs, and result tables/figures. See the sibling handoff `CODE_MAP.md`, `model_card.md`, and `tables/` for the final artifact map and results.

## Setup

Use Python 3.11. Install a PyTorch/TorchAudio pair compatible with your system first, then:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the compatible torch, torchvision, and torchaudio build for this host.
python -m pip install -r requirements.txt
```

The AST implementation uses `timm==0.4.5`. Training from scratch also needs the AudioSet-pretrained AST checkpoint at `Classification/pretrained_models/audioset_10_10_0.4593.pth`; obtain it separately and keep it out of Git.

## Reproduce the official split

Obtain the ICBHI 2017 audio and official `ICBHI_challenge_train_test.txt` split from the dataset provider. Set the paths in `hpc/slurm/prepare_split.slurm`, then submit on PRAJNA:

```bash
sbatch hpc/slurm/prepare_split.slurm
sbatch hpc/slurm/verify_split.slurm
```

The expected processed-cycle counts are 4,142 train and 2,756 test. The official split is recording-level and shares patients 156 and 218 across partitions; it is not patient-disjoint.

## Train and evaluate

Use the supplied Slurm jobs from a compute cluster; do not run model training on a login node. Before submission, create `hpc/logs/`, configure the input paths in `prepare_split.slurm`, and stage processed data/checkpoints at the project root. The copied scripts use this repository root as `SLURM_SUBMIT_DIR`.

```bash
# Example final candidate run (edit job paths/seed flags as needed):
mkdir -p hpc/logs
sbatch hpc/slurm/train_wp3_c4_remaining.slurm
# Evaluate the predeclared final probability ensembles:
sbatch hpc/slurm/evaluate_wp6_final_ensembles.slurm
```

Other scripts cover the baseline, Patch-Mix, synthetic/AFT runs, DiffWave, and QC. `Classification/collect_results.py` collects run logs; `Classification/audit_metric_families.py` recomputes both metric families from saved confusion matrices. The headline comparison uses official classwise ICBHI Se/Sp/Score. Cycle-micro values are a separate family and should be labelled accordingly.

## Run the demo

The separate handoff folder contains the Gradio app, eight official-test demo cycles, and the single C4 seed-1 demo checkpoint. The project's tested local stack is PyTorch 2.0.1, TorchVision 0.15.2, TorchAudio 2.0.2 with CUDA 11.8. Install it first, then the demo dependencies. From the complete handoff root:

```bash
python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r demo/requirements.txt
python demo/app.py
python demo/test_demo_smoke.py
```

The app accepts one respiratory-cycle WAV and automatically uses CUDA when available, otherwise CPU. Full recordings need annotation-based cycle segmentation before upload. The four-clip smoke check runs locally without Slurm. For CPU-only use, install the same versions from PyTorch's CPU wheel index. These commands follow the official [PyTorch previous-version instructions](https://docs.pytorch.org/get-started/previous-versions/). The demo uses C4 seed 1 for laptop loading; Ens-AST is the reported headline ensemble and is not used in the app.

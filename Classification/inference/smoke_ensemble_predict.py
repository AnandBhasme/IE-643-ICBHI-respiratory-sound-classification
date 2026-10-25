"""Smoke the selected Ens-AST ensemble on one official test clip per class."""
from pathlib import Path
import csv
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.predict import CLASSES, predict_ensemble

root = Path(sys.argv[1])
run_root = root / "artifacts/runs"
run_ids = [
    *[f"icbhi_ast_ce_wp3_C{i}_seed{s}_bf16" for i in range(4) for s in (1, 2, 3)],
    "icbhi_ast_patchmix_cl_wp3_C4_seed1_bf16_retry1",
    "icbhi_ast_patchmix_cl_wp3_C4_seed2_bf16_retry2",
    "icbhi_ast_patchmix_cl_wp3_C4_seed3_bf16_retry2",
]
checkpoints = [run_root / run / "last.pth" for run in run_ids]
missing = [str(path) for path in checkpoints if not path.is_file()]
if missing:
    raise FileNotFoundError("Missing Ens-AST last-epoch checkpoints: " + ", ".join(missing))

manifest = root / "artifacts/data/official_icbhi/splits/test_manifest.csv"
selected = {}
with manifest.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
        label = int(row["label"])
        if label not in selected:
            selected[label] = root / "artifacts/data/official_icbhi/test/real" / row["filename"]
        if len(selected) == 4:
            break
if set(selected) != set(range(4)):
    raise RuntimeError(f"Could not find one official test clip for every class: {selected}")

paths = [selected[i] for i in range(4)]
print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
print(f"ensemble_members={len(checkpoints)}; checkpoints are streamed one at a time", flush=True)
predictions = predict_ensemble(paths, checkpoints, device="cuda")
for label, path, probabilities in zip(range(4), paths, predictions):
    values = np.asarray([probabilities[name] for name in CLASSES])
    if not np.isfinite(values).all() or not np.isclose(values.sum(), 1.0, atol=1e-5):
        raise AssertionError(f"Invalid ensemble probabilities for {path}: {probabilities}")
    print(f"true={CLASSES[label]} file={path.name} probabilities={probabilities}", flush=True)
print(f"peak_cuda_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}", flush=True)

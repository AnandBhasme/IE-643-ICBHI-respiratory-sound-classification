"""Run three offline predictions on real official test cycles."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.predict import predict


project_root = Path(sys.argv[1])
checkpoint = Path(sys.argv[2])
test_dir = project_root / "artifacts/data/official_icbhi/test/real"
files = sorted(test_dir.glob("*.wav"))[:3]
if len(files) != 3:
    raise RuntimeError(f"Expected three official test WAVs under {test_dir}, found {len(files)}")

print(f"checkpoint={checkpoint}")
print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
for path in files:
    probabilities = predict(path, checkpoint=checkpoint, device="cuda")
    cpu_probabilities = predict(path, checkpoint=checkpoint, device="cpu")
    print(f"{path.name}\tgpu={probabilities}\tcpu={cpu_probabilities}")
print(f"peak_cuda_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}")

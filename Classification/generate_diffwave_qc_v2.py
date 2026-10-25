#!/usr/bin/env python3
"""Generate an isolated DiffWave v2 diagnostic pool from official-train mels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DIFFWAVE = ROOT / "Code" / "Code" / "AFT" / "Diffwave"
sys.path.insert(0, str(DIFFWAVE))
import inference as diffwave_inference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=64325)
    args = parser.parse_args()

    if "/artifacts/diag/" not in args.output.as_posix() + "/":
        raise ValueError(f"Diagnostic audio must stay under artifacts/diag/: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with args.args.open(encoding="utf-8") as f:
        train_args = json.load(f)
    params = diffwave_inference.get_params(train_args)
    step_text = args.checkpoint.stem.rsplit("-", 1)[-1]
    checkpoint_step = int(step_text) if step_text.isdigit() else None
    rows = []
    for cls in (2, 3):
        source_dir = ROOT / "artifacts/data/diffwave/wav_4secs_16000" / f"label{cls}" / "train"
        specs = sorted(source_dir.glob("*.wav.spec.npy"))
        if len(specs) < args.per_class:
            raise ValueError(f"Only {len(specs)} train conditioning mels for class {cls}")
        selected = random.sample(specs, args.per_class)
        out_dir = args.output / f"class{cls}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx, spec_path in enumerate(selected, start=1):
            sample_seed = args.seed + cls * 1000 + idx
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            source_name = spec_path.name.removesuffix(".spec.npy")
            mel = torch.from_numpy(np.load(spec_path)).float()
            audio, sr = diffwave_inference.predict(
                spectrogram=mel,
                model_dir=str(args.checkpoint),
                params=params,
                attrdict=diffwave_inference.AttrDict,
                class_condition=None,
                device=torch.device("cuda"),
                fast_sampling=True,
            )
            if sr != 16000:
                raise AssertionError(f"Expected 16 kHz output, received {sr}")
            output_name = f"{source_name.removesuffix('.wav')}_index={idx}.wav"
            sf.write(out_dir / output_name, audio.detach().cpu().squeeze().numpy(), sr, subtype="PCM_16")
            rows.append({
                "filename": output_name,
                "source_file": source_name,
                "label": cls,
                "checkpoint": str(args.checkpoint),
                "checkpoint_step": checkpoint_step,
                "sampler": "fast_6_step",
                "seed": sample_seed,
                "split": "official_train_conditioning_mel",
                "diagnostic_only": True,
            })
            if idx % 20 == 0:
                print(f"Generated v2 class{cls}: {idx}/{args.per_class}", flush=True)

    with (args.output / "v2_pool_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} diagnostic clips to {args.output}", flush=True)


if __name__ == "__main__":
    main()

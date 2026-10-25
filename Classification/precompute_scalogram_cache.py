#!/usr/bin/env python3
"""Build official-train/test CWT features on a Slurm compute node."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from util.scalogram import morlet_filterbank, scalogram


def rows_from(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_wave(path: Path, n_samples: int):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16_000:
        from math import gcd
        d = gcd(int(sr), 16_000)
        audio = resample_poly(audio, 16_000 // d, int(sr) // d).astype(np.float32)
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    elif len(audio) < n_samples:
        audio = np.resize(audio, n_samples).astype(np.float32)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="per-clip FFT batch; kept small for predictable GPU memory")
    args = parser.parse_args()
    device = torch.device(args.device)
    n_samples, n_scales, frames = 80_000, 128, 498
    bank = morlet_filterbank(n_samples, n_scales=n_scales, device=device)
    totals = {"sum": 0.0, "sum_sq": 0.0, "count": 0, "clips": 0,
              "float16_files": 0, "float32_fallback_files": 0,
              "max_cache_error": 0.0}
    train_sum = {"sum": 0.0, "sum_sq": 0.0, "count": 0}
    agreement_checks = []

    for split, manifest, folder in (
        ("train", "train_manifest.csv", args.data_root / "training" / "real"),
        ("test", "test_manifest.csv", args.data_root / "test" / "real"),
    ):
        split_rows = rows_from(args.split_dir / manifest)
        out_dir = args.cache_root / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for index, row in enumerate(split_rows, 1):
            filename = row["filename"]
            source = folder / filename
            waveform = load_wave(source, n_samples)
            feature = scalogram(torch.from_numpy(waveform), bank).cpu().numpy().astype(np.float32)
            if feature.shape != (frames, n_scales) or not np.isfinite(feature).all():
                raise RuntimeError(f"Invalid scalogram for {filename}: {feature.shape}")
            quantized = feature.astype(np.float16)
            error = float(np.max(np.abs(feature - quantized.astype(np.float32))))
            dtype = np.float16 if error < 0.01 else np.float32
            cached = feature.astype(dtype)
            cache_path = out_dir / f"{filename}.npy"
            np.save(cache_path, cached)
            # Check the stored representation against a fresh on-the-fly CWT
            # on five deterministic clips, including the float16 round trip.
            if len(agreement_checks) < 5:
                fresh = scalogram(torch.from_numpy(waveform), bank).cpu().numpy()
                loaded = np.load(cache_path).astype(np.float32)
                max_abs_error = float(np.max(np.abs(fresh - loaded)))
                if max_abs_error >= 0.01:
                    raise RuntimeError(
                        f"Cached/on-the-fly scalogram mismatch for {filename}: "
                        f"max_abs_error={max_abs_error:.6f} dB"
                    )
                agreement_checks.append({"split": split, "filename": filename,
                                         "max_abs_error_db": max_abs_error})
            totals["sum"] += float(feature.sum(dtype=np.float64))
            totals["sum_sq"] += float(np.square(feature, dtype=np.float64).sum())
            totals["count"] += int(feature.size)
            totals["clips"] += 1
            totals["max_cache_error"] = max(totals["max_cache_error"], error if dtype == np.float16 else 0.0)
            if dtype == np.float16:
                totals["float16_files"] += 1
            else:
                totals["float32_fallback_files"] += 1
            if split == "train":
                train_sum["sum"] += float(feature.sum(dtype=np.float64))
                train_sum["sum_sq"] += float(np.square(feature, dtype=np.float64).sum())
                train_sum["count"] += int(feature.size)
            if index % 500 == 0 or index == len(split_rows):
                print(f"{split}: {index}/{len(split_rows)} clips cached", flush=True)

    mean = train_sum["sum"] / train_sum["count"]
    variance = max(train_sum["sum_sq"] / train_sum["count"] - mean * mean, 1e-12)
    metadata = {
        "sample_rate": 16_000, "duration_seconds": 5.0, "n_samples": n_samples,
        "n_scales": n_scales, "f_min": 50.0, "f_max": 4_000.0,
        "wavelet": "analytic Morlet, w0=6", "pool_hop_samples": 160,
        "frames": frames, "feature_order": "time,frequency", "stored_dtype": "float16 with float32 fallback when abs quantization error >= 0.01 dB",
        "train_mean_db": mean, "train_std_db": math.sqrt(variance),
        "cache_counts": totals,
        "agreement_check": {
            "clips": agreement_checks,
            "max_abs_error_db": max(x["max_abs_error_db"] for x in agreement_checks),
            "tolerance_db": 0.01,
        },
    }
    args.cache_root.mkdir(parents=True, exist_ok=True)
    (args.cache_root / "stats.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

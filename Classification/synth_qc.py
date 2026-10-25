#!/usr/bin/env python3
"""QC DiffWave v1 audio against real official-train cycles."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "inference"))
from predict import load_model, preprocess  # noqa: E402


def read_audio(path: Path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio, sr


def audio_stats(paths: list[Path]):
    rms, peak, clipped, finite = [], [], [], True
    for path in paths:
        x, sr = read_audio(path)
        if sr != 16000:
            raise ValueError(f"Unexpected sample rate {sr}: {path}")
        finite &= bool(np.isfinite(x).all())
        rms.append(float(np.sqrt(np.mean(x.astype(np.float64) ** 2))))
        peak.append(float(np.max(np.abs(x))))
        clipped.append(float(np.mean(np.abs(x) >= 0.999)))
    return {
        "count": len(paths), "finite": finite,
        "rms_mean": float(np.mean(rms)), "rms_std": float(np.std(rms)),
        "peak_mean": float(np.mean(peak)), "peak_p99": float(np.quantile(peak, .99)),
        "clipped_fraction_mean": float(np.mean(clipped)),
    }


def classify(paths: list[Path], checkpoint: Path, device: str, batch_size: int):
    model, head = load_model(checkpoint, device=device)
    predictions = []
    for start in range(0, len(paths), batch_size):
        batch = torch.cat([preprocess(path) for path in paths[start:start + batch_size]], dim=0).to(device)
        with torch.inference_mode():
            logits = head(model(batch))
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())
    return predictions


def mel(path: Path):
    x, _ = read_audio(path)
    w = torch.from_numpy(x).float()
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, win_length=1024, hop_length=256, n_fft=1024,
        f_min=20.0, f_max=8000.0, n_mels=80, power=1.0, normalized=True,
    )(w)
    db = 20 * torch.log10(torch.clamp(spec, min=1e-5)) - 20
    return torch.clamp((db + 100) / 100, 0.0, 1.0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-root", type=Path, required=True)
    p.add_argument("--synth-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--real-per-class", type=int, default=300)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    synth_paths = sorted(args.synth_root.rglob("*.wav"))
    by_class = {c: [p for p in synth_paths if f"class{c}" in p.parts] for c in (1, 2, 3)}
    rows = []
    names = ("normal", "crackle", "wheeze", "both")
    rng = np.random.default_rng(1)
    for cls in (1, 2, 3):
        synth = by_class[cls]
        if not synth:
            rows.append({"class": cls, "class_name": names[cls], "status": "no synthetic pool"})
            continue
        src_names = {p.name.split("_index=")[0] + ".wav" for p in synth}
        real_all = sorted(p for p in args.real_root.glob("*.wav") if p.name in src_names)
        if not real_all:
            real_all = sorted(args.real_root.glob("*.wav"))
            real_all = [p for p in real_all if p.name.endswith(f"_label_{cls}.wav")]
        real = real_all
        if len(real) > args.real_per_class:
            indices = np.sort(rng.choice(len(real), args.real_per_class, replace=False))
            real = [real[i] for i in indices]
        real_stat, synth_stat = audio_stats(real), audio_stats(synth)
        real_pred = classify(real, args.checkpoint, args.device, args.batch_size)
        synth_pred = classify(synth, args.checkpoint, args.device, args.batch_size)
        rows.append({
            "class": cls, "class_name": names[cls], "status": "ok",
            "real_n": len(real), "synth_n": len(synth),
            "real_intended_acc": float(np.mean(np.asarray(real_pred) == cls)),
            "synth_intended_acc": float(np.mean(np.asarray(synth_pred) == cls)),
            **{f"real_{k}": v for k, v in real_stat.items() if k != "count"},
            **{f"synth_{k}": v for k, v in synth_stat.items() if k != "count"},
        })

        fig, axes = plt.subplots(2, 4, figsize=(14, 6), constrained_layout=True)
        for row_idx, paths in enumerate((real, synth)):
            chosen = paths[:4]
            for col, path in enumerate(chosen):
                axes[row_idx, col].imshow(mel(path), origin="lower", aspect="auto", vmin=0, vmax=1)
                axes[row_idx, col].set_title(path.name[:25], fontsize=7)
                axes[row_idx, col].set_axis_off()
            for col in range(len(chosen), 4):
                axes[row_idx, col].set_axis_off()
        axes[0, 0].set_ylabel("real", labelpad=3)
        axes[1, 0].set_ylabel("synthetic", labelpad=3)
        fig.suptitle(f"{names[cls]} (class {cls}) — real vs DiffWave v1")
        fig.savefig(args.output / f"mel_grid_class{cls}.png", dpi=150)
        plt.close(fig)

    fields = sorted({key for row in rows for key in row})
    with (args.output / "qc_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"QC saved: {args.output / 'qc_summary.csv'}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()

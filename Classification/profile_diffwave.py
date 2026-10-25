#!/usr/bin/env python3
"""Compare measured DiffWave throughput settings from the step-20k checkpoint."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DW = ROOT / "Code" / "Code" / "AFT" / "Diffwave"
sys.path.insert(0, str(DW))
from dataset import Collator, ConditionalDataset  # noqa: E402
from learner import DiffWaveLearner  # noqa: E402
from main import AttrDict  # noqa: E402
from model import DiffWave  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("DiffWave throughput profiling requires a scheduled CUDA GPU")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.set_num_threads(4)
    device = torch.device("cuda")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    specs = np.linspace(1e-4, 0.05, 50).tolist()
    results = []

    # ConditionalDataset loads all 4,142 waveforms and mel arrays into RAM once,
    # matching the repository's existing implementation.
    base = AttrDict(
        sample_rate=16000, n_mels=80, n_fft=1024, hop_samples=256,
        crop_mel_frames=62, residual_layers=30, residual_channels=64,
        dilation_cycle_length=10, unconditional=False, class_condition=False,
        noise_schedule=specs, inference_noise_schedule=[.0001, .001, .01, .05, .2, .5],
        audio_len=64000, max_grad_norm=False, learning_rate=2e-4,
    )
    print("Loading the preloaded in-RAM dataset once for all three profiles", flush=True)
    t_setup = time.perf_counter()
    dataset = ConditionalDataset([str(args.data_dir)])
    print(f"dataset_loaded_seconds={time.perf_counter() - t_setup:.2f}; examples={len(dataset)}", flush=True)

    profiles = [
        {"name": "baseline_workers8_batch16_fp32", "workers": 8, "batch": 16, "bf16": False, "persistent": False},
        {"name": "workers2_batch16_fp32_persistent", "workers": 2, "batch": 16, "bf16": False, "persistent": True},
        {"name": "workers2_batch32_bf16_persistent", "workers": 2, "batch": 32, "bf16": True, "persistent": True},
    ]
    for cfg in profiles:
        params = AttrDict(dict(base))
        params.batch_size = cfg["batch"]
        params.num_workers = cfg["workers"]
        params.persistent_workers = cfg["persistent"]
        # Reuse the same fully preloaded dataset but construct a new loader per profile.
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=params.batch_size,
            collate_fn=Collator(params).collate, shuffle=True, num_workers=params.num_workers,
            persistent_workers=params.persistent_workers and params.num_workers > 0,
            pin_memory=True, drop_last=True,
        )
        model = DiffWave(params).to(device)
        model.load_state_dict(checkpoint["model"])
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        learner = DiffWaveLearner("/tmp/ie643_diffwave_profile", model, loader, optimizer, params,
                                  fp16=False, bf16=cfg["bf16"])
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        step_times, losses = [], []
        iterator = iter(loader)
        for step in range(args.steps):
            batch_start = time.perf_counter()
            try:
                features = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                features = next(iterator)
            features = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in features.items()}
            loss = learner.train_step(features)
            torch.cuda.synchronize()
            duration = time.perf_counter() - batch_start
            value = float(loss.detach().float().cpu())
            if not np.isfinite(value):
                raise RuntimeError(f"Non-finite loss in profile {cfg['name']} at step {step + 1}")
            losses.append(value)
            step_times.append(duration)
            if (step + 1) % 25 == 0:
                print(f"{cfg['name']} step={step + 1}/{args.steps} loss={value:.6f} step_s={duration:.3f}", flush=True)
        elapsed = time.perf_counter() - started
        steady = step_times[min(20, len(step_times)):]
        result = {
            **cfg, "steps": args.steps, "elapsed_seconds": elapsed,
            "steps_per_second": args.steps / elapsed,
            "steady_steps_per_second": len(steady) / sum(steady),
            "mean_step_seconds_after_warmup": float(np.mean(steady)),
            "first_loss": losses[0], "last_loss": losses[-1],
            "mean_loss_last_20": float(np.mean(losses[-20:])),
            "loss_finite": True,
            "peak_cuda_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
        results.append(result)
        print(result, flush=True)
        del iterator, loader, learner, optimizer, model
        torch.cuda.empty_cache()

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"Wrote profile results to {args.output}", flush=True)


if __name__ == "__main__":
    main()

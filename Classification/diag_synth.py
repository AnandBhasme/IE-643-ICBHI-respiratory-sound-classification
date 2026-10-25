#!/usr/bin/env python3
"""Diagnose v1 synthetic audio; all generated diagnostics stay under artifacts/diag."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE / "inference"))
from predict import load_model as load_ast, preprocess as ast_preprocess  # noqa: E402
sys.path.insert(0, str(ROOT / "Code" / "Code" / "AFT" / "Diffwave"))
import inference as diffwave_inference  # noqa: E402


CLASS_NAMES = ("normal", "crackle", "wheeze", "both")
SR = 16000
HOP = 256
N_FFT = 1024


def load_mono(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} for {path}")
    return np.asarray(audio, dtype=np.float32)


def class_distribution(predictions: list[int]) -> dict[str, int]:
    counts = Counter(predictions)
    return {CLASS_NAMES[i]: int(counts[i]) for i in range(4)}


def ast_predict_arrays(arrays: list[np.ndarray], checkpoint: Path, device: str, batch_size: int):
    model, head = load_ast(checkpoint, device=device)
    result = []
    for offset in range(0, len(arrays), batch_size):
        batch = torch.cat(
            [ast_preprocess(x, sample_rate=SR) for x in arrays[offset:offset + batch_size]], dim=0
        ).to(device)
        with torch.inference_mode():
            probs = torch.softmax(head(model(batch)), dim=-1).cpu().numpy()
        result.extend(probs)
    return np.asarray(result)


def ast_predict_paths(paths: list[Path], checkpoint: Path, device: str, batch_size: int):
    model, head = load_ast(checkpoint, device=device)
    result = []
    for offset in range(0, len(paths), batch_size):
        batch = torch.cat(
            [ast_preprocess(path) for path in paths[offset:offset + batch_size]], dim=0
        ).to(device)
        with torch.inference_mode():
            probs = torch.softmax(head(model(batch)), dim=-1).cpu().numpy()
        result.extend(probs)
    return np.asarray(result)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


def gain_match(synthetic: np.ndarray, target_rms: float, peak_limit: float = 0.99) -> np.ndarray:
    current = rms(synthetic)
    if not np.isfinite(current) or current <= 1e-12:
        return synthetic.copy()
    matched = synthetic * (target_rms / current)
    peak = float(np.max(np.abs(matched)))
    if peak > peak_limit:
        matched = matched * (peak_limit / peak)
    return matched.astype(np.float32, copy=False)


def get_mel(audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(np.asarray(audio, dtype=np.float32))
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, win_length=HOP * 4, hop_length=HOP, n_fft=N_FFT,
        f_min=20.0, f_max=SR / 2.0, n_mels=80, power=1.0, normalized=True,
    )(waveform)
    db = 20.0 * torch.log10(torch.clamp(mel, min=1e-5)) - 20.0
    return torch.clamp((db + 100.0) / 100.0, 0.0, 1.0)


def save_rows(path: Path, rows: list[dict]):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sample_vocoder(mels: list[torch.Tensor], model_path: Path, params, device: str, fast: bool, seed: int):
    audio = []
    for index, mel in enumerate(mels):
        torch.manual_seed(seed + index)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed + index)
        output, sr = diffwave_inference.predict(
            spectrogram=mel,
            model_dir=str(model_path),
            params=params,
            attrdict=diffwave_inference.AttrDict,
            class_condition=None,
            device=torch.device(device),
            fast_sampling=fast,
        )
        if sr != SR:
            raise AssertionError(f"DiffWave produced unexpected sample rate {sr}")
        audio.append(output.detach().cpu().squeeze().numpy().astype(np.float32))
        if (index + 1) % 10 == 0:
            print(f"sampler={'fast' if fast else 'full'} completed {index + 1}/{len(mels)}", flush=True)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth-root", type=Path, required=True)
    parser.add_argument("--real-train", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--real-test", type=Path, required=True)
    parser.add_argument("--ast-best", type=Path, required=True)
    parser.add_argument("--diffwave-checkpoint", type=Path, required=True)
    parser.add_argument("--diffwave-args", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampler-per-class", type=int, default=50)
    parser.add_argument("--test-vocoder-per-class", type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if "/diag/" not in args.output.as_posix() + "/":
        raise ValueError(f"All generated diagnostics must be under artifacts/diag/: {args.output}")
    torch.set_num_threads(4)
    rng = np.random.default_rng(643)

    # Use E1's official test confusion matrices as the fair real-audio reference.
    metrics_path = ROOT / "artifacts/runs/icbhi_ast_ce_E1_seed1_official/metrics.csv"
    with metrics_path.open(newline="", encoding="utf-8") as f:
        metric_rows = list(csv.DictReader(f))
    best_row = max(metric_rows, key=lambda row: float(row["Score"]))
    last_row = metric_rows[-1]
    test_counts = {2: 385, 3: 143}
    real_test_recall = {
        "best": {cls: float(best_row[f"recall_{CLASS_NAMES[cls]}"]) for cls in (2, 3)},
        "last": {cls: float(last_row[f"recall_{CLASS_NAMES[cls]}"]) for cls in (2, 3)},
    }

    # (a,c) Raw and source-RMS-matched v1 samples; report intended accuracy and class distribution.
    synth_paths = sorted(args.synth_root.rglob("*.wav"))
    ast_rows = []
    synthetic_records = {}
    source_by_name = {p.name: p for p in args.real_train.glob("*.wav")}
    best_checkpoint = args.ast_best
    for cls in (2, 3):
        paths = [p for p in synth_paths if f"class{cls}" in p.parts]
        arrays, matched_arrays = [], []
        for path in paths:
            source_name = path.name.split("_index=")[0] + ".wav"
            source_path = source_by_name.get(source_name)
            if source_path is None:
                raise FileNotFoundError(f"No real training source for {path.name}: {source_name}")
            source = load_mono(source_path)
            audio = load_mono(path)
            arrays.append(audio)
            matched_arrays.append(gain_match(audio, rms(source)))
        raw_probs = ast_predict_arrays(arrays, best_checkpoint, args.device, args.batch_size)
        matched_probs = ast_predict_arrays(matched_arrays, best_checkpoint, args.device, args.batch_size)
        synthetic_records[cls] = (paths, arrays, matched_arrays)
        for mode, probs in (("raw", raw_probs), ("source_rms_matched", matched_probs)):
            preds = probs.argmax(axis=1).tolist()
            ast_rows.append({
                "section": "v1_synthetic", "class": cls, "class_name": CLASS_NAMES[cls],
                "audio_mode": mode, "n": len(paths),
                "intended_accuracy_pct": 100.0 * float(np.mean(np.asarray(preds) == cls)),
                "predicted_class_counts_json": json.dumps(class_distribution(preds), sort_keys=True),
                "real_test_recall_e1_best_pct": real_test_recall["best"][cls],
                "real_test_recall_e1_last_pct": real_test_recall["last"][cls],
                "test_class_support": test_counts[cls],
            })
            print(ast_rows[-1], flush=True)

    # (b) Compare the existing six-step sampler to the trained 50-step reverse process.
    model_dir = args.diffwave_checkpoint.parent
    with args.diffwave_args.open(encoding="utf-8") as f:
        dw_args = json.load(f)
    params = diffwave_inference.get_params(dw_args)
    sampler_rows = []
    sampler_predictions = {}
    for cls in (2, 3):
        paths = synthetic_records[cls][0]
        source_names = sorted({p.name.split("_index=")[0] + ".wav" for p in paths})
        if len(source_names) < args.sampler_per_class:
            raise ValueError(f"Class {cls} has only {len(source_names)} unique sources")
        selected = rng.choice(source_names, args.sampler_per_class, replace=False).tolist()
        mels = []
        for name in selected:
            spec_path = ROOT / "artifacts" / "data" / "diffwave" / "wav_4secs_16000" / f"label{cls}" / "train" / f"{name}.spec.npy"
            if not spec_path.is_file():
                raise FileNotFoundError(spec_path)
            mels.append(torch.from_numpy(np.load(spec_path)).float())
        for fast in (True, False):
            waves = sample_vocoder(mels, args.diffwave_checkpoint, params, args.device, fast, seed=2000 + cls * 10)
            probs = ast_predict_arrays(waves, best_checkpoint, args.device, args.batch_size)
            preds = probs.argmax(axis=1).tolist()
            sampler_predictions[(cls, fast)] = float(np.mean(np.asarray(preds) == cls))
            sampler_rows.append({
                "class": cls, "class_name": CLASS_NAMES[cls],
                "sampler": "fast_6_step" if fast else "full_50_step", "n": len(waves),
                "intended_accuracy_pct": 100.0 * sampler_predictions[(cls, fast)],
                "predicted_class_counts_json": json.dumps(class_distribution(preds), sort_keys=True),
            })
            print(sampler_rows[-1], flush=True)

    # Preselect sampler from the train-source diagnostic only; never tune on the held-out test set.
    fast_mean = np.mean([sampler_predictions[(cls, True)] for cls in (2, 3)])
    full_mean = np.mean([sampler_predictions[(cls, False)] for cls in (2, 3)])
    use_fast = bool(fast_mean >= full_mean)
    selected_sampler = "fast_6_step" if use_fast else "full_50_step"
    print(f"Selected held-out diagnostic sampler from train-source results: {selected_sampler}", flush=True)

    # (d,e) Held-out test cycles are sampled for diagnostics only and saved only under artifacts/diag.
    test_rows = []
    with args.test_manifest.open(newline="", encoding="utf-8") as f:
        test_items = list(csv.DictReader(f))
    vocoder_root = args.output / "test_vocoder"
    for cls in (2, 3):
        candidates = [row for row in test_items if int(row["label"]) == cls]
        if len(candidates) < args.test_vocoder_per_class:
            raise ValueError(f"Not enough held-out test examples for class {cls}")
        selected_rows = [candidates[i] for i in sorted(rng.choice(len(candidates), args.test_vocoder_per_class, replace=False))]
        input_paths = [args.real_test / row["filename"] for row in selected_rows]
        mels, originals = [], []
        for path in input_paths:
            audio = load_mono(path)[:4 * SR]
            if len(audio) < 4 * SR:
                audio = np.pad(audio, (0, 4 * SR - len(audio)))
            originals.append(audio)
            mels.append(get_mel(audio))
        vocoded = sample_vocoder(mels, args.diffwave_checkpoint, params, args.device, use_fast, seed=9000 + cls)
        orig_probs = ast_predict_paths(input_paths, best_checkpoint, args.device, args.batch_size)
        voc_probs = ast_predict_arrays(vocoded, best_checkpoint, args.device, args.batch_size)
        class_dir = vocoder_root / f"class{cls}"
        class_dir.mkdir(parents=True, exist_ok=True)
        agrees, orig_correct, voc_correct, mel_l1 = [], [], [], []
        for i, (row, voc) in enumerate(zip(selected_rows, vocoded)):
            out_path = class_dir / row["filename"]
            sf.write(out_path, voc, SR, subtype="PCM_16")
            agrees.append(int(voc_probs[i].argmax() == orig_probs[i].argmax()))
            orig_correct.append(int(orig_probs[i].argmax() == cls))
            voc_correct.append(int(voc_probs[i].argmax() == cls))
            recon = get_mel(voc)
            target = mels[i]
            frames = min(recon.shape[-1], target.shape[-1])
            mel_l1.append(float(torch.mean(torch.abs(recon[..., :frames] - target[..., :frames])).item()))
        test_rows.append({
            "class": cls, "class_name": CLASS_NAMES[cls], "n": len(selected_rows),
            "sampler_selected_on_train": selected_sampler,
            "prediction_agreement_pct": 100 * float(np.mean(agrees)),
            "original_correct_pct": 100 * float(np.mean(orig_correct)),
            "vocoder_correct_pct": 100 * float(np.mean(voc_correct)),
            "mean_mel_l1": float(np.mean(mel_l1)), "std_mel_l1": float(np.std(mel_l1)),
            "e1_real_test_recall_pct": real_test_recall["best"][cls],
            "artifacts_only_under_diag": True,
        })
        print(test_rows[-1], flush=True)

    save_rows(args.output / "v1_ast_gain_qc.csv", ast_rows)
    save_rows(args.output / "v1_sampler_qc.csv", sampler_rows)
    save_rows(args.output / "v1_test_vocoder_qc.csv", test_rows)
    summary = {
        "e1_best_checkpoint": str(best_checkpoint),
        "e1_last_checkpoint_available": False,
        "e1_last_checkpoint_note": "The E1 training run saved best.pth only; last-epoch model weights were not retained.",
        "sampler_selected_using_train_source_diagnostics_only": selected_sampler,
        "held_out_test_outputs_are_diagnostic_only": True,
        "files": ["v1_ast_gain_qc.csv", "v1_sampler_qc.csv", "v1_test_vocoder_qc.csv"],
    }
    (args.output / "diag_metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved diagnostic outputs under {args.output}", flush=True)


if __name__ == "__main__":
    main()

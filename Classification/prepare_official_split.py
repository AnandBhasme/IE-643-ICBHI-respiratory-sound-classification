#!/usr/bin/env python3
"""Build cycle WAVs and manifests from the official ICBHI recording split."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ALIASES = {"226_1b1_Pl_sc_Meditron": "226_1b1_Pl_sc_LittC2SE"}
SAMPLE_RATE = 16000
DESIRED_LENGTH_SEC = 5


def parse_split(path: Path) -> dict[str, str]:
    split_of: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t"), start=1):
            if not row or not row[0].strip():
                continue
            if len(row) != 2:
                raise ValueError(f"Expected two tab-separated fields at {path}:{line_no}: {row!r}")
            name, split = (cell.strip() for cell in row)
            name = Path(name).stem
            name = ALIASES.get(name, name)
            split = split.lower()
            if split not in {"train", "test"}:
                raise ValueError(f"Invalid split {split!r} at {path}:{line_no}")
            if name in split_of:
                raise ValueError(f"Duplicate recording in official split: {name}")
            split_of[name] = split
    return split_of


def mono_resample(audio: np.ndarray, source_rate: int) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono or multi-channel audio, got shape {audio.shape}")
    audio = audio.astype(np.float32, copy=False)
    if source_rate != SAMPLE_RATE:
        divisor = math.gcd(source_rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // divisor, source_rate // divisor).astype(np.float32)
    return audio


def fade(audio: np.ndarray, samples: int, fade_in: bool, fade_out: bool) -> np.ndarray:
    audio = audio.copy()
    n = min(samples, len(audio))
    if n == 0:
        return audio
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if fade_in:
        audio[:n] *= ramp
    if fade_out:
        audio[-n:] *= ramp[::-1]
    return audio


def fixed_length(audio: np.ndarray) -> np.ndarray:
    target = SAMPLE_RATE * DESIRED_LENGTH_SEC
    if len(audio) > target:
        return audio[:target]
    if len(audio) == 0:
        raise ValueError("Encountered an empty respiratory cycle")
    repeats = math.ceil(target / len(audio))
    padded = np.tile(audio, repeats)[:target]
    # Match the classifier loader's fade-out after repeat padding.
    return fade(padded, SAMPLE_RATE // 16, fade_in=False, fade_out=True)


def write_manifest(path: Path, rows: list[dict]) -> None:
    fields = ["filename", "recording", "patient", "device", "location", "label", "duration", "split"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(data_dir: Path, split_file: Path, output_root: Path) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Official split file not found: {split_file}")

    actual_recordings = {p.stem for p in data_dir.glob("*.wav") if p.with_suffix(".txt").is_file()}
    split_of = parse_split(split_file)
    if set(split_of) != actual_recordings:
        missing_audio = sorted(set(split_of) - actual_recordings)
        absent_split = sorted(actual_recordings - set(split_of))
        raise ValueError(f"Recording coverage mismatch; split-only={missing_audio}, data-only={absent_split}")
    if Counter(split_of.values()) != Counter({"train": 539, "test": 381}):
        raise ValueError(f"Expected 539 train and 381 test recordings, got {Counter(split_of.values())}")

    # Refuse to merge into a prior run. The caller must choose a clean output root.
    for split in ("train", "test"):
        directory = "training" if split == "train" else "test"
        existing = output_root / directory / "real"
        if existing.exists() and any(existing.iterdir()):
            raise FileExistsError(f"Output directory is not empty: {existing}")

    manifests: dict[str, list[dict]] = {"train": [], "test": []}
    for recording in sorted(actual_recordings):
        split = split_of[recording]
        audio, source_rate = sf.read(data_dir / f"{recording}.wav", dtype="float32", always_2d=False)
        audio = mono_resample(audio, source_rate)
        audio = fade(audio, SAMPLE_RATE // 16, fade_in=True, fade_out=True)
        patient = recording.split("_", 1)[0]
        tokens = recording.split("_")
        location = tokens[2]
        device = tokens[-1]
        target_dir = output_root / ("training" if split == "train" else "test") / "real"
        target_dir.mkdir(parents=True, exist_ok=True)

        with (data_dir / f"{recording}.txt").open(encoding="utf-8") as f:
            for event_idx, line in enumerate(f):
                values = line.split()
                if len(values) != 4:
                    raise ValueError(f"Expected four annotation fields in {recording}.txt line {event_idx + 1}")
                start, end = float(values[0]), float(values[1])
                crackle, wheeze = int(values[2]), int(values[3])
                if crackle not in (0, 1) or wheeze not in (0, 1) or end <= start:
                    raise ValueError(f"Invalid annotation in {recording}.txt line {event_idx + 1}")
                start_sample = min(int(start * SAMPLE_RATE), len(audio))
                end_sample = min(int(end * SAMPLE_RATE), len(audio))
                cycle = audio[start_sample:end_sample]
                duration = len(cycle) / SAMPLE_RATE
                label = crackle + 2 * wheeze
                filename = f"{recording}_event_{event_idx}_label_{label}.wav"
                sf.write(target_dir / filename, fixed_length(cycle), SAMPLE_RATE, subtype="PCM_16")
                manifests[split].append({
                    "filename": filename,
                    "recording": recording,
                    "patient": patient,
                    "device": device,
                    "location": location,
                    "label": label,
                    "duration": f"{duration:.8f}",
                    "split": split,
                })

    for split, rows in manifests.items():
        write_manifest(output_root / "splits" / f"{split}_manifest.csv", rows)
    aliases = output_root / "splits" / "recording_aliases.csv"
    aliases.parent.mkdir(parents=True, exist_ok=True)
    with aliases.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["official_name", "dataset_name"])
        writer.writerows(ALIASES.items())
    print(f"Created {len(manifests['train'])} train and {len(manifests['test'])} test cycle WAVs under {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    build(args.data_dir, args.split_file, args.output_root)

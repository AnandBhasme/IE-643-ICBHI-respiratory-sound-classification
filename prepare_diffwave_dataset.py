#!/usr/bin/env python3
"""Create class-organized four-second DiffWave audio from train manifest only."""

import argparse
import csv
from pathlib import Path

import numpy as np
import soundfile as sf


SAMPLE_RATE = 16000
TARGET_SAMPLES = 4 * SAMPLE_RATE


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fixed_four_seconds(audio: np.ndarray) -> np.ndarray:
    if audio.ndim != 1:
        raise ValueError(f"Expected mono cycle audio, got {audio.shape}")
    if len(audio) == 0:
        raise ValueError("Empty audio cycle in train manifest")
    if len(audio) >= TARGET_SAMPLES:
        return audio[:TARGET_SAMPLES]
    repeats = (TARGET_SAMPLES + len(audio) - 1) // len(audio)
    return np.tile(audio, repeats)[:TARGET_SAMPLES]


def main(root: Path, output: Path) -> None:
    split_dir = root / "splits"
    train_rows = read_rows(split_dir / "train_manifest.csv")
    test_rows = read_rows(split_dir / "test_manifest.csv")
    train_names = {row["filename"] for row in train_rows}
    test_names = {row["filename"] for row in test_rows}
    if train_names & test_names:
        raise AssertionError("Train/test cycle filenames overlap")

    audio_root = output / "wav_4secs_16000"
    if audio_root.exists() and any(audio_root.rglob("*.wav")):
        raise FileExistsError(f"Refusing to merge into an existing DiffWave dataset: {audio_root}")

    manifest_path = output / "train_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["filename", "source_file", "class", "sample_rate", "split"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in train_rows:
            name = row["filename"]
            label = int(row["label"])
            if name not in train_names or name in test_names:
                raise AssertionError(f"Non-training source in DiffWave input: {name}")
            source = root / "training" / "real" / name
            audio, sr = sf.read(source, dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                raise ValueError(f"Unexpected sample rate {sr} in {source}")
            audio = fixed_four_seconds(audio)
            dest_dir = audio_root / f"label{label}" / "train"
            dest_dir.mkdir(parents=True, exist_ok=True)
            sf.write(dest_dir / name, audio, SAMPLE_RATE, subtype="PCM_16")
            writer.writerow({
                "filename": name,
                "source_file": str(source),
                "class": label,
                "sample_rate": SAMPLE_RATE,
                "split": "train",
            })

    written = list(audio_root.rglob("*.wav"))
    if len(written) != len(train_names):
        raise AssertionError(f"Expected {len(train_names)} DiffWave train files, wrote {len(written)}")
    print(f"Created {len(written)} training-only four-second clips at {audio_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    options = parser.parse_args()
    main(options.data_root, options.output_root)

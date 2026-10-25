#!/usr/bin/env python3
"""Create a training-only real+synthetic view for a fill-to-N run."""

import argparse
import csv
import random
from collections import Counter
from pathlib import Path


def read_train_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("Train manifest contains a non-train row")
    return rows


def prepare(data_root: Path, synth_root: Path, n_per_class: int,
            fill_classes: list[int], seed: int) -> Path:
    manifest_path = data_root / "splits" / "train_manifest.csv"
    real_root = data_root / "training" / "real"
    mixed_root = data_root / "training" / "real_with_gen"
    if mixed_root.exists():
        raise FileExistsError(f"Refusing to merge into existing mixed data: {mixed_root}")
    if not real_root.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError("Official training WAVs or train manifest are missing")

    real_rows = read_train_manifest(manifest_path)
    real_counts = Counter(int(row["label"]) for row in real_rows)
    rng = random.Random(seed)
    selected_synth = {}
    for cls in fill_classes:
        need = max(0, n_per_class - real_counts[cls])
        pool = sorted((synth_root / f"class{cls}").rglob("*.wav"))
        rng.shuffle(pool)
        if len(pool) < need:
            raise ValueError(f"Class {cls} needs {need} synthetic clips, found {len(pool)}")
        selected_synth[cls] = pool[:need]

    for cls in range(4):
        (mixed_root / f"label{cls}").mkdir(parents=True, exist_ok=False)

    used_rows = []
    for row in real_rows:
        cls = int(row["label"])
        source = real_root / row["filename"]
        if not source.is_file():
            raise FileNotFoundError(source)
        target = mixed_root / f"label{cls}" / source.name
        target.symlink_to(source.resolve())
        used_rows.append({"filename": source.name, "path": str(source.resolve()),
                          "label": cls, "is_synth": 0, "source_spec": ""})

    synth_rows = []
    for cls, paths in selected_synth.items():
        for source in paths:
            # The repository parser reads the class from the second-last token;
            # DiffWave retains `_label_<class>` and appends `_index=<sample>`.
            stem_tokens = source.stem.split("_")
            if "index" not in source.stem or len(stem_tokens) < 3 or int(stem_tokens[-2]) != cls:
                raise ValueError(f"Generated filename does not encode class {cls}: {source.name}")
            target = mixed_root / f"label{cls}" / source.name
            target.symlink_to(source.resolve())
            row = {"filename": source.name, "path": str(source.resolve()),
                   "label": cls, "is_synth": 1, "source_spec": source.stem.rsplit("_index=", 1)[0]}
            synth_rows.append(row)
            used_rows.append(row)

    counts = Counter(int(row["label"]) for row in used_rows)
    fields = ["filename", "path", "label", "is_synth", "source_spec"]
    output_manifest = data_root / "splits" / f"mixed_N{n_per_class}_seed{seed}.csv"
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(used_rows)
    synth_manifest = data_root / "splits" / f"synthetic_N{n_per_class}_seed{seed}.csv"
    with synth_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(synth_rows)

    print(f"mixed_root={mixed_root}")
    print(f"real_class_counts={dict(sorted(real_counts.items()))}")
    print(f"mixed_class_counts={dict(sorted(counts.items()))}")
    print(f"synthetic_counts={dict(sorted(Counter(row['label'] for row in synth_rows).items()))}")
    print(f"mixed_manifest={output_manifest}")
    print(f"synthetic_manifest={synth_manifest}")
    return mixed_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--synth-root", type=Path, required=True)
    parser.add_argument("--N", type=int, required=True)
    parser.add_argument("--fill-classes", default="2,3")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    classes = [int(value) for value in args.fill_classes.split(",") if value]
    prepare(args.data_root, args.synth_root, args.N, classes, args.seed)

#!/usr/bin/env python3
"""Verify cycle manifests and file layout against the official ICBHI split."""

import argparse
import csv
from collections import Counter
from pathlib import Path


EXPECTED = {
    "train": {"cycles": 4142, "recordings": 539, "classes": {0: 2063, 1: 1215, 2: 501, 3: 363}},
    "test": {"cycles": 2756, "recordings": 381, "classes": {0: 1579, 1: 649, 2: 385, 3: 143}},
}
EXPECTED_SHARED_PATIENTS = {"156", "218"}
ALIASES = {"226_1b1_Pl_sc_Meditron": "226_1b1_Pl_sc_LittC2SE"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    required = {"filename", "recording", "patient", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise AssertionError(f"Missing rows or required manifest columns in {path}")
    return rows


def official_recording_splits(path: Path) -> dict[str, str]:
    result = {}
    with path.open(encoding="utf-8") as f:
        for line_no, row in enumerate(csv.reader(f, delimiter="\t"), start=1):
            if not row or not row[0].strip():
                continue
            if len(row) != 2:
                raise AssertionError(f"Malformed official split line {line_no}: {row!r}")
            name = Path(row[0].strip()).stem
            name = ALIASES.get(name, name)
            split = row[1].strip().lower()
            if name in result:
                raise AssertionError(f"Duplicate official recording: {name}")
            result[name] = split
    return result


def main(root: Path, split_file: Path) -> None:
    manifests = {split: read_manifest(root / "splits" / f"{split}_manifest.csv") for split in EXPECTED}
    names: dict[str, set[str]] = {}
    patients: dict[str, set[str]] = {}
    all_files: dict[str, set[str]] = {}
    actual_recording_splits: dict[str, str] = {}
    for split, rows in manifests.items():
        spec = EXPECTED[split]
        labels = Counter(int(row["label"]) for row in rows)
        recordings = {row["recording"] for row in rows}
        for recording in recordings:
            if recording in actual_recording_splits:
                raise AssertionError(f"Recording occurs in both manifests: {recording}")
            actual_recording_splits[recording] = split
        names[split] = {row["filename"] for row in rows}
        patients[split] = {row["patient"] for row in rows}
        directory = "training" if split == "train" else "test"
        all_files[split] = {str(p) for p in (root / directory / "real").glob("*.wav")}
        assert len(rows) == spec["cycles"], f"{split}: expected {spec['cycles']} cycles, got {len(rows)}"
        assert len(recordings) == spec["recordings"], f"{split}: expected {spec['recordings']} recordings, got {len(recordings)}"
        assert labels == Counter(spec["classes"]), f"{split}: class counts {dict(labels)} != {spec['classes']}"
        assert len(names[split]) == len(rows), f"{split}: duplicate cycle filenames in manifest"
        expected_paths = {str(root / directory / "real" / name) for name in names[split]}
        assert all_files[split] == expected_paths, f"{split}: manifest and WAV directory differ"
        assert all(row["split"] == split for row in rows), f"{split}: incorrect split column"
        print(f"{split}: {len(rows)} cycles; {len(recordings)} recordings; classes={dict(sorted(labels.items()))}")

    overlap_names = names["train"] & names["test"]
    assert not overlap_names, f"Cycle filename overlap: {sorted(overlap_names)[:10]}"
    overlap_patients = patients["train"] & patients["test"]
    assert overlap_patients == EXPECTED_SHARED_PATIENTS, f"Expected shared patients {EXPECTED_SHARED_PATIENTS}, got {overlap_patients}"
    expected_recording_splits = official_recording_splits(split_file)
    assert actual_recording_splits == expected_recording_splits, "Manifest recording assignments differ from official split file"
    print(f"recordings: train=539 test=381")
    print(f"cycle filename overlap: {len(overlap_names)}")
    print(f"patients in both official recording-level splits: {sorted(overlap_patients, key=int)}")
    print("OFFICIAL SPLIT VERIFICATION PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="data root containing training/, test/, and splits/")
    parser.add_argument("--split-file", type=Path, required=True, help="official ICBHI_challenge_train_test.txt")
    args = parser.parse_args()
    main(args.root, args.split_file)

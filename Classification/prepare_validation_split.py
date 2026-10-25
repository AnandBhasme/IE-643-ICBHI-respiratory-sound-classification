#!/usr/bin/env python3
"""Create the deterministic, patient-grouped validation subset from official train."""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    train_path = args.split_dir / "train_manifest.csv"
    test_path = args.split_dir / "test_manifest.csv"
    with train_path.open(newline="", encoding="utf-8") as f:
        train = list(csv.DictReader(f))
    with test_path.open(newline="", encoding="utf-8") as f:
        test = list(csv.DictReader(f))
    patients = sorted({r["patient"] for r in train} - {"156", "218"}, key=int)
    candidate_size = round(0.15 * len(patients))
    train_class_frac = {
        c: sum(int(r["label"]) == c for r in train) / len(train) for c in range(4)
    }
    rng = random.Random(0)
    best, best_cost = None, float("inf")
    for _ in range(5000):
        candidate = set(rng.sample(patients, k=candidate_size))
        val = [r for r in train if r["patient"] in candidate]
        class_counts = {c: sum(int(r["label"]) == c for r in val) for c in range(4)}
        if class_counts[3] < 30:
            continue
        class_frac = {c: class_counts[c] / len(val) for c in range(4)}
        cost = 10 * abs(len(val) / len(train) - 0.12) + sum(
            abs(class_frac[c] - train_class_frac[c]) for c in range(4)
        )
        if cost < best_cost:
            best, best_cost = (candidate, val, class_counts), cost
    if best is None:
        raise RuntimeError("No candidate has at least 30 both-class cycles")
    val_patients, val, val_counts = best
    train_minus_val = [r for r in train if r["patient"] not in val_patients]
    test_names = {r["filename"] for r in test}
    assert not ({r["filename"] for r in val} & test_names)
    assert not ({r["patient"] for r in val} & {r["patient"] for r in train_minus_val})
    assert all(any(r["patient"] == p for r in train_minus_val) for p in ("156", "218"))
    assert len(val) + len(train_minus_val) == len(train)

    (args.split_dir / "val_patients.txt").write_text(
        "\n".join(sorted(val_patients, key=int)) + "\n", encoding="utf-8"
    )
    for name, rows in (("val_manifest.csv", val), ("train_minus_val_manifest.csv", train_minus_val)):
        with (args.split_dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=train[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(f"train={len(train)} val={len(val)} ({len(val)/len(train):.4%}) train_minus_val={len(train_minus_val)}")
    print(f"validation_patients={len(val_patients)} candidate_patients={len(patients)} candidate_size={candidate_size} cost={best_cost:.8f}")
    print("train_counts=" + str({c: sum(int(r["label"]) == c for r in train) for c in range(4)}))
    print("val_counts=" + str(val_counts))
    print("val_patients=" + ",".join(sorted(val_patients, key=int)))


if __name__ == "__main__":
    main()

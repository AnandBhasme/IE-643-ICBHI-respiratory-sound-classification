#!/usr/bin/env python3
"""Evaluate pre-registered equal-weight ensembles whose full membership is ready."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
AST_TO_MANIFEST = None


def load_probabilities(run_name: str) -> np.ndarray:
    path = ROOT / "artifacts/runs" / run_name / "probs_test_last.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing pre-registered ensemble member output: {path}")
    probs = np.load(path)
    if probs.shape != (2756, 4):
        raise ValueError(f"Expected (2756, 4) probabilities at {path}, got {probs.shape}")
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError(f"Invalid probability rows in {path}")
    # The AST loader enumerates sorted WAV paths; Petmezas' loader preserves
    # test_manifest.csv order. Align both explicitly before averaging.
    if run_name.startswith("icbhi_ast_"):
        probs = probs[AST_TO_MANIFEST]
    return probs.astype(np.float64, copy=False)


def metrics(labels: np.ndarray, probs: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    pred = probs.argmax(axis=1)
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    abnormal_total = int(cm[1:, :].sum())
    normal_total = int(cm[0, :].sum())
    official_se = float(cm[1, 1] + cm[2, 2] + cm[3, 3]) / abnormal_total
    official_sp = float(cm[0, 0]) / normal_total
    accuracy = float(np.trace(cm)) / int(cm.sum())
    # For four-class single-label micro one-vs-rest metrics.
    micro_sp = (2.0 + accuracy) / 3.0
    metrics_row = {
        "official_se_pct": 100.0 * official_se,
        "official_sp_pct": 100.0 * official_sp,
        "official_score_pct": 50.0 * (official_se + official_sp),
        "micro_se_pct": 100.0 * accuracy,
        "micro_sp_pct": 100.0 * micro_sp,
        "micro_score_pct": 50.0 * (accuracy + micro_sp),
        "accuracy_pct": 100.0 * accuracy,
    }
    return metrics_row, cm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = ROOT / "artifacts/data/official_icbhi/splits/test_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    global AST_TO_MANIFEST
    names = [row["filename"] for row in manifest]
    labels = np.asarray([int(row["label"]) for row in manifest], dtype=np.int64)
    if len(manifest) != 2756 or len(set(names)) != len(names):
        raise ValueError(f"Unexpected official test manifest: {len(manifest)} rows")
    ast_names = sorted(p.name for p in (ROOT / "artifacts/data/official_icbhi/test/real").glob("*.wav"))
    if set(names) != set(ast_names):
        raise ValueError("Official test manifest and AST test directory contain different file sets")
    ast_index = {name: index for index, name in enumerate(ast_names)}
    AST_TO_MANIFEST = np.asarray([ast_index[name] for name in names], dtype=np.int64)

    # These sets are fixed in PREREG.md and have all member probabilities now.
    sets = {
        "Ens-C0": [f"icbhi_ast_ce_wp3_C0_seed{s}_bf16" for s in (1, 2, 3)],
        "Ens-C1": [f"icbhi_ast_ce_wp3_C1_seed{s}_bf16" for s in (1, 2, 3)],
        "Ens-MP": [
            *[f"icbhi_ast_ce_wp3_C1_seed{s}_bf16" for s in (1, 2, 3)],
            *[f"petmezas_lean_seed{s}" for s in (1, 2, 3)],
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ensemble_name, members in sets.items():
        probs = np.mean([load_probabilities(name) for name in members], axis=0)
        row, cm = metrics(labels, probs)
        row.update({
            "ensemble": ensemble_name,
            "members": ";".join(members),
            "n_members": len(members),
            "selection_status": "preliminary; not used for demo selection",
            "tta": "not computed; included in final WP6 pass",
        })
        rows.append(row)
        np.save(args.output_dir / f"{ensemble_name}_probs_last.npy", probs.astype(np.float32))
        np.savetxt(args.output_dir / f"{ensemble_name}_confusion_last.csv", cm, fmt="%d", delimiter=",")
        print(row, flush=True)

    output_csv = args.output_dir / "wp6_ready_ensembles_preliminary.csv"
    fields = list(rows[0])
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved preliminary pre-registered ensemble metrics: {output_csv}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired test-cycle bootstrap for WP6 ensemble sets whose members are ready.

This reports uncertainty only; it does not select a model, tune weights, or
change the preregistered equal-weight probabilities.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]


def load_labels_and_order() -> tuple[np.ndarray, np.ndarray]:
    manifest_path = ROOT / "artifacts/data/official_icbhi/splits/test_manifest.csv"
    with manifest_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2756:
        raise ValueError(f"Expected 2756 official test rows, found {len(rows)}")
    names = [row["filename"] for row in rows]
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    ast_names = sorted(
        path.name
        for path in (ROOT / "artifacts/data/official_icbhi/test/real").glob("*.wav")
    )
    if set(names) != set(ast_names) or len(set(names)) != len(names):
        raise ValueError("Official test manifest and AST WAV set do not match")
    ast_pos = {name: index for index, name in enumerate(ast_names)}
    ast_to_manifest = np.asarray([ast_pos[name] for name in names], dtype=np.int64)
    return labels, ast_to_manifest


def load_probs(path: Path, ast_to_manifest: np.ndarray | None = None) -> np.ndarray:
    probs = np.load(path)
    if probs.shape != (2756, 4):
        raise ValueError(f"Expected (2756, 4) probabilities at {path}, got {probs.shape}")
    if ast_to_manifest is not None:
        probs = probs[ast_to_manifest]
    if not np.isfinite(probs).all() or not np.allclose(
        probs.sum(axis=1), 1.0, atol=1e-4
    ):
        raise ValueError(f"Invalid probability rows in {path}")
    return probs.astype(np.float64, copy=False)


def metric_vector(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    cm = np.zeros((4, 4), dtype=np.int64)
    np.add.at(cm, (labels, pred), 1)
    n = int(cm.sum())
    abnormal = int(cm[1:, :].sum())
    normal = int(cm[0, :].sum())
    official_se = float(cm[1, 1] + cm[2, 2] + cm[3, 3]) / abnormal
    official_sp = float(cm[0, 0]) / normal
    accuracy = float(np.trace(cm)) / n
    micro_sp = (2.0 + accuracy) / 3.0
    return {
        "official_se": official_se,
        "official_sp": official_sp,
        "official_score": (official_se + official_sp) / 2.0,
        "micro_se": accuracy,
        "micro_sp": micro_sp,
        "micro_score": (accuracy + micro_sp) / 2.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.resamples < 1000:
        raise ValueError("The registered WP6 bootstrap requires at least 1000 resamples")

    labels, ast_to_manifest = load_labels_and_order()
    input_dir = args.input_dir
    ensembles = {
        name: load_probs(input_dir / f"{name}_probs_last.npy")
        for name in ("Ens-C0", "Ens-C1", "Ens-MP")
    }
    petmezas = np.mean(
        [
            load_probs(
                ROOT / "artifacts/runs" / f"petmezas_lean_seed{seed}" / "probs_test_last.npy"
            )
            for seed in (1, 2, 3)
        ],
        axis=0,
    )
    # Fixed comparisons: matched three-member control, and the three-seed
    # Petmezas reproduction. Ens-MP is also compared as preregistered.
    comparisons = [
        ("Ens-C1", "Ens-C0", ensembles["Ens-C1"], ensembles["Ens-C0"]),
        ("Ens-MP", "Ens-C0", ensembles["Ens-MP"], ensembles["Ens-C0"]),
        ("Ens-C1", "Petmezas-3seed", ensembles["Ens-C1"], petmezas),
        ("Ens-MP", "Petmezas-3seed", ensembles["Ens-MP"], petmezas),
    ]
    metric_names = (
        "official_se",
        "official_sp",
        "official_score",
        "micro_se",
        "micro_sp",
        "micro_score",
    )
    rng = np.random.default_rng(args.seed)
    samples: dict[tuple[str, str, str], np.ndarray] = {}
    for left_name, right_name, left_probs, right_probs in comparisons:
        full_left = metric_vector(labels, left_probs)
        full_right = metric_vector(labels, right_probs)
        deltas = {name: np.empty(args.resamples, dtype=np.float64) for name in metric_names}
        for iteration in range(args.resamples):
            idx = rng.integers(0, len(labels), size=len(labels))
            m_left = metric_vector(labels[idx], left_probs[idx])
            m_right = metric_vector(labels[idx], right_probs[idx])
            for name in metric_names:
                deltas[name][iteration] = m_left[name] - m_right[name]
        for name in metric_names:
            samples[(left_name, right_name, name)] = deltas[name]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for left_name, right_name, left_probs, right_probs in comparisons:
        full_delta = metric_vector(labels, left_probs)
        right_metrics = metric_vector(labels, right_probs)
        for metric_name in metric_names:
            distribution = samples[(left_name, right_name, metric_name)]
            rows.append(
                {
                    "comparison": f"{left_name} minus {right_name}",
                    "metric": metric_name,
                    "point_difference_percentage_points": 100.0
                    * (full_delta[metric_name] - right_metrics[metric_name]),
                    "ci95_low_percentage_points": 100.0
                    * float(np.quantile(distribution, 0.025)),
                    "ci95_high_percentage_points": 100.0
                    * float(np.quantile(distribution, 0.975)),
                    "resamples": args.resamples,
                    "bootstrap_seed": args.seed,
                    "method": "paired official-test-cycle percentile bootstrap",
                    "selection_status": "uncertainty only; no model/weight selection",
                }
            )
    output = args.output_dir / "wp6_preliminary_paired_bootstrap.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} paired bootstrap intervals to {output}", flush=True)
    for row in rows:
        if row["metric"] in ("official_se", "official_sp", "official_score"):
            print(row, flush=True)


if __name__ == "__main__":
    main()

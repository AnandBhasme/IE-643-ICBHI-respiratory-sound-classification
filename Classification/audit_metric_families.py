#!/usr/bin/env python3
"""Recompute official ICBHI and Petmezas-style micro metrics from confusion matrices."""
from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np

from util.metrics import evaluate_confusion


RESULT_FIELDS = [
    "record_type", "reference_basis", "run_id", "method", "data", "N",
    "synth_version", "seed", "job_id", "best_epoch", "best_Se", "best_Sp",
    "best_Score", "best_micro_Se", "best_micro_Sp", "best_micro_Score",
    "best_micro_Acc", "last_epoch", "last_Se", "last_Sp", "last_Score",
    "last_micro_Se", "last_micro_Sp", "last_micro_Score", "last_micro_Acc",
    "last_accuracy", "last_recall_normal", "last_recall_crackle",
    "last_recall_wheeze", "last_recall_both", "last_se_binary", "wall_time",
    "epochs_logged", "confusion_best", "confusion_last",
]


def pct(v: float) -> float:
    return 100.0 * float(v)


def metrics_from_accuracy(accuracy: float, n_classes: int = 4) -> dict[str, float]:
    """Micro-average identities for a single-label, K-class task."""
    se = float(accuracy)
    sp = (n_classes - 2 + se) / (n_classes - 1)
    acc = (n_classes - 2 + 2.0 * se) / n_classes
    return {"Se": se, "Sp": sp, "Score": (se + sp) / 2.0, "Acc": acc}


def cm_from_row(row: dict[str, Any]) -> np.ndarray:
    k = 4
    return np.asarray([[int(float(row[f"cm_{i}{j}"])) for j in range(k)] for i in range(k)], dtype=np.int64)


def format_pct(v: Any) -> str:
    if v is None or v == "":
        return ""
    return str(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def add_run_metrics(source_row: dict[str, str], run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = source_row["run_id"]
    metrics_paths = list(run_root.rglob(run_id + "/metrics.csv"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics.csv found for run_id={run_id}")
    metrics_path = metrics_paths[0]
    with metrics_path.open(newline="", encoding="utf-8") as f:
        epoch_rows = list(csv.DictReader(f))
    if not epoch_rows:
        raise ValueError(f"Empty metrics file: {metrics_path}")
    if not all(f"cm_{i}{j}" in epoch_rows[-1] for i in range(4) for j in range(4)):
        raise ValueError(f"Missing 4x4 confusion columns in {metrics_path}")

    best = max(epoch_rows, key=lambda row: float(row["Score"]))
    last = epoch_rows[-1]
    best_cm, last_cm = cm_from_row(best), cm_from_row(last)
    best_metrics, last_metrics = evaluate_confusion(best_cm), evaluate_confusion(last_cm)
    support = np.asarray([1579, 649, 385, 143])
    if not np.array_equal(best_cm.sum(axis=1), support) or not np.array_equal(last_cm.sum(axis=1), support):
        raise AssertionError(f"Unexpected official-test support for {run_id}: {best_cm.sum(1).tolist()}")

    # Confirm the legacy official fields agree with recomputation from the exact matrix.
    for row, calculated, label in ((best, best_metrics, "best"), (last, last_metrics, "last")):
        for key in ("Se", "Sp", "Score"):
            observed = float(row[key])
            expected = pct(calculated[key])
            if not math.isclose(observed, expected, abs_tol=2e-4):
                raise AssertionError(f"{run_id} {label} {key}: logged {observed}, confusion-matrix value {expected}")

    result = dict(source_row)
    result.update({
        "record_type": "run",
        "reference_basis": "",
        "best_epoch": best["epoch"],
        "best_Se": pct(best_metrics["Se"]),
        "best_Sp": pct(best_metrics["Sp"]),
        "best_Score": pct(best_metrics["Score"]),
        "best_micro_Se": pct(best_metrics["Se_micro"]),
        "best_micro_Sp": pct(best_metrics["Sp_micro"]),
        "best_micro_Score": pct(best_metrics["Score_micro"]),
        "best_micro_Acc": pct(best_metrics["accuracy_micro"]),
        "last_epoch": last["epoch"],
        "last_Se": pct(last_metrics["Se"]),
        "last_Sp": pct(last_metrics["Sp"]),
        "last_Score": pct(last_metrics["Score"]),
        "last_micro_Se": pct(last_metrics["Se_micro"]),
        "last_micro_Sp": pct(last_metrics["Sp_micro"]),
        "last_micro_Score": pct(last_metrics["Score_micro"]),
        "last_micro_Acc": pct(last_metrics["accuracy_micro"]),
        "last_accuracy": pct(last_metrics["accuracy"]),
        "last_recall_normal": pct(last_metrics["recall"][0]),
        "last_recall_crackle": pct(last_metrics["recall"][1]),
        "last_recall_wheeze": pct(last_metrics["recall"][2]),
        "last_recall_both": pct(last_metrics["recall"][3]),
        "last_se_binary": pct(last_metrics["se_binary"]),
        "epochs_logged": len(epoch_rows),
        "confusion_best": json.dumps(best_cm.tolist(), separators=(",", ":")),
        "confusion_last": json.dumps(last_cm.tolist(), separators=(",", ":")),
    })
    return result, {"run_id": run_id, "epochs": len(epoch_rows), "best": best_metrics, "last": last_metrics}


def reference_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def blank(identifier: str, basis: str) -> dict[str, Any]:
        row = {key: "" for key in RESULT_FIELDS}
        row["record_type"] = "reference"
        row["run_id"] = identifier
        row["method"] = identifier
        row["reference_basis"] = basis
        return row

    for identifier, basis, accuracy, published_score in (
        ("REF_PETMEZAS_OFFICIAL_PUBLISHED", "Published official split; values are Petmezas-style micro averages", 0.4737, 64.92),
        ("REF_PETMEZAS_10FOLD_PUBLISHED", "Published 10-fold CV; values are Petmezas-style micro averages", 0.5278, 68.52),
    ):
        row = blank(identifier, basis)
        m = metrics_from_accuracy(accuracy)
        row.update({f"best_micro_{k}": pct(v) for k, v in m.items() if k != "Acc"})
        row["best_micro_Acc"] = pct(m["Acc"])
        row["best_micro_Score"] = published_score
        row.update({f"last_micro_{k}": pct(v) for k, v in m.items() if k != "Acc"})
        row["last_micro_Acc"] = pct(m["Acc"])
        row["last_micro_Score"] = published_score
        rows.append(row)

    petmezas_cm = np.asarray([
        [2688, 714, 197, 43],
        [1169, 575, 74, 46],
        [570, 169, 95, 52],
        [277, 117, 54, 58],
    ], dtype=np.int64)
    m = evaluate_confusion(petmezas_cm)
    row = blank("REF_PETMEZAS_10FOLD_FIG5_MATRIX", "Figure 5 pooled 10-fold confusion matrix; re-expressed with official ICBHI and micro definitions")
    row["best_Se"], row["best_Sp"], row["best_Score"] = (pct(m[k]) for k in ("Se", "Sp", "Score"))
    row["best_micro_Se"], row["best_micro_Sp"], row["best_micro_Score"], row["best_micro_Acc"] = (pct(m[k]) for k in ("Se_micro", "Sp_micro", "Score_micro", "accuracy_micro"))
    row["confusion_best"] = json.dumps(petmezas_cm.tolist(), separators=(",", ":"))
    rows.append(row)

    supports = [1579, 649, 385, 143]
    always_normal = np.zeros((4, 4), dtype=np.int64)
    always_normal[:, 0] = supports
    m = evaluate_confusion(always_normal)
    row = blank("REF_ALWAYS_NORMAL_OFFICIAL_TEST", "Always predict normal; official test supports [1579,649,385,143]")
    row["best_Se"], row["best_Sp"], row["best_Score"] = (pct(m[k]) for k in ("Se", "Sp", "Score"))
    row["best_micro_Se"], row["best_micro_Sp"], row["best_micro_Score"], row["best_micro_Acc"] = (pct(m[k]) for k in ("Se_micro", "Sp_micro", "Score_micro", "accuracy_micro"))
    row["confusion_best"] = json.dumps(always_normal.tolist(), separators=(",", ":"))
    rows.append(row)
    return rows


def write_audit_report(path: Path, audit: list[dict[str, Any]], references: list[dict[str, Any]]) -> None:
    lines = [
        "# Dual-metric audit",
        "",
        "All run metrics below were recomputed from the saved 4×4 truth-row/prediction-column confusion matrices. Official ICBHI metrics are primary. Petmezas-style micro metrics are included only for comparison with the paper.",
        "",
        "## Reference checks",
        "",
        "| Reference | Official Se / Sp / Score (%) | Micro Se / Sp / Score / Acc (%) | Basis |",
        "|---|---:|---:|---|",
    ]
    for row in references:
        official = " / ".join(format_pct(row.get(f"best_{k}", "")) for k in ("Se", "Sp", "Score"))
        micro = " / ".join(format_pct(row.get(f"best_micro_{k}", "")) for k in ("Se", "Sp", "Score", "Acc"))
        lines.append(f"| {row['run_id']} | {official} | {micro} | {row['reference_basis']} |")
    lines += [
        "",
        "For a single-label four-class task with top-1 accuracy `a`, micro Se = `a`, micro Sp = `(2+a)/3`, micro Acc = `(1+a)/2`, and micro Score = `(Se+Sp)/2`. The published official-split figures fit: Se 47.37 → Sp 82.46, Acc 73.69, Score 64.92. The published 10-fold figures fit: Se 52.78 → Sp 84.26, Acc 76.39, Score 68.52. Values round to the paper's reported precision.",
        "",
        "## Per-run metrics (percent)",
        "",
        "| Run | Protocol note | Epochs | Best official Se / Sp / Score | Best micro Se / Sp / Score / Acc | Last official Se / Sp / Score | Last micro Se / Sp / Score / Acc |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in audit:
        b = " / ".join(format_pct(row.get(f"best_{k}", "")) for k in ("Se", "Sp", "Score"))
        bm = " / ".join(format_pct(row.get(f"best_micro_{k}", "")) for k in ("Se", "Sp", "Score", "Acc"))
        l = " / ".join(format_pct(row.get(f"last_{k}", "")) for k in ("Se", "Sp", "Score"))
        lm = " / ".join(format_pct(row.get(f"last_micro_{k}", "")) for k in ("Se", "Sp", "Score", "Acc"))
        basis = row.get("reference_basis", "")
        lines.append(f"| {row['run_id']} | {basis} | {row['epochs_logged']} | {b} | {bm} | {l} | {lm} |")
    lines += [
        "",
        "## Interpretation note",
        "",
        "Micro Se is plain top-1 accuracy; it is not the ICBHI abnormal-class sensitivity. On the official test supports, always-normal obtains official Se/Sp/Score = 0/100/50 and micro Se/Sp/Score = 57.29/85.76/71.53 (micro accuracy 78.65). Thus micro averages are not a clinically meaningful improvement target on this imbalanced four-class set.",
        "",
        "The Figure-5 pooled 10-fold confusion matrix is a separate re-expression, not an official-split result: official Se/Sp/Score = 22.36/73.81/48.08. Its row-level pooled matrix is retained in `results.csv` for verification.",
        "",
        "Legacy best-epoch values remain selected by test Score in the original experiments; they are labelled test-selected. Under the v3.2 lean protocol, all new runs use the full official training set, have no validation split or screening, and headline fixed-budget last-epoch test metrics. The earlier validation-based Petmezas run is a pre-change pilot and is excluded from the v3.2 candidate comparison.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-results", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    args = parser.parse_args()
    with args.input_results.open(newline="", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    audited, reports = [], []
    for row in source_rows:
        result, report = add_run_metrics(row, args.run_root)
        audited.append(result)
        reports.append(report)
    refs = reference_rows()
    output_rows = audited + refs
    args.output_results.parent.mkdir(parents=True, exist_ok=True)
    with args.output_results.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    write_audit_report(args.audit_report, audited, refs)
    total_epochs = sum(item["epochs"] for item in reports)
    print(f"Audited {len(audited)} run summaries and {total_epochs} epoch rows; appended {len(refs)} reference rows.")
    print(f"Wrote {args.output_results} and {args.audit_report}")


if __name__ == "__main__":
    main()

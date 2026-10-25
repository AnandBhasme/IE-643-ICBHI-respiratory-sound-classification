#!/usr/bin/env python3
"""Collect best-by-test-Score and last-epoch rows from run metrics CSVs."""

import argparse
import csv
import json
import subprocess
from pathlib import Path

from util.metrics import evaluate_confusion

OUTPUT_FIELDS = (
    "run_id", "reference_basis", "method", "data", "N", "synth_version", "seed", "job_id",
    "best_epoch", "best_Se", "best_Sp", "best_Score",
    "best_micro_Se", "best_micro_Sp", "best_micro_Score", "best_micro_Acc", "last_epoch",
    "last_Se", "last_Sp", "last_Score", "last_accuracy",
    "last_micro_Se", "last_micro_Sp", "last_micro_Score", "last_micro_Acc",
    "last_recall_normal", "last_recall_crackle", "last_recall_wheeze",
    "last_recall_both", "last_se_binary", "wall_time", "epochs_logged",
)


def elapsed_from_slurm(job_id) -> str:
    if not job_id:
        return ""
    try:
        result = subprocess.run(
            ["sacct", "-n", "-P", "-X", "-j", str(job_id), "--format=Elapsed"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        return next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    except (OSError, subprocess.SubprocessError):
        return ""


def collect(run_root: Path, output: Path) -> None:
    rows = []
    for metrics_path in sorted(run_root.rglob("metrics.csv")):
        with metrics_path.open(newline="", encoding="utf-8") as f:
            metrics = list(csv.DictReader(f))
        if not metrics:
            continue
        required = {"Score", "Se", "Sp", *(f"cm_{i}{j}" for i in range(4) for j in range(4))}
        if not required.issubset(metrics[-1]):
            print(f"Skipping legacy nonstandard metrics file (not a v3.2 result schema): {metrics_path}")
            continue
        run_dir = metrics_path.parent
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        args_path = run_dir / "train_args.json"
        args = json.loads(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
        cfg = {**args, **config}
        # A job can time out mid-candidate and leave a syntactically valid
        # metrics.csv. Do not report that partial run as a completed seed.
        expected_epochs = int(cfg.get("epochs") or 0)
        is_preregistered_candidate = any(
            marker in run_dir.name for marker in ("wp3_C", "wp4_SC1", "wp4_SC2")
        )
        if is_preregistered_candidate and expected_epochs and len(metrics) < expected_epochs:
            print(
                f"Skipping incomplete pre-registered candidate ({len(metrics)}/{expected_epochs} epochs): {run_dir}",
                flush=True,
            )
            continue
        validation_cycles = int(cfg.get("validation_cycles") or 0)
        if validation_cycles or cfg.get("val_mode") == "holdout":
            reference_basis = "pre-v3.2 validation pilot; excluded from lean candidate comparison"
        elif "wp1_precision" in run_dir.name:
            reference_basis = "precision diagnostic; not a pre-registered candidate"
        else:
            reference_basis = ""
        best = max(metrics, key=lambda row: float(row["Score"]))
        last = metrics[-1]
        best_cm = [[int(best[f"cm_{i}{j}"]) for j in range(4)] for i in range(4)]
        last_cm = [[int(last[f"cm_{i}{j}"]) for j in range(4)] for i in range(4)]
        best_all = evaluate_confusion(best_cm)
        last_all = evaluate_confusion(last_cm)
        row = {
            "run_id": run_dir.name,
            "reference_basis": reference_basis,
            "method": cfg.get("method", "unknown"),
            "data": "real_only" if cfg.get("train_real_only", False) else "real_plus_synth",
            "N": cfg.get("N", ""),
            "synth_version": cfg.get("synth_version", ""),
            "seed": cfg.get("seed", ""),
            "job_id": cfg.get("slurm_job_id", ""),
            "best_epoch": best["epoch"],
            "best_Se": best["Se"],
            "best_Sp": best["Sp"],
            "best_Score": best["Score"],
            "best_micro_Se": 100.0 * best_all["Se_micro"],
            "best_micro_Sp": 100.0 * best_all["Sp_micro"],
            "best_micro_Score": 100.0 * best_all["Score_micro"],
            "best_micro_Acc": 100.0 * best_all["accuracy_micro"],
            "last_epoch": last["epoch"],
            "last_Se": last["Se"],
            "last_Sp": last["Sp"],
            "last_Score": last["Score"],
            "last_micro_Se": 100.0 * last_all["Se_micro"],
            "last_micro_Sp": 100.0 * last_all["Sp_micro"],
            "last_micro_Score": 100.0 * last_all["Score_micro"],
            "last_micro_Acc": 100.0 * last_all["accuracy_micro"],
            "last_accuracy": last["accuracy"],
            "last_recall_normal": last["recall_normal"],
            "last_recall_crackle": last["recall_crackle"],
            "last_recall_wheeze": last["recall_wheeze"],
            "last_recall_both": last["recall_both"],
            "last_se_binary": last["se_binary"],
            "wall_time": cfg.get("wall_time") or elapsed_from_slurm(cfg.get("slurm_job_id")),
            "epochs_logged": len(metrics),
        }
        rows.append(row)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} run summaries to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.run_root, args.output)

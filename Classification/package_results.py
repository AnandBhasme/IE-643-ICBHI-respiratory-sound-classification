#!/usr/bin/env python3
"""Package slim checkpoints and report figures from completed ICBHI runs."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "artifacts/runs"
OUT = ROOT / "artifacts"
FINAL = OUT / "final_models"
FIG = OUT / "results/figures"

RUNS_TO_PACKAGE = {
    "E1": "icbhi_ast_ce_E1_seed1_official",
    "E2": "icbhi_ast_patchmix_cl_E2_seed1_official",
    # Highest best-epoch ICBHI Score among completed synthetic runs.
    "best_synthetic_E4": "icbhi_ast_ce_mixed_E4_AFT_N1000_seed1_v1",
}
RUN_DIRS = {
    "E1": RUNS / RUNS_TO_PACKAGE["E1"],
    "E2": RUNS / RUNS_TO_PACKAGE["E2"],
    "E3": RUNS / "icbhi_ast_ce_E3_minority_N1000_seed1_v1",
    "E4": RUNS / "aft" / RUNS_TO_PACKAGE["best_synthetic_E4"],
    "E5": RUNS / "aft/icbhi_ast_patchmix_cl_mixed_E5_PMCL_AFT_N1000_seed1_v1_retry2",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def package_checkpoints():
    FINAL.mkdir(parents=True, exist_ok=True)
    for name, run_id in RUNS_TO_PACKAGE.items():
        src = RUNS / ("aft/" if name == "best_synthetic_E4" else "") / run_id / "best.pth"
        state = torch.load(src, map_location="cpu", weights_only=False)
        slim = {"model": state["model"], "classifier": state["classifier"]}
        dst = FINAL / f"{name}.pth"
        torch.save(slim, dst)
        print(f"saved {dst} ({dst.stat().st_size} bytes)", flush=True)
        del state, slim


def plot_curves():
    FIG.mkdir(parents=True, exist_ok=True)
    for label, run_dir in RUN_DIRS.items():
        path = run_dir / "metrics.csv"
        if not path.exists():
            continue
        rows = read_rows(path)
        epochs = [int(row["epoch"]) for row in rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        for field, legend in (("Se", "Sensitivity"), ("Sp", "Specificity"), ("Score", "ICBHI Score")):
            ax.plot(epochs, [float(row[field]) for row in rows], label=legend)
        ax.set(xlabel="Epoch", ylabel="Percent", title=f"{label} training curve")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"{label.lower()}_curve.png", dpi=150)
        plt.close(fig)


def plot_confusions():
    FIG.mkdir(parents=True, exist_ok=True)
    for label in ("E1", "E2", "best_synthetic_E4"):
        run_dir = RUN_DIRS["E4"] if label == "best_synthetic_E4" else RUN_DIRS[label]
        rows = read_rows(run_dir / "metrics.csv")
        best = max(rows, key=lambda row: float(row["Score"]))
        for suffix, row in (("best", best), ("last", rows[-1])):
            matrix = np.array([[int(row[f"cm_{i}{j}"]) for j in range(4)] for i in range(4)])
            fig, ax = plt.subplots(figsize=(5, 4.5))
            image = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(image, ax=ax)
            ax.set(xticks=range(4), yticks=range(4), xticklabels=("Normal", "Crackle", "Wheeze", "Both"),
                   yticklabels=("Normal", "Crackle", "Wheeze", "Both"), xlabel="Predicted", ylabel="True",
                   title=f"{label} {suffix} epoch {row['epoch']}")
            for i in range(4):
                for j in range(4):
                    ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
            fig.tight_layout()
            fig.savefig(FIG / f"{label.lower()}_confusion_{suffix}.png", dpi=150)
            plt.close(fig)


def plot_class_recalls():
    labels = ("normal", "crackle", "wheeze", "both")
    selected = ("E1", "E4")
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    for offset, name in zip((-0.18, 0.18), selected):
        rows = read_rows(RUN_DIRS[name] / "metrics.csv")
        best = max(rows, key=lambda row: float(row["Score"]))
        values = [float(best[f"recall_{label}"]) for label in labels]
        ax.bar(x + offset, values, width=0.36, label=name)
    ax.set(xticks=x, xticklabels=labels, ylabel="Recall (%)", title="Best-epoch class recall")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "class_recall_e1_e4.png", dpi=150)
    plt.close(fig)


def write_model_card():
    card = """# Packaged classifier checkpoints\n\n"""
    card += "All checkpoints are CPU-loadable PyTorch dictionaries containing only `model` and `classifier` state dictionaries (no optimizer state). Inputs are one ICBHI cycle processed to 16 kHz, 5 seconds, 128-bin Kaldi fbank using the repository normalization.\n\n"
    card += "| Checkpoint | Run | Best test Se | Best test Sp | Best Score |\n|---|---|---:|---:|---:|\n"
    labels = {
        "E1": (RUN_DIRS["E1"], "AST + CE, real only"),
        "E2": (RUN_DIRS["E2"], "AST + Patch-Mix CL, real only"),
        "best_synthetic_E4": (RUN_DIRS["E4"], "AST + AFT, real + synthetic v1 N=1000"),
    }
    for checkpoint, (run_dir, desc) in labels.items():
        row = max(read_rows(run_dir / "metrics.csv"), key=lambda r: float(r["Score"]))
        card += f"| `{checkpoint}.pth` | {desc} (`{run_dir.name}`, seed 1) | {float(row['Se']):.2f}% | {float(row['Sp']):.2f}% | {float(row['Score']):.2f}% |\n"
    card += "\nMetrics use the official test set and select the best epoch by test Score; treat them as test-selected estimates. The synthetic checkpoint is selected by the highest best-epoch Score among completed E3–E5 runs. It did not improve both Se and Sp over E1.\n"
    (FINAL / "model_card.md").write_text(card, encoding="utf-8")


if __name__ == "__main__":
    package_checkpoints()
    plot_curves()
    plot_confusions()
    plot_class_recalls()
    write_model_card()

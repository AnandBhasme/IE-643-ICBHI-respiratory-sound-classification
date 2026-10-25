#!/usr/bin/env python3
"""Build compact WP3/WP4/WP6 result figures from audited CSV artifacts."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, required=True)
    root = p.parse_args().project_root
    out = root / "artifacts/results/figures"
    out.mkdir(parents=True, exist_ok=True)
    runs = read_csv(root / "artifacts/results/results.csv")
    ensembles = read_csv(root / "artifacts/results/wp6_final/wp6_final_ensembles.csv")
    candidates = read_csv(root / "artifacts/results/wp6_final/wp3_wp4_candidate_summary.csv")

    names = [r["candidate"] for r in candidates]
    means = np.asarray([float(r["last_Score_mean"]) for r in candidates])
    stds = np.asarray([float(r["last_Score_sample_sd"]) for r in candidates])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=["#6c8ebf"]*5+["#82b366"]*2)
    ax.set_ylabel("Official ICBHI last-epoch Score (%)")
    ax.set_title("Predeclared candidate results (mean ± sample SD, available seeds)")
    ax.grid(axis="y", alpha=.25)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2, mean+.12, f"{mean:.2f}", ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "wp3_wp4_candidate_scores.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for row in runs:
        if row.get("record_type") != "run" or not row.get("last_Se"):
            continue
        ax.scatter(float(row["last_Sp"]), float(row["last_Se"]), s=22, alpha=.35, color="#6c8ebf")
    for row in ensembles:
        if row["ensemble"] == "Petmezas-3seed":
            color, marker = "#d79b00", "D"
        else:
            color, marker = "#b85450", "*"
        ax.scatter(float(row["official_sp_pct"]), float(row["official_se_pct"]), s=125,
                   color=color, marker=marker, edgecolor="black", linewidth=.4)
        ax.annotate(row["ensemble"], (float(row["official_sp_pct"]), float(row["official_se_pct"])),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Official specificity (%)"); ax.set_ylabel("Official sensitivity (%)")
    ax.set_title("Official ICBHI Se–Sp: run last epochs and fixed ensembles")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "official_se_sp_runs_ensembles.png", dpi=180); plt.close(fig)

    show = [r for r in ensembles if r["ensemble"] != "Petmezas-3seed"]
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
    for ax, row in zip(axes.flat, show):
        cm = np.loadtxt(root / "artifacts/results/wp6_final" / f"{row['ensemble']}_confusion_last.csv", delimiter=",")
        ax.imshow(cm, cmap="Blues")
        ax.set_title(row["ensemble"], fontsize=9)
        ax.set_xticks(range(4), ["N", "C", "W", "B"], fontsize=7)
        ax.set_yticks(range(4), ["N", "C", "W", "B"], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True", fontsize=7)
        threshold = cm.max() * .55
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(cm[i,j])}", ha="center", va="center",
                        color="white" if cm[i,j] > threshold else "black", fontsize=7)
    for ax in axes.flat[len(show):]: ax.axis("off")
    fig.suptitle("Predeclared equal-weight ensemble confusion matrices", y=.99)
    fig.tight_layout(); fig.savefig(out / "wp6_ensemble_confusions.png", dpi=180); plt.close(fig)
    print(f"Wrote 3 figures under {out}", flush=True)


if __name__ == "__main__":
    main()

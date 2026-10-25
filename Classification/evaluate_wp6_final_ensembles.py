#!/usr/bin/env python3
"""Evaluate all ready preregistered v3.2 equal-weight ensembles."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("IE643_PROJECT_ROOT", Path(__file__).resolve().parents[1]))


def member_names():
    c0 = [f"icbhi_ast_ce_wp3_C0_seed{s}_bf16" for s in (1, 2, 3)]
    c1 = [f"icbhi_ast_ce_wp3_C1_seed{s}_bf16" for s in (1, 2, 3)]
    c2 = [f"icbhi_ast_ce_wp3_C2_seed{s}_bf16" for s in (1, 2, 3)]
    c3 = [f"icbhi_ast_ce_wp3_C3_seed{s}_bf16" for s in (1, 2, 3)]
    c4 = [
        "icbhi_ast_patchmix_cl_wp3_C4_seed1_bf16_retry1",
        "icbhi_ast_patchmix_cl_wp3_C4_seed2_bf16_retry2",
        "icbhi_ast_patchmix_cl_wp3_C4_seed3_bf16_retry2",
    ]
    sc1 = [
        "icbhi_ast_ce_wp4_SC1_seed1_fp32",
        "icbhi_ast_ce_wp4_SC1_seed2_fp32",
        "icbhi_ast_ce_wp4_SC1_seed3_retry1_fp32",
    ]
    sc2 = [
        "icbhi_ast_ce_wp4_SC2_seed1_fp32",
        "icbhi_ast_ce_wp4_SC2_seed2_fp32",
        "icbhi_ast_ce_wp4_SC2_seed3_retry1_fp32",
    ]
    pet = [f"petmezas_lean_seed{s}" for s in (1, 2, 3)]
    return {
        "Ens-C0": c0,
        "Ens-C1": c1,
        "Ens-C4": c4,
        "Ens-MS": c1 + sc1,
        "Ens-MP": c1 + pet,
        "Ens-AST": c0 + c1 + c2 + c3 + c4,
        "Ens-ALL": c0 + c1 + c2 + c3 + c4 + pet + sc1 + sc2,
        "Petmezas-3seed": pet,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    manifest_file = ROOT / "artifacts/data/official_icbhi/splits/test_manifest.csv"
    with manifest_file.open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    if len(manifest) != 2756:
        raise ValueError(f"Expected 2756 test rows, found {len(manifest)}")
    filenames = [r["filename"] for r in manifest]
    labels = np.asarray([int(r["label"]) for r in manifest], dtype=np.int64)
    ast_names = sorted(x.name for x in (ROOT / "artifacts/data/official_icbhi/test/real").glob("*.wav"))
    if len(set(filenames)) != 2756 or set(filenames) != set(ast_names):
        raise ValueError("Test manifest and AST test directory do not match")
    ast_pos = {n: i for i, n in enumerate(ast_names)}
    ast_order = np.asarray([ast_pos[n] for n in filenames])

    def load(run):
        path = ROOT / "artifacts/runs" / run / "probs_test_last.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        x = np.load(path)
        if x.shape != (2756, 4) or not np.isfinite(x).all() or not np.allclose(x.sum(1), 1, atol=1e-4):
            raise ValueError(f"Invalid probability output: {path}")
        return x[ast_order] if run.startswith("icbhi_ast_") else x

    def scores(prob):
        pred = prob.argmax(1)
        cm = np.zeros((4, 4), dtype=np.int64)
        np.add.at(cm, (labels, pred), 1)
        abnormal, normal = cm[1:, :].sum(), cm[0, :].sum()
        acc = np.trace(cm) / cm.sum()
        se = (cm[1, 1] + cm[2, 2] + cm[3, 3]) / abnormal
        sp = cm[0, 0] / normal
        return {
            "official_se_pct": 100 * se, "official_sp_pct": 100 * sp,
            "official_score_pct": 50 * (se + sp), "micro_se_pct": 100 * acc,
            "micro_sp_pct": 100 * (2 + acc) / 3,
            "micro_score_pct": 50 * (acc + (2 + acc) / 3), "accuracy_pct": 100 * acc,
        }, cm

    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    probs_by_name = {}
    for name, members in member_names().items():
        probs = np.mean([load(m) for m in members], axis=0)
        probs_by_name[name] = probs
        metric, cm = scores(probs)
        row = {"ensemble": name, "members": ";".join(members), "n_members": len(members),
               "tta": "not computed; omitted under time-short priority", "selection_status": "predeclared equal weights"}
        row.update(metric)
        rows.append(row)
        np.save(a.output_dir / f"{name}_probs_last.npy", probs.astype(np.float32))
        np.savetxt(a.output_dir / f"{name}_confusion_last.csv", cm, fmt="%d", delimiter=",")
    with (a.output_dir / "wp6_final_ensembles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # Candidate headline table from the audited per-run result collector.
    collected_path = ROOT / "artifacts/results/results.csv"
    candidate_rows = []
    with collected_path.open(newline="", encoding="utf-8") as f:
        collected = list(csv.DictReader(f))
    groups = {name: [] for name in ("C0", "C1", "C2", "C3", "C4", "SC1", "SC2")}
    for run in collected:
        run_id = run.get("run_id", "")
        for candidate in groups:
            marker = f"wp3_{candidate}_" if candidate.startswith("C") else f"wp4_{candidate}_"
            if marker in run_id and not run.get("reference_basis"):
                groups[candidate].append(run)
    fields = ("last_Se", "last_Sp", "last_Score", "last_micro_Se", "last_micro_Sp", "last_micro_Score")
    for candidate, group in groups.items():
        values = {key: np.asarray([float(r[key]) for r in group], dtype=np.float64) for key in fields}
        row = {"candidate": candidate, "completed_seeds": len(group)}
        for key, arr in values.items():
            row[f"{key}_mean"] = float(arr.mean()) if len(arr) else ""
            row[f"{key}_sample_sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else ""
        candidate_rows.append(row)
    with (a.output_dir / "wp3_wp4_candidate_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(candidate_rows[0])); w.writeheader(); w.writerows(candidate_rows)

    rng = np.random.default_rng(20261025)
    boot_rows = []
    metric_names = ("official_se", "official_sp", "official_score", "micro_se", "micro_sp", "micro_score")
    def vector(y, prob):
        cm = np.zeros((4,4), dtype=np.int64); np.add.at(cm, (y, prob.argmax(1)), 1)
        acc = np.trace(cm) / cm.sum(); se = (cm[1,1]+cm[2,2]+cm[3,3])/cm[1:,:].sum(); sp=cm[0,0]/cm[0,:].sum()
        return np.asarray([se, sp, (se+sp)/2, acc, (2+acc)/3, (acc+(2+acc)/3)/2])
    controls = ["Ens-C0", "Petmezas-3seed"]
    for name in ("Ens-C1", "Ens-C4", "Ens-MS", "Ens-MP", "Ens-AST", "Ens-ALL"):
        for control in controls:
            point = 100 * (vector(labels, probs_by_name[name]) - vector(labels, probs_by_name[control]))
            samples = np.empty((1000, 6), dtype=np.float64)
            for i in range(1000):
                ix = rng.integers(0, len(labels), len(labels))
                samples[i] = 100 * (vector(labels[ix], probs_by_name[name][ix]) - vector(labels[ix], probs_by_name[control][ix]))
            for j, metric in enumerate(metric_names):
                boot_rows.append({"comparison": f"{name} minus {control}", "metric": metric,
                                  "point_difference_percentage_points": point[j],
                                  "ci95_low_percentage_points": np.quantile(samples[:,j], .025),
                                  "ci95_high_percentage_points": np.quantile(samples[:,j], .975),
                                  "resamples": 1000, "bootstrap_seed": 20261025,
                                  "method": "paired test-cycle percentile bootstrap",
                                  "selection_status": "uncertainty only; no tuning"})
    with (a.output_dir / "wp6_final_paired_bootstrap.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(boot_rows[0])); w.writeheader(); w.writerows(boot_rows)
    print(f"Wrote {len(rows)} ensemble rows, {len(candidate_rows)} candidate summaries, and {len(boot_rows)} paired intervals to {a.output_dir}", flush=True)
    for row in rows:
        print(row, flush=True)


if __name__ == "__main__":
    main()

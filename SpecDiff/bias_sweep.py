#!/usr/bin/env python3
"""Report-control: is T_K3's gain more than an operating-point shift?

CPU only, a few seconds. Place this file in SpecDiff/ (next to ensemble_nov.py,
prior_correct.py, test_labels.py) and run:   python bias_sweep.py --out <OUT>/bias_sweep

For every real-only model (M0, seeds 1-10) one scalar bias b is added to the NORMAL-class log-probability
(p_normal * exp(b), others unchanged, argmax). This traces M0's Score-vs-operating-point curve.
Each T_K3 (+ prior correction) model is then compared with the same-seed M0 at matched specificity and at
matched sensitivity (linear interpolation on the curve). The same is done for M3 + correction, for the raw
arms, and for the 10-model ensembles. The oracle (best b chosen on the test labels) is an upper bound for M0.
No parameter of any arm is tuned; test labels are used only to score, as in every earlier look.
"""
import argparse, csv, json, os
import numpy as np
from scipy import stats

GRID = np.round(np.arange(-4.0, 4.0001, 0.01), 4)


def score(pred, lab):
    ab = lab > 0
    se = (pred[ab] == lab[ab]).mean() * 100
    sp = (pred[~ab] == 0).mean() * 100
    return se, sp, (se + sp) / 2


def curve(p, lab, grid=GRID):
    """rows (b, Se, Sp, Score) for normal-class bias b."""
    out = []
    for b in grid:
        q = p.copy(); q[:, 0] = q[:, 0] * np.exp(b)
        out.append((b,) + score(q.argmax(1), lab))
    return np.array(out)


def at(cur, key_col, value):
    """Score of the curve where column key_col (1=Se, 2=Sp) equals value (linear interpolation)."""
    x, y = cur[:, key_col], cur[:, 3]
    o = np.argsort(x, kind='stable'); x, y = x[o], y[o]
    if value < x[0] or value > x[-1]:
        return float('nan')
    # average duplicates of x so interp is well defined
    ux, inv = np.unique(x, return_inverse=True)
    uy = np.bincount(inv, weights=y) / np.bincount(inv)
    return float(np.interp(value, ux, uy))


def paired(d):
    d = np.asarray([v for v in d if np.isfinite(v)])
    n = len(d)
    if n < 2:
        return dict(n=n, mean=float(d.mean()) if n else float('nan'))
    se = d.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, d.mean(), se)
    try:
        pw = float(stats.wilcoxon(d).pvalue)
    except ValueError:
        pw = float('nan')
    return dict(n=n, mean=float(d.mean()), sd=float(d.std(ddof=1)), ci_lo=float(lo), ci_hi=float(hi),
                p_t=float(stats.ttest_1samp(d, 0).pvalue), p_wilcoxon=pw, wins=int((d > 0).sum()))


def analyse(m0, arms, lab, out):
    """m0: {seed: probs}; arms: {name: {seed: probs (already corrected if needed)}}; writes CSV/JSON to out."""
    os.makedirs(out, exist_ok=True)
    seeds = sorted(m0)
    curves = {s: curve(m0[s], lab) for s in seeds}
    with open(os.path.join(out, 'm0_curve_per_seed.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['seed', 'b', 'Se', 'Sp', 'Score'])
        for s in seeds:
            for r in curves[s]:
                w.writerow([s] + [round(float(v), 4) for v in r])
    ens = np.mean([m0[s] for s in seeds], 0)
    ens_curve = curve(ens, lab)
    with open(os.path.join(out, 'm0_ens10_curve.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['b', 'Se', 'Sp', 'Score'])
        for r in ens_curve:
            w.writerow([round(float(v), 4) for v in r])

    per_seed, summary = [], []
    b0 = int(np.argmin(np.abs(GRID)))
    for name, probs in arms.items():
        for s in sorted(probs):
            if s not in curves:
                continue
            se, sp, sc = score(probs[s].argmax(1), lab)
            c = curves[s]
            per_seed.append(dict(arm=name, seed=s, Se=se, Sp=sp, Score=sc,
                                 M0_raw_Score=c[b0, 3],
                                 M0_at_matched_Sp=at(c, 2, sp), M0_at_matched_Se=at(c, 1, se),
                                 M0_oracle_Score=float(c[:, 3].max()),
                                 M0_oracle_b=float(c[np.argmax(c[:, 3]), 0])))
        for subset, ss in (('1-10', range(1, 11)), ('1-5', range(1, 6)), ('6-10', range(6, 11))):
            rows = [r for r in per_seed if r['arm'] == name and r['seed'] in ss]
            if not rows:
                continue
            for ref in ('M0_raw_Score', 'M0_at_matched_Sp', 'M0_at_matched_Se', 'M0_oracle_Score'):
                summary.append(dict(arm=name, seeds=subset, versus=ref,
                                    **paired([r['Score'] - r[ref] for r in rows])))
        # ensemble of this arm vs M0 ensemble curve
        es = [s for s in seeds if s in probs]
        e = np.mean([probs[s] for s in es], 0)
        se, sp, sc = score(e.argmax(1), lab)
        summary.append(dict(arm=name, seeds=f'ens{len(es)}', versus='M0ens_at_matched_Sp', n=1,
                            mean=sc - at(ens_curve, 2, sp), arm_Se=se, arm_Sp=sp, arm_Score=sc))
        summary.append(dict(arm=name, seeds=f'ens{len(es)}', versus='M0ens_oracle', n=1,
                            mean=sc - float(ens_curve[:, 3].max()), arm_Score=sc))

    keys = sorted({k for r in per_seed for k in r}, key=lambda k: list(per_seed[0]).index(k) if k in per_seed[0] else 99)
    with open(os.path.join(out, 'matched_per_seed.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in per_seed:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    keys = ['arm', 'seeds', 'versus', 'n', 'mean', 'sd', 'ci_lo', 'ci_hi', 'p_t', 'p_wilcoxon', 'wins',
            'arm_Se', 'arm_Sp', 'arm_Score']
    with open(os.path.join(out, 'matched_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in summary:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    return per_seed, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from test_labels import test_order
    from ensemble_nov import load_probs
    from prior_correct import mix_prior, pr_real
    names, lab, dev = test_order()

    def load(arm, correct):
        res = {}
        w = pr_real / mix_prior(arm) if correct else np.ones(4)
        for s in range(1, 11):
            p = load_probs(arm, s)
            if p is not None:
                assert p.shape == (len(lab), 4), (arm, s, p.shape)
                res[s] = p * w
        return res

    m0 = load('M0', False)
    assert len(m0) == 10, f'expected 10 M0 seeds, found {sorted(m0)}'
    arms = {'T_K3_corrected': load('T_K3', True), 'M3_corrected': load('M3', True),
            'T_K3_raw': load('T_K3', False), 'M3_raw': load('M3', False)}
    for k in ('M1b', 'K3'):
        arms[f'{k}_corrected'] = load(k, True)
    os.makedirs(args.out, exist_ok=True)
    # sanity check against the published per-run numbers
    with open(os.path.join(args.out, 'sanity_check.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['arm', 'seed', 'Se', 'Sp', 'Score'])
        for s, p in m0.items():
            w.writerow(['M0_raw', s] + [round(v, 2) for v in score(p.argmax(1), lab)])
        for name, d in arms.items():
            for s, p in d.items():
                w.writerow([name, s] + [round(v, 2) for v in score(p.argmax(1), lab)])
    analyse(m0, arms, lab, args.out)
    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(dict(grid=[float(GRID[0]), float(GRID[-1]), 0.01], n_test=int(len(lab)),
                       mix_prior={a: mix_prior(a).tolist() for a in ('T_K3', 'M3', 'M1b', 'K3')},
                       pr_real=pr_real.tolist(), seeds={k: sorted(v) for k, v in arms.items()}), f, indent=2)
    print('done ->', args.out)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Core results tables and figures from saved test probabilities (no training, no GPU).

Reads raw last-epoch probabilities of every available arm (freshly trained runs under artifacts/runs_nov, else the
copies shipped in results/saved/probs) and writes, to <out_dir>:
  per_run.csv                 Se / Sp / Score of every run, raw and prior-corrected
  arm_summary.csv             per arm: mean +- SD over seeds, paired difference to M0 (same seed), ensembles
  per_device_per_class.csv    ensemble Score per stethoscope and recall per class
  confusion_<arm>.csv         ensemble confusion matrices (corrected)
  RESULTS.md                  the same tables as Markdown
  F_paired_seeds.png/pdf      per-seed Score, M0 vs T_K3 (corrected)
  F_se_sp.png/pdf             Se-Sp plane of every arm, raw vs corrected
Prior correction (parameter free): p_c <- p_c * pi_real_c / pi_mix_c, applied to the (averaged) probabilities.
Bootstrap: 2000 resamples of the test cycles, seed 20261118.
"""
import csv
import os
import sys

import numpy as np
from scipy import stats

from ensemble_nov import load_probs, score
from prior_correct import mix_prior, pr_real
from test_labels import test_order

CLASSES = ['normal', 'crackle', 'wheeze', 'both']
ARMS = ['M0', 'M1b', 'M3', 'K3', 'T_K3']
LABELS = {'M0': 'real only', 'M1b': 'real duplicates balanced (control)', 'M3': 'SpecDiff balance',
          'K3': 'SpecDiff balance + soft labels', 'T_K3': 'transductive SpecDiff + soft labels'}
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'results/latest')
os.makedirs(OUT, exist_ok=True)

names, lab, dev = test_order()
dev = np.array(dev)
rng = np.random.default_rng(20261118)
BOOT = rng.integers(0, len(lab), (2000, len(lab)))


def weights(arm, corrected):
    return pr_real / mix_prior(arm) if corrected and arm != 'M0' else np.ones(4)


def collect():
    P = {}
    for arm in ARMS:
        seeds = {s: load_probs(arm, s) for s in range(1, 11)}
        seeds = {s: p for s, p in seeds.items() if p is not None}
        if seeds:
            P[arm] = seeds
    return P


def fmt(x, nd=2):
    return f'{x:.{nd}f}'


def main():
    P = collect()
    assert 'M0' in P, 'no M0 probabilities found (train M0 or keep results/saved/probs)'
    rows, scores = [], {}
    for arm, seeds in P.items():
        for corrected in (False, True):
            w = weights(arm, corrected)
            if corrected and arm == 'M0':
                continue
            for s, p in seeds.items():
                se, sp, sc = score((p * w).argmax(1), lab)
                scores[(arm, corrected, s)] = sc
                rows.append([arm, s, 'corrected' if corrected else 'raw', fmt(se), fmt(sp), fmt(sc)])
    with open(os.path.join(OUT, 'per_run.csv'), 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['arm', 'seed', 'probabilities', 'Se', 'Sp', 'Score'])
        w_.writerows(rows)

    m0_single = {s: scores[('M0', False, s)] for s in P['M0']}
    m0_ens = np.mean(list(P['M0'].values()), 0).argmax(1)
    summary, md = [], []
    for arm in P:
        for corrected in ((False,) if arm == 'M0' else (False, True)):
            w = weights(arm, corrected)
            sc = np.array([scores[(arm, corrected, s)] for s in sorted(P[arm])])
            common = [s for s in sorted(P[arm]) if s in m0_single]
            d = np.array([scores[(arm, corrected, s)] - m0_single[s] for s in common])
            ens_p = (np.mean(list(P[arm].values()), 0) * w).argmax(1)
            ens = score(ens_p, lab)
            m0_matched = np.mean([P['M0'][s] for s in common], 0).argmax(1)
            boot = np.array([score(ens_p[i], lab[i])[2] - score(m0_matched[i], lab[i])[2] for i in BOOT]) if arm != 'M0' else None
            row = dict(arm=arm, probabilities='corrected' if corrected else 'raw', n_seeds=len(sc), score_mean=sc.mean(),
                       score_sd=sc.std(ddof=1) if len(sc) > 1 else float('nan'), ens_Se=ens[0], ens_Sp=ens[1], ens_Score=ens[2])
            if arm != 'M0' and len(d) > 1:
                try:
                    pw = stats.wilcoxon(d).pvalue
                except ValueError:
                    pw = float('nan')
                lo, hi = np.percentile(boot, [2.5, 97.5])
                row.update(paired_diff=d.mean(), paired_wins=int((d > 0).sum()), paired_n=len(d),
                           p_ttest=stats.ttest_1samp(d, 0).pvalue, p_wilcoxon=pw,
                           ens_minus_M0ens=ens[2] - score(m0_matched, lab)[2], ens_ci_lo=lo, ens_ci_hi=hi)
            summary.append(row)
    keys = ['arm', 'probabilities', 'n_seeds', 'score_mean', 'score_sd', 'paired_diff', 'paired_wins', 'paired_n', 'p_ttest',
            'p_wilcoxon', 'ens_Se', 'ens_Sp', 'ens_Score', 'ens_minus_M0ens', 'ens_ci_lo', 'ens_ci_hi']
    with open(os.path.join(OUT, 'arm_summary.csv'), 'w', newline='') as f:
        w_ = csv.DictWriter(f, fieldnames=keys)
        w_.writeheader()
        for r in summary:
            w_.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k in keys})

    md.append('# Results (official ICBHI test, 2,756 cycles, last epoch, Score = (Se + Sp) / 2, %)\n')
    md.append('| Arm | Probabilities | Seeds | Single model mean ± SD | Paired vs M0 (same seed) | Wins | p (t-test / Wilcoxon) | Ensemble Se / Sp / Score | Ensemble − size-matched M0 ensemble [95% CI] |')
    md.append('|---|---|---|---|---|---|---|---|---|')
    for r in summary:
        pd_ = f"{r['paired_diff']:+.2f}" if 'paired_diff' in r else '–'
        wins = f"{r['paired_wins']}/{r['paired_n']}" if 'paired_wins' in r else '–'
        pv = f"{r['p_ttest']:.3f} / {r['p_wilcoxon']:.3f}" if 'p_ttest' in r else '–'
        ec = f"{r['ens_minus_M0ens']:+.2f} [{r['ens_ci_lo']:+.2f}, {r['ens_ci_hi']:+.2f}]" if 'ens_ci_lo' in r else '–'
        md.append(f"| {r['arm']} | {r['probabilities']} | {r['n_seeds']} | {r['score_mean']:.2f} ± {r['score_sd']:.2f} | {pd_} | {wins} | {pv} | "
                  f"{r['ens_Se']:.1f} / {r['ens_Sp']:.1f} / {r['ens_Score']:.2f} | {ec} |")

    # per device / per class / confusion of the ensembles
    dev_rows = []
    md.append('\n## Per stethoscope and per class (ensemble of all seeds)\n')
    md.append('| Ensemble | AKG C417L | Meditron | Litt3200 | recall N | C | W | B |')
    md.append('|---|---|---|---|---|---|---|---|')
    for arm in P:
        w = weights(arm, True)
        ens_p = (np.mean(list(P[arm].values()), 0) * w).argmax(1)
        devs = [score(ens_p[dev == k], lab[dev == k])[2] for k in ('AKGC417L', 'Meditron', 'Litt3200')]
        rec = [(ens_p[lab == c] == c).mean() * 100 for c in range(4)]
        tag = f'{arm}x{len(P[arm])}' + ('' if arm == 'M0' else ' corrected')
        dev_rows.append([tag] + [fmt(v, 1) for v in devs + rec])
        md.append(f'| {tag} | ' + ' | '.join(fmt(v, 1) for v in devs + rec) + ' |')
        cm = np.zeros((4, 4), int)
        for t, q in zip(lab, ens_p):
            cm[t, q] += 1
        with open(os.path.join(OUT, f'confusion_{arm}.csv'), 'w', newline='') as f:
            cw = csv.writer(f)
            cw.writerow(['true\\pred'] + CLASSES)
            for i in range(4):
                cw.writerow([CLASSES[i]] + list(cm[i]))
    with open(os.path.join(OUT, 'per_device_per_class.csv'), 'w', newline='') as f:
        cw = csv.writer(f)
        cw.writerow(['ensemble', 'AKGC417L', 'Meditron', 'Litt3200', 'recall_normal', 'recall_crackle', 'recall_wheeze', 'recall_both'])
        cw.writerows(dev_rows)
    with open(os.path.join(OUT, 'RESULTS.md'), 'w') as f:
        f.write('\n'.join(md) + '\n')

    figures(P, scores)
    print('\n'.join(md))
    print(f'\nwrote {OUT}')


def figures(P, scores):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                         'axes.grid': True, 'grid.color': '#e4e3df', 'axes.axisbelow': True, 'legend.frameon': False})
    if 'T_K3' in P:
        seeds = sorted(set(P['T_K3']) & set(P['M0']))
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        for s in seeds:
            a, b = scores[('M0', False, s)], scores[('T_K3', True, s)]
            ax.plot([0, 1], [a, b], color=C3 if b > a else C2, lw=1.4, alpha=0.9)
        ax.scatter(np.zeros(len(seeds)), [scores[('M0', False, s)] for s in seeds], color='#52514e', s=28, zorder=3)
        ax.scatter(np.ones(len(seeds)), [scores[('T_K3', True, s)] for s in seeds], color=C1, s=28, zorder=3)
        ax.set_xticks([0, 1], ['M0 real only', 'T_K3 + prior correction'])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel('ICBHI Score (%)')
        ax.set_title('Same-seed pairs (green = synthetic arm ahead)', fontsize=10)
        fig.savefig(os.path.join(OUT, 'F_paired_seeds.png'))
        fig.savefig(os.path.join(OUT, 'F_paired_seeds.pdf'))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    xs = np.linspace(30, 90, 50)
    for sc_ in (56, 58, 60):
        ax.plot(xs, 2 * sc_ - xs, color='#c9c8c3', lw=0.8, ls='--')
        ax.text(xs[-1], 2 * sc_ - xs[-1], f' {sc_}', color='#9a9893', fontsize=7, va='center')
    for arm in P:
        for corrected, marker in ((False, 'o'), (True, 's')):
            if corrected and arm == 'M0':
                continue
            w = weights(arm, corrected)
            pts = [score((p * w).argmax(1), lab) for p in P[arm].values()]
            ax.scatter([q[1] for q in pts], [q[0] for q in pts], s=14, marker=marker, alpha=0.55,
                       label=f'{arm} {"corrected" if corrected else "raw"}' if True else None)
    ax.set_xlabel('Specificity (%)')
    ax.set_ylabel('Sensitivity (%)')
    ax.set_xlim(60, 90)
    ax.set_ylim(30, 55)
    ax.legend(fontsize=6.5, ncol=2, loc='upper right')
    fig.savefig(os.path.join(OUT, 'F_se_sp.png'))
    fig.savefig(os.path.join(OUT, 'F_se_sp.pdf'))
    plt.close(fig)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the deliverables folder (tables + figures) from saved outputs only.

No training and no new test-set decisions: every number comes from saved
probabilities / json / npz written by the logged runs (looks 1-9 in EXPERIMENT_LOG.md).
Usage: python make_deliverables.py <out_dir>
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from test_labels import test_order
from ensemble_nov import run_dir, score
from prior_correct import mix_prior, pr_real

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.abspath(sys.argv[1])
RES, FIG = os.path.join(OUT, 'results'), os.path.join(OUT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

CLASSES = ['normal', 'crackle', 'wheeze', 'both']
DEVICES = ['AKGC417L', 'Meditron', 'Litt3200']
# validated reference palette, first three slots (all-pairs safe), plus neutral grays
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, GRAY = '#0b0b0b', '#52514e', '#e4e3df', '#9a9893'
plt.rcParams.update({
    'font.size': 10, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'legend.frameon': False,
    'lines.linewidth': 2, 'lines.solid_capstyle': 'round',
})

ARM_DESC = {
    'M0': 'real only (control)',
    'M1b': 'real duplicates, class-balanced (non-generative control)',
    'M3': 'SpecDiff synthetic, class-balanced',
    'M3f': 'SpecDiff synthetic, class-balanced, probe-filtered (p>=0.6); locked primary arm of look 1',
    'M6': 'SpecDiff synthetic, every class to 2x normal count',
    'K3': 'SpecDiff synthetic balance, soft labels from real-only teacher',
    'T_K3': 'transductive SpecDiff (unlabelled test audio) + test-device mix + soft labels, balance',
    'X3_tt': 'SDEdit translation of train cycles to test-device mix, balance',
    'A3': 'M3 + adversarial fine-tuning (AFT, base paper)',
    'A3t': 'T_K3-style transductive pool + AFT',
}
ARMS = list(ARM_DESC)


def write_csv(name, header, rows):
    with open(os.path.join(RES, name), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'))
    plt.close(fig)


def r2(x):
    return round(float(x), 2)


# ---------------------------------------------------------------- test set
names, lab, dev = test_order()
dev = np.array(dev)
rng = np.random.default_rng(20261118)
BOOT = rng.integers(0, len(lab), (2000, len(lab)))


def recalls(pred):
    return [(pred[lab == c] == c).mean() * 100 for c in range(4)]


def dev_scores(pred):
    return [score(pred[dev == k], lab[dev == k])[2] for k in DEVICES]


def boot(pa, pb):
    d = np.array([score(pa[i], lab[i])[2] - score(pb[i], lab[i])[2] for i in BOOT])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return score(pa, lab)[2] - score(pb, lab)[2], lo, hi, (d > 0).mean()


probs, weights = {}, {}
for arm in ARMS:
    weights[arm] = np.ones(4) if arm == 'M0' else pr_real / mix_prior(arm)
    probs[arm] = {}
    for s in range(1, 11):
        d = run_dir(arm, s)
        if d:
            probs[arm][s] = np.load(os.path.join(d, 'probs_test_last.npy'))

rows = []
per = {}  # (arm, kind) -> list of (seed, se, sp, sc, rec, devsc)
for arm in ARMS:
    for s, p in sorted(probs[arm].items()):
        for kind, w in (('raw', np.ones(4)), ('corrected', weights[arm])):
            if arm == 'M0' and kind == 'corrected':
                continue
            pred = (p * w).argmax(1)
            se, sp, sc = score(pred, lab)
            rec, ds = recalls(pred), dev_scores(pred)
            per.setdefault((arm, kind), []).append((s, se, sp, sc, rec, ds))
            rows.append([arm, s, kind, r2(se), r2(sp), r2(sc)] + [r2(x) for x in rec] + [r2(x) for x in ds])
write_csv('test_per_run.csv', ['arm', 'seed', 'probs', 'Se', 'Sp', 'Score', 'rec_normal', 'rec_crackle', 'rec_wheeze',
                               'rec_both', 'Score_AKGC417L', 'Score_Meditron', 'Score_Litt3200'], rows)

m0_5 = np.mean([probs['M0'][s] for s in range(1, 6)], 0).argmax(1)
m0_10 = np.mean([probs['M0'][s] for s in range(1, 11)], 0).argmax(1)
summary = []
for arm in ARMS:
    seeds = sorted(probs[arm])
    s5 = [s for s in seeds if s <= 5]
    for kind in ('raw', 'corrected'):
        if arm == 'M0' and kind == 'corrected':
            continue
        w = weights[arm] if kind == 'corrected' else np.ones(4)
        v = np.array([x[3] for x in per[(arm, kind)] if x[0] <= 5])
        e5 = (np.mean([probs[arm][s] for s in s5], 0) * w).argmax(1)
        dd, lo, hi, pp = boot(e5, m0_5)
        se, sp, sc = score(e5, lab)
        summary.append([arm, kind, len(v), r2(v.mean()), r2(v.std(ddof=1)), r2(se), r2(sp), r2(sc), r2(dd), r2(lo), r2(hi),
                        round(pp, 3)] + [r2(x) for x in dev_scores(e5)] + [r2(x) for x in recalls(e5)] + [ARM_DESC[arm]])
write_csv('test_arm_summary_seeds1-5.csv',
          ['arm', 'probs', 'n_seeds', 'single_mean', 'single_sd', 'ens5_Se', 'ens5_Sp', 'ens5_Score', 'ens5_vs_M0x5',
           'ci_lo', 'ci_hi', 'P_gt0', 'ens5_AKG', 'ens5_Meditron', 'ens5_Litt3200', 'ens5_rec_N', 'ens5_rec_C', 'ens5_rec_W',
           'ens5_rec_B', 'description'], summary)

# look 9: 10 seeds, fixed correction
l9 = []
m0_single = {x[0]: x for x in per[('M0', 'raw')]}
for arm in ('M0', 'T_K3', 'M3'):
    kind = 'raw' if arm == 'M0' else 'corrected'
    P = per[(arm, kind)]
    v = np.array([x[3] for x in P])
    diffs = np.array([x[3] - m0_single[x[0]][3] for x in P]) if arm != 'M0' else np.zeros(len(P))
    e10 = (np.mean([probs[arm][s] for s in range(1, 11)], 0) * (weights[arm] if arm != 'M0' else 1)).argmax(1)
    dd, lo, hi, pp = boot(e10, m0_10) if arm != 'M0' else (0, 0, 0, 0)
    sdd = diffs.std(ddof=1) if arm != 'M0' else 0
    se_, sp_, sc_ = score(e10, lab)
    l9.append([arm, len(P), r2(v.mean()), r2(v.std(ddof=1)), r2(np.mean([x[1] for x in P])), r2(np.mean([x[2] for x in P])),
               r2(diffs.mean()), r2(diffs.mean() - 2.262 * sdd / np.sqrt(len(P))), r2(diffs.mean() + 2.262 * sdd / np.sqrt(len(P))),
               int((diffs > 0).sum()), r2(se_), r2(sp_), r2(sc_), r2(dd), r2(lo), r2(hi), round(pp, 3)]
              + [r2(x) for x in dev_scores(e10)] + [r2(x) for x in recalls(e10)])
write_csv('look9_10seed_summary.csv',
          ['arm', 'n_seeds', 'single_mean', 'single_sd', 'single_Se', 'single_Sp', 'paired_diff_vs_M0', 'paired_ci_lo',
           'paired_ci_hi', 'wins_of_10', 'ens10_Se', 'ens10_Sp', 'ens10_Score', 'ens10_vs_M0x10', 'boot_lo', 'boot_hi',
           'P_gt0', 'ens10_AKG', 'ens10_Meditron', 'ens10_Litt3200', 'ens10_rec_N', 'ens10_rec_C', 'ens10_rec_W',
           'ens10_rec_B'], l9)

# confusion matrices (10-model ensembles)
for arm, tag in (('M0', 'M0x10_raw'), ('T_K3', 'T_K3x10_corrected'), ('M3', 'M3x10_corrected')):
    w = weights[arm] if arm != 'M0' else np.ones(4)
    pred = (np.mean([probs[arm][s] for s in range(1, 11)], 0) * w).argmax(1)
    cm = np.zeros((4, 4), int)
    for t, q in zip(lab, pred):
        cm[t, q] += 1
    write_csv(f'confusion_{tag}.csv', ['true\\pred'] + CLASSES, [[CLASSES[i]] + list(cm[i]) for i in range(4)])

# ---------------------------------------------------------------- validation
os.system(f'cd {os.path.dirname(os.path.abspath(__file__))} && {sys.executable} collect_nov.py --runs_dir {ROOT}/artifacts/runs_nov '
          f'--phase sel --out_csv {RES}/validation_screening_all_arms.csv > {RES}/validation_screening_all_arms.txt 2>&1')
os.system(f'cd {os.path.dirname(os.path.abspath(__file__))} && {sys.executable} collect_nov.py --runs_dir {ROOT}/artifacts/runs_nov '
          f'--phase final --out_csv {RES}/test_collect_nov_final.csv > {RES}/test_collect_nov_final.txt 2>&1')

vp = np.load(os.path.join(ROOT, 'artifacts/specdiff/valprobs.npz'))
REALV = np.array([1804, 1066, 445, 323], float)


def mix_prior_sel(arm):
    c = REALV.copy()
    with open(os.path.join(ROOT, f'artifacts/specdiff/mixes/sel_{arm}.csv'), newline='') as f:
        for row in csv.DictReader(f):
            c[int(row['label'])] += 1
    return c / c.sum()


vrows = []
for arm in [k for k in vp.files if k not in ('label', 'device')]:
    P, vl = vp[arm], vp['label']
    w = np.ones(4) if arm == 'M0' else (REALV / REALV.sum()) / mix_prior_sel(arm)
    raw = np.array([score(p.argmax(1), vl) for p in P])
    adj = np.array([score((p * w).argmax(1), vl) for p in P])
    vrows.append([arm, len(P), r2(raw[:, 2].mean()), r2(raw[:, 0].mean()), r2(raw[:, 1].mean()), r2(adj[:, 2].mean()),
                  r2(adj[:, 0].mean()), r2(adj[:, 1].mean())])
write_csv('validation_prior_correction_check.csv',
          ['arm', 'n_seeds', 'raw_Score', 'raw_Se', 'raw_Sp', 'corrected_Score', 'corrected_Se', 'corrected_Sp'], vrows)

# ---------------------------------------------------------------- generator gate
gate = [['DiffWave v1 (old, vocoder)', 'E1 seed1', 'train augmentation pool', '', '', 6.01, 2.35, '', '', 'handoff.md / diag v1'],
        ['DiffWave v2 (old, 33.9k steps)', 'E1 seed1', 'train pool (120/class)', '', '', 21.67, 2.50, '', '', 'artifacts/diag/diag_report.md']]
qfiles = sorted(glob.glob(os.path.join(ROOT, 'artifacts/specdiff/pools/*/quality_*.json')))
for q in qfiles:
    j = json.load(open(q))
    c = j['classes']
    pool = os.path.basename(os.path.dirname(q))
    gate.append([f'SpecDiff {pool}', os.path.basename(os.path.dirname(j['probe'])), j['ref']]
                + [r2(c[k]['syn_intended_acc'] * 100) for k in CLASSES]
                + [' / '.join(f"{c[k]['kid_syn_vs_ref']:.2f}" for k in CLASSES),
                   ' / '.join(f"{c[k]['nn_ratio']:.2f}" for k in CLASSES), os.path.relpath(q, ROOT)])
    gate.append([f'  real reference recall ({pool})', '', j['ref']] + [r2(c[k]['real_ref_recall'] * 100) for k in CLASSES]
                + [' / '.join(f"{c[k]['kid_realtrain_vs_ref']:.2f}" for k in CLASSES), '', ''])
# old rows have only wheeze/both: pad to the same columns
gate[0] = gate[0][:3] + ['', ''] + gate[0][5:7] + gate[0][7:]
gate[1] = gate[1][:3] + ['', ''] + gate[1][5:7] + gate[1][7:]
write_csv('generator_quality_gate.csv',
          ['generator', 'probe', 'reference', 'acc_normal', 'acc_crackle', 'acc_wheeze', 'acc_both', 'KID_N/C/W/B',
           'nn_ratio_N/C/W/B', 'source'], [g[:10] for g in gate])

# ---------------------------------------------------------------- diffusion classifier curve
dc = []
for f in sorted(glob.glob(os.path.join(ROOT, 'artifacts/specdiff/dclf/*.npz'))):
    d = np.load(f)
    tag = os.path.basename(f)[:-4]
    pred = d['err'][:, :, :4].mean(1).argmin(1)
    se, sp, sc = score(pred, d['label'])
    gen = tag.rsplit('_s', 1)[0]
    dc.append([tag, gen, int(d['generator_step']), 'test' if tag.endswith('test') else 'validation', r2(se), r2(sp), r2(sc)])
write_csv('diffusion_classifier_by_step.csv', ['file', 'generator', 'step', 'set', 'Se', 'Sp', 'Score'], dc)

# ---------------------------------------------------------------- figures
# F1 real vs synthetic log-mel grid
tr = np.load(os.path.join(ROOT, 'artifacts/data/fbank_cache/train.npz'))
pool_dir = os.path.join(ROOT, 'artifacts/specdiff/pools/full_tt_s20000_w2.0_target')
pool = np.load(os.path.join(pool_dir, 'pool.npy'), mmap_mode='r')
with open(os.path.join(pool_dir, 'pool.csv'), newline='') as f:
    prow = list(csv.DictReader(f))
g = np.random.default_rng(7)
FR = 300  # 3 s at 10 ms hop
fig, ax = plt.subplots(4, 4, figsize=(11, 7.2), sharex=True, sharey=True)
for c in range(4):
    real_idx = np.where((tr['label'] == c) & (tr['device'] == 0) & (tr['duration'] >= 3.0))[0]
    syn_idx = [i for i, r in enumerate(prow) if int(r['label']) == c and r['device'] == 'AKGC417L' and float(r['duration']) >= 3.0]
    picks = [('real', tr['fbank'][i]) for i in g.choice(real_idx, 2, replace=False)] + \
            [('SpecDiff', np.asarray(pool[i], np.float32)) for i in g.choice(syn_idx, 2, replace=False)]
    for k, (src, m) in enumerate(picks):
        a = ax[c, k]
        a.imshow(m[:FR].T, origin='lower', aspect='auto', cmap='magma', vmin=-1.3, vmax=0.8, extent=[0, 3, 0, 128])
        a.grid(False)
        if c == 0:
            a.set_title(f'{src} {k % 2 + 1}', fontsize=10, color=INK)
        if k == 0:
            a.set_ylabel(f'{CLASSES[c]}\nmel bin', color=INK)
        if c == 3:
            a.set_xlabel('time (s)')
fig.suptitle('Real vs SpecDiff-generated log-mel spectrograms (AKG C417L, first 3 s; pool used by T_K3)', color=INK, y=1.0)
save(fig, 'F1_real_vs_synthetic_logmel_grid')

# F2 generator gate: intended-class accuracy for wheeze / both
q = json.load(open(os.path.join(ROOT, 'artifacts/specdiff/pools/full_s50000_w3.5_empirical/quality_test.json')))['classes']
labels_ = ['wheeze', 'both']
old1, old2 = [6.01, 2.35], [21.67, 2.50]
new = [q['wheeze']['syn_intended_acc'] * 100, q['both']['syn_intended_acc'] * 100]
ceil = [q['wheeze']['real_ref_recall'] * 100, q['both']['real_ref_recall'] * 100]
fig, a = plt.subplots(figsize=(6.2, 3.6))
x = np.arange(2)
bw = 0.24
for i, (vals, col, nm) in enumerate(((old1, GRAY, 'DiffWave v1 (old)'), (old2, INK2, 'DiffWave v2 (old)'), (new, C1, 'SpecDiff (new)'))):
    a.bar(x + (i - 1) * bw, vals, bw * 0.9, color=col, label=nm)
    for xx, v in zip(x + (i - 1) * bw, vals):
        a.text(xx, v + 1.5, f'{v:.0f}%', ha='center', fontsize=8.5, color=INK)
a.scatter(x + bw, ceil, marker='_', s=500, color=INK, linewidths=2, label='probe recall on real test cycles', zorder=3)
a.set_xticks(x, labels_)
a.set_ylabel('% classified as intended class\nby a real-only AST probe')
a.set_ylim(0, 105)
a.legend(loc='upper right', fontsize=8.5)
a.set_title('Generator quality gate: class identity of generated samples', color=INK)
save(fig, 'F2_generator_gate_old_vs_new')

# F3 diffusion classifier Score vs step
fig, a = plt.subplots(figsize=(6.2, 3.6))
for gen, col, nm in (('sel', C1, 'SpecDiff G_sel (plain)'), ('sel_r3', C2, 'SpecDiff r3 (class-margin hinge)')):
    pts = sorted((r[2], r[6]) for r in dc if r[1] == gen and r[3] == 'validation')
    if pts:
        a.plot([p[0] / 1000 for p in pts], [p[1] for p in pts], color=col, marker='o', ms=6, label=nm)
a.axhline(70.0, color=GRAY, lw=1.5, label='AST real-only (M0), validation')
tst = [r for r in dc if r[3] == 'test' and r[1] == 'full_r3']
if tst:
    a.scatter([25], [tst[0][6]], marker='x', color=C2, s=60, zorder=3, label=f'r3 on official test ({tst[0][6]:.1f})')
a.set_xlabel('generator training step (thousands)')
a.set_ylabel('ICBHI Score (validation)')
a.set_title('Generator used directly as a classifier', color=INK)
a.legend(fontsize=8, loc='lower right')
save(fig, 'F3_diffusion_classifier_curve')

# F4 paired seeds (look 9)
fig, a = plt.subplots(figsize=(6.2, 3.6))
s_ = np.arange(1, 11)
m0v = [m0_single[s][3] for s in s_]
tk = {x[0]: x[3] for x in per[('T_K3', 'corrected')]}
tkv = [tk[s] for s in s_]
for s, y0, y1 in zip(s_, m0v, tkv):
    a.plot([s, s], [y0, y1], color=GRID, lw=2, zorder=1)
a.scatter(s_, m0v, color=GRAY, s=40, zorder=2, label=f'M0 real only (mean {np.mean(m0v):.2f})', edgecolors='white', linewidths=1.5)
a.scatter(s_, tkv, color=C2, s=40, zorder=3, label=f'T_K3 + prior correction (mean {np.mean(tkv):.2f})', edgecolors='white', linewidths=1.5)
a.axvline(5.5, color=GRAY, lw=1)
a.text(5.65, min(m0v) + 0.05, 'seeds 6-10: added in look 9', fontsize=8, color=INK2)
a.set_xticks(s_)
a.set_xlabel('training seed (same seed = paired)')
a.set_ylabel('official test Score')
d_ = np.array(tkv) - np.array(m0v)
a.set_title(f'Per-seed comparison: +{d_.mean():.2f} mean, {int((d_ > 0).sum())}/10 seeds ahead', color=INK)
a.legend(fontsize=8, loc='upper right')
save(fig, 'F4_paired_seeds_T_K3_vs_M0')

# F5 all arms: single-model mean +- 95% CI, raw and corrected (seeds 1-5)
fig, a = plt.subplots(figsize=(7, 4.6))
ylab = []
for i, arm in enumerate(ARMS):
    for kind, col, off in (('raw', GRAY, -0.15), ('corrected', C1, 0.15)):
        if (arm, kind) not in per:
            continue
        v = np.array([x[3] for x in per[(arm, kind)] if x[0] <= 5])
        ci = 2.776 * v.std(ddof=1) / np.sqrt(len(v))
        a.errorbar(v.mean(), i + off, xerr=ci, fmt='o', color=col, ms=6, capsize=0, lw=2,
                   label=('raw' if kind == 'raw' else 'prior-corrected') if i == 1 else None)
    ylab.append(arm)
m0m = np.mean([x[3] for x in per[('M0', 'raw')] if x[0] <= 5])
a.axvline(m0m, color=INK2, lw=1)
a.text(m0m + 0.05, len(ARMS) - 0.45, f'M0 {m0m:.2f}', fontsize=8, color=INK2)
a.set_yticks(range(len(ARMS)), ylab)
a.invert_yaxis()
a.set_xlabel('official test Score, single model, mean ± 95% CI over seeds 1-5')
a.set_title('Every arm tested on the official test set (looks 1-8)', color=INK)
a.legend(fontsize=8, loc='upper left')
save(fig, 'F5_all_arms_test_forest')

# F6 Se vs Sp: raw -> corrected
fig, a = plt.subplots(figsize=(6.2, 4.4))
F6_ARMS = {'M1b': (C3, (0.4, 0.3)), 'M3': (C1, (-0.2, -0.7)), 'T_K3': (C2, (0.3, 0.3)), 'A3t': (C1, (0.3, 0.1))}
for arm, (col, off) in F6_ARMS.items():
    rs = per[(arm, 'raw')]
    cs = per[(arm, 'corrected')]
    r_ = (np.mean([x[1] for x in rs]), np.mean([x[2] for x in rs]))
    c_ = (np.mean([x[1] for x in cs]), np.mean([x[2] for x in cs]))
    a.annotate('', xy=(c_[1], c_[0]), xytext=(r_[1], r_[0]), arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
    a.scatter([r_[1]], [r_[0]], color=GRAY, s=25, zorder=3)
    a.text(c_[1] + off[0], c_[0] + off[1], arm, fontsize=8.5, color=INK)
    a.text(r_[1] - 0.3, r_[0] + 0.2, arm, fontsize=7.5, color=INK2, ha='right')
m0s = (np.mean([x[1] for x in per[('M0', 'raw')]]), np.mean([x[2] for x in per[('M0', 'raw')]]))
a.scatter([m0s[1]], [m0s[0]], color=INK, marker='s', s=40, zorder=4)
a.text(m0s[1] + 0.2, m0s[0] + 0.3, 'M0', fontsize=8, color=INK)
sp_grid = np.linspace(62, 82, 50)
for lvl in (58, 60):
    a.plot(sp_grid, 2 * lvl - sp_grid, color=GRID, lw=1, zorder=0)
    a.text(2 * lvl - 51.2 + 0.2, 51.2, f'Score {lvl}', fontsize=7.5, color=INK2, ha='left')
a.set_xlabel('specificity Sp (%)')
a.set_ylabel('sensitivity Se (%)')
a.set_title('Prior correction trades Se for Sp\n(gray dot = raw mean over all seeds, arrow head = corrected; M1b = real-duplicate control)', color=INK, fontsize=9.5)
a.set_xlim(63, 82)
a.set_ylim(38, 52)
save(fig, 'F6_se_sp_prior_correction')

# F7 per-device and F8 per-class: per-model means over 10 seeds
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
groups = [('M0', 'raw', GRAY, 'M0 real only'), ('M3', 'corrected', C1, 'M3 + correction'),
          ('T_K3', 'corrected', C2, 'T_K3 + correction')]
for j, (ttl, idx, xt) in enumerate((('Score by stethoscope (test)', 5, DEVICES), ('Recall by class (test)', 4, CLASSES))):
    a = axs[j]
    x = np.arange(len(xt))
    for i, (arm, kind, col, nm) in enumerate(groups):
        vals = np.mean([x_[idx] for x_ in per[(arm, kind)]], 0)
        a.bar(x + (i - 1) * 0.26, vals, 0.24, color=col, label=nm)
    a.set_xticks(x, xt)
    a.set_title(ttl + ', mean of 10 single models', color=INK, fontsize=9.5)
    a.set_ylabel('%')
h, l_ = axs[0].get_legend_handles_labels()
fig.legend(h, l_, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
save(fig, 'F7_per_device_and_per_class')

# F8 validation vs test (raw, per-model means)
vmap = {'M0': 'M0', 'M1b': 'M1b', 'M3': 'M3_s50k_w3.5', 'M3f': 'M3f_s50k_w3.5', 'M6': 'M6_s50k_w3.5', 'K3': 'K3',
        'T_K3': 'K3_tv', 'X3_tt': 'X3_tv'}
vsel = {}
with open(os.path.join(RES, 'validation_screening_all_arms.csv'), newline='') as f:
    for row in csv.DictReader(f):
        vsel[row.get('arm')] = row
fig, a = plt.subplots(figsize=(5.4, 4))
pts = []
for arm, va in vmap.items():
    if va not in vsel:
        continue
    vx = float(vsel[va]['Score'])
    ty = np.mean([x[3] for x in per[(arm, 'raw')] if x[0] <= 5])
    pts.append((arm, vx, ty))
    a.scatter(vx, ty, color=INK if arm == 'M0' else C1, s=45, zorder=3, edgecolors='white', linewidths=1.5)
    a.text(vx + 0.05, ty + 0.08, arm, fontsize=8, color=INK)
a.set_xlabel('validation Score (12 train patients)')
a.set_ylabel('official test Score (raw)')
a.set_title('Validation did not predict test', color=INK)
save(fig, 'F8_validation_vs_test')
write_csv('validation_vs_test_points.csv', ['arm', 'validation_score', 'test_score_raw_seeds1-5'], [[p[0], r2(p[1]), r2(p[2])] for p in pts])
print('done', OUT)

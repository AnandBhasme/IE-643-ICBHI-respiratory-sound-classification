#!/usr/bin/env python3
"""Look 9 (declared in EXPERIMENT_LOG): size-matched 10-seed comparison with the fixed look-8 prior correction."""
import os
import numpy as np
from test_labels import test_order
from ensemble_nov import load_probs, score
from prior_correct import mix_prior, pr_real

names, lab, dev = test_order()
dev = np.array(dev)
rng = np.random.default_rng(20261118)
boot = rng.integers(0, len(lab), (2000, len(lab)))
SEEDS = range(1, 11)


def load(arm):
    w = np.ones(4) if arm == 'M0' else pr_real / mix_prior(arm)
    out = {}
    for s in SEEDS:
        p = load_probs(arm, s)
        if p is not None:
            out[s] = p
    return out, w


def boot_diff(pa, pb):
    d = np.array([score(pa[i], lab[i])[2] - score(pb[i], lab[i])[2] for i in boot])
    return np.percentile(d, [2.5, 97.5]), (d > 0).mean()


m0, _ = load('M0')
m0_ens = np.mean(list(m0.values()), 0).argmax(1)
m0_single = {s: score(p.argmax(1), lab)[2] for s, p in m0.items()}
print(f'M0   n={len(m0)} single {np.mean(list(m0_single.values())):.2f} ± {np.std(list(m0_single.values()), ddof=1):.2f} | ens {score(m0_ens, lab)[2]:.2f}')
for arm in ('T_K3', 'M3'):
    P, w = load(arm)  # as look 8: weights on each model, and on the averaged ensemble
    sc = {s: score((p * w).argmax(1), lab) for s, p in P.items()}
    v = np.array([x[2] for x in sc.values()])
    paired = [sc[s][2] - m0_single[s] for s in sc if s in m0_single]
    ens = (np.mean(list(P.values()), 0) * w).argmax(1)
    se, sp, es = score(ens, lab)
    (lo, hi), pp = boot_diff(ens, m0_ens)
    devs = ' '.join(f'{k[:6]} {score(ens[dev == k], lab[dev == k])[2]:.1f}' for k in ('AKGC417L', 'Meditron', 'Litt3200'))
    rec = ' '.join(f'{(ens[lab == c] == c).mean() * 100:.1f}' for c in range(4))
    print(f'{arm:4s} n={len(P)} single {v.mean():.2f} ± {v.std(ddof=1):.2f} (Se {np.mean([x[0] for x in sc.values()]):.1f} Sp {np.mean([x[1] for x in sc.values()]):.1f}) '
          f'| paired vs M0 {np.mean(paired):+.2f}, wins {sum(d > 0 for d in paired)}/{len(paired)} [{" ".join(f"{d:+.2f}" for d in paired)}]')
    print(f'      ens Se {se:.2f} Sp {sp:.2f} Score {es:.2f} vs M0x{len(m0)} {es - score(m0_ens, lab)[2]:+.2f} [{lo:+.2f}, {hi:+.2f}] P(>0)={pp:.3f} | {devs} | recall N/C/W/B {rec}')

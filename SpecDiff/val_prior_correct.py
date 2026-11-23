#!/usr/bin/env python3
"""Validation check of the look-8 prior correction (sel arms; real = train minus val)."""
import csv, os
import numpy as np
from ensemble_nov import score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REAL = np.array([1804, 1066, 445, 323], dtype=float)
d = np.load(os.path.join(ROOT, 'artifacts/specdiff/valprobs.npz'))
lab = d['label']
pr = REAL / REAL.sum()


def mix_prior(arm):
    c = REAL.copy()
    with open(os.path.join(ROOT, f'artifacts/specdiff/mixes/sel_{arm}.csv'), newline='') as f:
        for row in csv.DictReader(f):
            c[int(row['label'])] += 1
    return c / c.sum()


for arm in [k for k in d.files if k not in ('label', 'device')]:
    P = d[arm]
    w = np.ones(4) if arm == 'M0' else pr / mix_prior(arm)
    raw = np.array([score(p.argmax(1), lab) for p in P])
    adj = np.array([score((p * w).argmax(1), lab) for p in P])
    print(f'{arm:14s} n={len(P)} raw {raw[:,2].mean():.2f} (Se {raw[:,0].mean():.1f} Sp {raw[:,1].mean():.1f}) '
          f'-> adj {adj[:,2].mean():.2f} (Se {adj[:,0].mean():.1f} Sp {adj[:,1].mean():.1f}) | '
          f'ens {score(P.mean(0).argmax(1), lab)[2]:.2f} -> {score((P.mean(0) * w).argmax(1), lab)[2]:.2f}  w={np.round(w,2)}')

#!/usr/bin/env python3
"""Look 8: parameter-free prior correction of saved test probabilities (declared in EXPERIMENT_LOG)."""
import csv, glob, os
import numpy as np
from test_labels import test_order
from ensemble_nov import run_dir, score

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
REAL = np.array([2063, 1215, 501, 363], dtype=float)
names, lab, dev = test_order()


SAVED_COUNTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/saved/mix_class_counts.json')


def mix_prior(arm):
    """Class prior of the final training mix of `arm` (real cycles + added rows).
    Uses the manifest written by build_mix.py when present, else results/saved/mix_class_counts.json."""
    path = os.path.join(ROOT, f'artifacts/specdiff/mixes/final_{arm}.csv')
    if os.path.isfile(path):
        counts = REAL.copy()
        with open(path, newline='') as f:
            for row in csv.DictReader(f):
                counts[int(row['label'])] += 1
    else:
        import json
        with open(SAVED_COUNTS) as f:
            counts = np.array(json.load(f)[arm], dtype=float)
    return counts / counts.sum()


pr_real = REAL / REAL.sum()


def main():
  for arm in['M0', 'M1b', 'M3', 'M3f', 'M6', 'K3', 'T_K3', 'X3_tt', 'A3', 'A3t']:
      w = np.ones(4) if arm == 'M0' else pr_real / mix_prior(arm)
      probs = [np.load(os.path.join(d, 'probs_test_last.npy')) for d in
               (run_dir(arm, s) for s in range(1, 6)) if d]
      if not probs:
          continue
      raw = [score(p.argmax(1), lab) for p in probs]
      adj = [score((p * w).argmax(1), lab) for p in probs]
      e_raw = score(np.mean(probs, 0).argmax(1), lab)
      e_adj = score((np.mean(probs, 0) * w).argmax(1), lab)
      r, a = np.array(raw), np.array(adj)
      print(f'{arm:6s} n={len(probs)} raw {r[:,2].mean():.2f}±{r[:,2].std(ddof=1) if len(r)>1 else 0:.2f} (Se {r[:,0].mean():.1f} Sp {r[:,1].mean():.1f}) '
            f'-> adj {a[:,2].mean():.2f}±{a[:,2].std(ddof=1) if len(a)>1 else 0:.2f} (Se {a[:,0].mean():.1f} Sp {a[:,1].mean():.1f}) | ens {e_raw[2]:.2f} -> {e_adj[2]:.2f}')


if __name__ == '__main__':
    main()

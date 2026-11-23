#!/usr/bin/env python3
"""Locked evaluation of AST + diffusion-classifier fusion (k fixed in advance).
Prints single-model mean ± SD, ensemble, per-device Score and a paired bootstrap
(ensemble fused minus ensemble AST) over evaluation cycles."""
import argparse
import numpy as np
from fuse_dclf import score, log_softmax

DEVICES = ['AKGC417L', 'LittC2SE', 'Meditron', 'Litt3200']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz', nargs='+')
    p.add_argument('--k', type=float, default=25.0)
    p.add_argument('--n_boot', type=int, default=2000)
    a = p.parse_args()
    for path in a.npz:
        d = np.load(path)
        lab, dev, err, probs = d['label'], d['device'], d['err'], d['ast_probs']
        g = log_softmax(-a.k * err[:, :, :4].mean(1) / err[:, :, 4:5].mean(1))
        la = np.log(probs + 1e-8)
        print(f'== {path}  (generator step {int(d["generator_step"])}, k={a.k}, N={len(lab)})')
        print('dclf alone          Se %.2f Sp %.2f Score %.2f' % score(g.argmax(1), lab))
        for name, fn in [('AST', lambda x: x), ('AST+dclf', lambda x: x + g)]:
            s = np.array([score(fn(x).argmax(1), lab) for x in la])
            print(f'{name:9s} single  Se {s[:,0].mean():.2f} Sp {s[:,1].mean():.2f} Score {s[:,2].mean():.2f} ± {s[:,2].std(ddof=1):.2f}   per seed {np.round(s[:,2],2).tolist()}')
        ens_a = np.log(probs.mean(0) + 1e-8)
        ens_f = ens_a + g
        for name, z in [('AST ens', ens_a), ('fused ens', ens_f)]:
            pr = z.argmax(1)
            row = ' '.join(f'{DEVICES[k][:6]} {score(pr[dev==k], lab[dev==k])[2]:.1f}' for k in range(4) if (dev == k).any())
            rec = ' '.join(f'{(pr[lab==c]==c).mean()*100:.1f}' for c in range(4))
            print(f'{name:9s} Se %.2f Sp %.2f Score %.2f' % score(pr, lab) + f' | {row} | recall N/C/W/B {rec}')
        rng = np.random.default_rng(20261118)
        pa, pf = ens_a.argmax(1), ens_f.argmax(1)
        diffs = []
        for _ in range(a.n_boot):
            i = rng.integers(0, len(lab), len(lab))
            diffs.append(score(pf[i], lab[i])[2] - score(pa[i], lab[i])[2])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f'paired bootstrap fused-minus-AST ensemble Score: {np.mean(diffs):+.2f} [{lo:+.2f}, {hi:+.2f}]')


if __name__ == '__main__':
    main()

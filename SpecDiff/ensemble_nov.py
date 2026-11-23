#!/usr/bin/env python3
"""Official-test ensembles of nov_final runs (probabilities in test-loader order).
Usage: ensemble_nov.py NAME=ARM:seeds[+ARM:seeds...] ...   e.g. M0x10=M0:1-10  M0+T=M0:1-5+T_K3:1-5
Prints Se/Sp/Score, per-device Score and a paired bootstrap against the first ensemble."""
import os
import sys
import numpy as np
from test_labels import test_order

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../artifacts/runs_nov')
SAVED = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results/saved/probs')


def run_dir(arm, seed):
    for d in (f'icbhi_ast_ce_nov_final_{arm}_seed{seed}', f'aft/icbhi_ast_ce_mixed/nov_final_{arm}_seed{seed}'):
        if os.path.exists(os.path.join(RUNS, d, 'probs_test_last.npy')):
            return os.path.join(RUNS, d)
    import glob
    hits = glob.glob(os.path.join(RUNS, '**', f'*nov_final_{arm}_seed{seed}', 'probs_test_last.npy'), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def load_probs(arm, seed):
    """Raw (uncorrected) last-epoch test probabilities of one run, or None.
    Prefers a freshly trained run under artifacts/runs_nov, then the shipped copy in results/saved/probs."""
    d = run_dir(arm, seed)
    if d and os.path.basename(d) in (f'icbhi_ast_ce_nov_final_{arm}_seed{seed}', f'nov_final_{arm}_seed{seed}'):
        return np.load(os.path.join(d, 'probs_test_last.npy'))
    # Shipped copies are used only when NO freshly trained run of this arm exists, so a partial re-run is never mixed
    # with the shipped seeds. IE643_NO_SAVED=1 disables the fallback altogether.
    if os.environ.get('IE643_NO_SAVED') == '1' or _has_fresh(arm):
        return None
    f = os.path.join(SAVED, f'{arm}_seed{seed}.npy')
    return np.load(f) if os.path.isfile(f) else None


def _has_fresh(arm):
    return any(os.path.isfile(os.path.join(RUNS, f'icbhi_ast_ce_nov_final_{arm}_seed{s}', 'probs_test_last.npy'))
               for s in range(1, 31))


def parse(spec):
    members = []
    for part in spec.split('+'):
        arm, seeds = part.rsplit(':', 1)
        lo, _, hi = seeds.partition('-')
        members += [(arm, s) for s in range(int(lo), int(hi or lo) + 1)]
    return members


def score(pred, lab):
    ab = lab > 0
    se, sp = (pred[ab] == lab[ab]).mean() * 100, (pred[~ab] == 0).mean() * 100
    return se, sp, (se + sp) / 2


def main():
    names, lab, dev = test_order()
    dev = np.array(dev)
    rng = np.random.default_rng(20261118)
    boot = rng.integers(0, len(lab), (2000, len(lab)))
    ref = None
    for arg in sys.argv[1:]:
        name, spec = arg.split('=', 1)
        mem = parse(spec)
        probs = [load_probs(a, s) for a, s in mem]
        missing = [m for m, p_ in zip(mem, probs) if p_ is None]
        if missing:
            print(f'{name}: missing {missing}')
            continue
        p = np.mean(probs, 0)
        pred = p.argmax(1)
        se, sp, sc = score(pred, lab)
        devs = ' '.join(f'{k[:6]} {score(pred[dev == k], lab[dev == k])[2]:.1f}' for k in ('AKGC417L', 'Meditron', 'Litt3200'))
        line = f'{name:14s} n={len(mem):2d} Se {se:.2f} Sp {sp:.2f} Score {sc:.2f} | {devs}'
        if ref is None:
            ref = pred
        else:
            d = np.array([score(pred[i], lab[i])[2] - score(ref[i], lab[i])[2] for i in boot])
            lo, hi = np.percentile(d, [2.5, 97.5])
            line += f' | vs first {sc - score(ref, lab)[2]:+.2f} [{lo:+.2f}, {hi:+.2f}] P(>0)={(d > 0).mean():.2f}'
        print(line, flush=True)


if __name__ == '__main__':
    main()

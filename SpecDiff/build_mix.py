#!/usr/bin/env python3
"""Build an --extra_manifest CSV (kind,source,index,label) for main.py.

--kind synth     takes generated log-mels from a SpecDiff pool
--kind real_dup  duplicates real training cycles (oversampling control)

--mode fill:<N>:<classes>  top the listed classes up to N items, e.g. fill:1000:2+3
--mode balance             top every class up to the largest real class count
--mode mult:<k>            add k times the real count of every class

Real counts come from --real_manifest, which must be the ACTIVE real training
set (train_minus_val_manifest.csv during selection, train_manifest.csv for the
final runs). Selection of pool rows is deterministic: the first rows of each
class that pass the optional probe filter.
"""
import argparse
import csv
import os

import numpy as np


def extra_counts(mode, real_counts):
    extra = [0, 0, 0, 0]
    if mode.startswith('fill:'):
        _, n, classes = mode.split(':')
        for c in (int(v) for v in classes.split('+')):
            extra[c] = max(0, int(n) - real_counts[c])
    elif mode == 'balance':
        top = max(real_counts)
        extra = [top - v for v in real_counts]
    elif mode.startswith('mult:'):
        k = float(mode.split(':')[1])
        extra = [int(round(k * v)) for v in real_counts]
    else:
        raise ValueError(mode)
    return extra


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', choices=['synth', 'real_dup'], required=True)
    p.add_argument('--mode', required=True)
    p.add_argument('--real_manifest', required=True)
    p.add_argument('--pool_dir', default=None)
    p.add_argument('--min_p_intended', type=float, default=0.0,
                   help='keep synthetic rows whose probe probability of the intended class is at least this')
    p.add_argument('--teacher_tag', default='',
                   help='attach soft labels <pool>/teacher_<tag>.npy (from teacher_probs.py) to synthetic rows')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    with open(args.real_manifest, newline='', encoding='utf-8') as f:
        real = list(csv.DictReader(f))
    real_counts = [sum(int(r['label']) == c for r in real) for c in range(4)]
    extra = extra_counts(args.mode, real_counts)
    rows = []
    if args.kind == 'real_dup':
        rng = np.random.default_rng(args.seed)
        for c in range(4):
            names = sorted(r['filename'] for r in real if int(r['label']) == c)
            order = []
            while len(order) < extra[c]:
                order.extend(rng.permutation(names).tolist())
            rows += [('real_dup', name, '', c) for name in order[:extra[c]]]
    else:
        pool_path = os.path.abspath(os.path.join(args.pool_dir, 'pool.npy'))
        with open(os.path.join(args.pool_dir, 'pool.csv'), newline='') as f:
            pool = list(csv.DictReader(f))
        keep = np.ones(len(pool), dtype=bool)
        if args.min_p_intended > 0:
            with open(os.path.join(args.pool_dir, 'probe.csv'), newline='') as f:
                probe = list(csv.DictReader(f))
            assert len(probe) == len(pool)
            keep = np.asarray([float(r['p_intended']) >= args.min_p_intended for r in probe])
        for c in range(4):
            candidates = [int(r['index']) for r, k in zip(pool, keep) if k and int(r['label']) == c]
            if len(candidates) < extra[c]:
                raise ValueError(f'class {c}: need {extra[c]} synthetic rows, pool has {len(candidates)} after filtering')
            rows += [('synth', pool_path, i, c) for i in candidates[:extra[c]]]
        if args.teacher_tag:
            teacher = os.path.abspath(os.path.join(args.pool_dir, f'teacher_{args.teacher_tag}.npy'))
            assert os.path.isfile(teacher), teacher
            rows = [r + (teacher,) for r in rows]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = f'{args.out}.{os.getpid()}.tmp'  # atomic replace: another job may be reading the same mix
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'source', 'index', 'label'] + (['teacher'] if args.teacher_tag else []))
        w.writerows(rows)
    os.replace(tmp, args.out)
    total = [real_counts[c] + extra[c] for c in range(4)]
    print(f'{args.out}: real={real_counts} extra={extra} total={total} kind={args.kind} mode={args.mode}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Soft labels for synthetic pools from an ensemble of real-only AST models.

For every pool directory, writes <pool>/teacher_<tag>.npy: float32 [N, 4], the
mean class probability over the given checkpoints, row-aligned with pool.npy.
The checkpoints must be trained on real data only, and only on the real
training set that is active in the phase where the labels are used
(train-minus-val models during selection, full-train models for final runs).
"""
import argparse
import json
import os

import numpy as np
import torch

from evaluate_synth import load_probe, run_probe


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool_dirs', nargs='+', required=True)
    p.add_argument('--ckpts', nargs='+', required=True)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    dev = torch.device('cuda')
    pools = [np.load(os.path.join(d, 'pool.npy'), mmap_mode='r') for d in args.pool_dirs]
    sums = [np.zeros((len(pool), 4), dtype=np.float64) for pool in pools]
    for ckpt in args.ckpts:
        model = load_probe(ckpt, dev)
        for acc, pool in zip(sums, pools):
            acc += run_probe(model, pool, dev)[1]
        del model
        torch.cuda.empty_cache()
        print(f'done {ckpt}', flush=True)
    for d, acc in zip(args.pool_dirs, sums):
        probs = (acc / len(args.ckpts)).astype(np.float32)
        np.save(os.path.join(d, f'teacher_{args.tag}.npy'), probs)
        with open(os.path.join(d, f'teacher_{args.tag}.json'), 'w') as f:
            json.dump(dict(ckpts=args.ckpts, n=len(probs),
                           pred_hist=np.bincount(probs.argmax(1), minlength=4).tolist(),
                           mean_max_prob=float(probs.max(1).mean())), f, indent=2)
        print(f'{d}: teacher_{args.tag}.npy pred hist {np.bincount(probs.argmax(1), minlength=4).tolist()}')


if __name__ == '__main__':
    main()

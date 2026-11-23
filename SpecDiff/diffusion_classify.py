#!/usr/bin/env python3
"""Use the conditional diffusion model as a generative classifier of REAL cycles.

For every real cycle and every class c, the model's denoising error is measured
when it is told the cycle is class c (true stethoscope and duration, which are
known at inference time). A lower error means the cycle is more likely under
class c. The same noise is used for all classes of a cycle (paired comparison).

Output npz: filename, label, device, t [T], err [N, T, 5] (v-prediction MSE for
classes 0..3 and for the null class), and ast_probs [K, N, 4] for the given
real-only AST checkpoints on the same cycles.

--ref val  : validation cycles (the generator must be G_sel, which never saw them)
--ref test : official test cycles (final, locked evaluation only)
"""
import argparse
import csv
import os

import numpy as np
import torch

from diffusion import alpha_sigma
from evaluate_synth import load_probe, run_probe
from model import N_CLASSES, SpecUNet
from train import to_model_space


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ref', choices=['val', 'test'], required=True)
    p.add_argument('--train_cache', default=None)
    p.add_argument('--val_manifest', default=None)
    p.add_argument('--test_cache', default=None)
    p.add_argument('--ast_ckpts', nargs='*', default=[])
    p.add_argument('--n_t', type=int, default=24)
    p.add_argument('--n_noise', type=int, default=2)
    p.add_argument('--batch', type=int, default=24)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    dev = torch.device('cuda')

    if args.ref == 'val':
        data = np.load(args.train_cache, allow_pickle=False)
        with open(args.val_manifest, newline='', encoding='utf-8') as f:
            names = {row['filename'] for row in csv.DictReader(f)}
        keep = np.asarray([n in names for n in data['filename']])
        assert keep.sum() == len(names)
    else:
        data = np.load(args.test_cache, allow_pickle=False)
        keep = np.ones(len(data['label']), dtype=bool)
    fbank, label, device, duration, filename = (data[k][keep] for k in ('fbank', 'label', 'device', 'duration', 'filename'))
    n = len(label)

    ck = torch.load(args.ckpt, map_location='cpu')
    model = SpecUNet(**ck['model_config']).to(dev).eval()
    model.load_state_dict(ck['ema'])
    x_all = to_model_space(torch.from_numpy(fbank).float(), ck['mean'], ck['std'])
    ts = torch.linspace(0.04, 0.96, args.n_t)
    n_cond = N_CLASSES + 1
    err = np.zeros((n, args.n_t, n_cond), dtype=np.float64)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    cls_all = torch.arange(n_cond, device=dev)
    with torch.no_grad():
        for start in range(0, n, args.batch):
            sl = slice(start, min(start + args.batch, n))
            b = sl.stop - sl.start
            x0 = x_all[sl].to(dev)
            d = torch.from_numpy(device[sl]).long().to(dev)
            du = torch.from_numpy(np.minimum(duration[sl], 5.0) / 5.0).float().to(dev)
            for ti, t in enumerate(ts):
                a, s = alpha_sigma(t.to(dev))
                for _ in range(args.n_noise):
                    eps = torch.randn(x0.shape, device=dev, generator=gen)
                    xt = a * x0 + s * eps
                    v = a * eps - s * x0
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = model(xt.repeat_interleave(n_cond, 0), t.to(dev).expand(b * n_cond),
                                     cls_all.repeat(b), d.repeat_interleave(n_cond, 0), du.repeat_interleave(n_cond, 0))
                    e = ((pred.float() - v.repeat_interleave(n_cond, 0)) ** 2).mean(dim=(1, 2, 3)).view(b, n_cond)
                    err[sl, ti] += e.cpu().numpy() / args.n_noise
            print(f'{sl.stop}/{n}', flush=True)

    ast = []
    for path in args.ast_ckpts:
        m = load_probe(path, dev)
        ast.append(run_probe(m, fbank, dev)[1])
        del m
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, filename=filename, label=label, device=device, t=ts.numpy(), err=err,
             ast_probs=np.asarray(ast, dtype=np.float32), generator_step=ck['step'])
    pred = err[:, :, :N_CLASSES].mean(1).argmin(1)
    ab = label > 0
    se, sp = (pred[ab] == label[ab]).mean() * 100, (pred[~ab] == 0).mean() * 100
    print(f'diffusion classifier alone ({args.ref}): Se {se:.2f} Sp {sp:.2f} Score {(se + sp) / 2:.2f}')


if __name__ == '__main__':
    main()

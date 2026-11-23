#!/usr/bin/env python3
"""Test-time use of the generator: re-generate each REAL evaluation cycle with
the diffusion model before it is classified (no labels are used).

Each cycle is noised to diffusion time t0 and denoised with the class set to
"unknown" (null class, no guidance), its true duration, and a stethoscope
chosen by the policy:
  akg  : the dominant training stethoscope (AKGC417L) -> stethoscope normalisation
  same : the cycle's own stethoscope                  -> plain generative test-time augmentation
The real-only AST checkpoints then classify the original and re-generated cycles.

Output npz: filename, label, device, configs (strings "t0|policy"),
ast_orig [K, N, 4], ast_tta [C, V, K, N, 4].
"""
import argparse
import csv
import os

import numpy as np
import torch

from diffusion import ddim_sample
from evaluate_synth import load_probe, run_probe
from model import DEVICES, N_CLASSES, SpecUNet
from train import T_PAD, to_model_space


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--ref', choices=['val', 'test'], required=True)
    p.add_argument('--train_cache', default=None)
    p.add_argument('--val_manifest', default=None)
    p.add_argument('--test_cache', default=None)
    p.add_argument('--ast_ckpts', nargs='+', required=True)
    p.add_argument('--t0', nargs='+', type=float, default=[0.2, 0.35, 0.5])
    p.add_argument('--policies', nargs='+', default=['akg', 'same'])
    p.add_argument('--n_var', type=int, default=2)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--batch', type=int, default=128)
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
    mean, std = ck['mean'], ck['std']
    x_all = to_model_space(torch.from_numpy(fbank).float(), mean, std)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    akg = DEVICES.index('AKGC417L')

    configs, regenerated = [], []
    for t0 in args.t0:
        for policy in args.policies:
            variants = []
            for _ in range(args.n_var):
                out = np.zeros((n, 498, 128), dtype=np.float32)
                for start in range(0, n, args.batch):
                    sl = slice(start, min(start + args.batch, n))
                    b = sl.stop - sl.start
                    d = torch.from_numpy(device[sl]).long().to(dev)
                    if policy == 'akg':
                        d = torch.full_like(d, akg)
                    x = ddim_sample(
                        model, (b, 1, 128, T_PAD), torch.full((b,), N_CLASSES, device=dev, dtype=torch.long), d,
                        torch.from_numpy(np.minimum(duration[sl], 5.0) / 5.0).float().to(dev),
                        steps=max(1, int(round(args.steps * t0))), guidance=1.0, generator=gen,
                        x_init=x_all[sl].to(dev), t_start=t0)
                    out[sl] = (x[:, 0, :, :498].transpose(1, 2) * (2.0 * std) + mean).cpu().numpy()
                variants.append(out)
            configs.append(f'{t0}|{policy}')
            regenerated.append(variants)
            print(f'regenerated {configs[-1]}', flush=True)
    del model
    torch.cuda.empty_cache()

    ast_orig = []
    ast_tta = np.zeros((len(configs), args.n_var, len(args.ast_ckpts), n, 4), dtype=np.float32)
    for k, path in enumerate(args.ast_ckpts):
        m = load_probe(path, dev)
        ast_orig.append(run_probe(m, fbank, dev)[1])
        for c, variants in enumerate(regenerated):
            for v, fb in enumerate(variants):
                ast_tta[c, v, k] = run_probe(m, fb, dev)[1]
        del m
        print(f'classified with {path}', flush=True)
    ast_orig = np.asarray(ast_orig, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, filename=filename, label=label, device=device, configs=np.asarray(configs),
             ast_orig=ast_orig, ast_tta=ast_tta, generator_step=ck['step'])

    def score(prob):
        pred = prob.argmax(1)
        ab = label > 0
        se, sp = (pred[ab] == label[ab]).mean() * 100, (pred[~ab] == 0).mean() * 100
        return (se + sp) / 2

    base = np.mean([score(pk) for pk in ast_orig])
    print(f'original cycles: mean single-model Score {base:.2f}')
    for c, name in enumerate(configs):
        only = np.mean([score(ast_tta[c, :, k].mean(0)) for k in range(len(args.ast_ckpts))])
        both = np.mean([score(0.5 * ast_orig[k] + 0.5 * ast_tta[c, :, k].mean(0)) for k in range(len(args.ast_ckpts))])
        print(f'{name:12s} regenerated only {only:.2f}   original+regenerated {both:.2f}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Sample a synthetic log-mel pool from a trained SpecDiff checkpoint.

Conditioning for each sample of class c:
  empirical : (stethoscope, duration) copied from a random real training cycle
              of class c (keeps the real class/device joint distribution).
  balanced  : duration from a random real training cycle of class c, but the
              stethoscope drawn uniformly from the three well-represented
              devices (counterfactual class/device combinations).

  other     : stethoscope drawn uniformly from the devices that differ from
              the source cycle's own device.

--translate_t T (0 < T < 1) switches from pure generation to partial
re-generation of real training cycles: each picked real cycle is noised to
diffusion time T and denoised under its own class and duration and the
stethoscope chosen by --device_policy. With policy 'other' this is a
label-preserving stethoscope translation of a real cycle.

Output: <out_dir>/pool.npy  float16 [N, 498, 128] in AST-normalised units
        <out_dir>/pool.csv  index,label,device,duration,policy
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from diffusion import ddim_sample
from model import DEVICES, SpecUNet
from train import T_PAD, load_training_arrays, to_model_space


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--cache', required=True)
    p.add_argument('--exclude_manifest', default=None)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--per_class', default='1500,1500,1500,1500')
    p.add_argument('--device_policy', choices=['empirical', 'balanced', 'other', 'target'], default='empirical')
    p.add_argument('--translate_t', type=float, default=0.0)
    p.add_argument('--target_cache', default=None,
                   help="policy 'target': stethoscopes drawn from this cache's device histogram (labels not read)")
    p.add_argument('--target_include', default=None, help='manifest selecting rows of --target_cache')
    p.add_argument('--guidance', type=float, default=2.0)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=100)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device('cuda')

    ck = torch.load(args.ckpt, map_location='cpu')
    model = SpecUNet(**ck['model_config']).to(dev).eval()
    model.load_state_dict(ck['ema'])
    mean, std = ck['mean'], ck['std']

    arrays = load_training_arrays(args.cache, args.exclude_manifest)
    rng = np.random.default_rng(args.seed)
    if args.device_policy == 'target':
        from train import load_unlabelled_arrays
        tdev = load_unlabelled_arrays(args.target_cache, args.target_include)['device']
        target_p = np.bincount(tdev, minlength=4) / len(tdev)
        print(f'target device distribution {np.round(target_p, 3).tolist()}', flush=True)
    labels, devices, durations, sources = [], [], [], []
    for c, count in enumerate(int(v) for v in args.per_class.split(',')):
        rows = np.flatnonzero(arrays['label'] == c)
        pick = rng.choice(rows, size=count, replace=True)
        labels.append(np.full(count, c))
        durations.append(arrays['duration'][pick])
        sources.append(pick)
        if args.device_policy == 'empirical':
            devices.append(arrays['device'][pick])
        elif args.device_policy == 'balanced':
            devices.append(rng.integers(0, 4, size=count))
        elif args.device_policy == 'target':
            devices.append(rng.choice(4, size=count, p=target_p))
        else:
            devices.append((arrays['device'][pick] + rng.integers(1, 4, size=count)) % 4)
    labels, devices, durations, sources = (np.concatenate(v) for v in (labels, devices, durations, sources))
    n = len(labels)

    gen = torch.Generator(device=dev).manual_seed(args.seed)
    out = np.zeros((n, 498, 128), dtype=np.float16)
    tic = time.time()
    for start in range(0, n, args.batch_size):
        sl = slice(start, min(start + args.batch_size, n))
        b = sl.stop - sl.start
        x_init, steps = None, args.steps
        if args.translate_t > 0:
            x_init = to_model_space(torch.from_numpy(arrays['fbank'][sources[sl]]).float(), mean, std).to(dev)
            steps = max(1, int(round(args.steps * args.translate_t)))
        x = ddim_sample(
            model, (b, 1, 128, T_PAD),
            torch.from_numpy(labels[sl]).to(dev),
            torch.from_numpy(devices[sl]).to(dev),
            torch.from_numpy(np.minimum(durations[sl], 5.0) / 5.0).float().to(dev),
            steps=steps, guidance=args.guidance, generator=gen,
            x_init=x_init, t_start=args.translate_t if x_init is not None else 1.0)
        fb = x[:, 0, :, :498].transpose(1, 2) * (2.0 * std) + mean
        out[sl] = fb.cpu().numpy().astype(np.float16)
        print(f'{sl.stop}/{n} samples, {(time.time() - tic) / sl.stop:.3f} s/sample', flush=True)

    np.save(os.path.join(args.out_dir, 'pool.npy'), out)
    with open(os.path.join(args.out_dir, 'pool.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'label', 'device', 'duration', 'policy', 'source_device', 'translate_t'])
        for i in range(n):
            w.writerow([i, int(labels[i]), DEVICES[int(devices[i])], f'{durations[i]:.4f}', args.device_policy,
                        DEVICES[int(arrays['device'][sources[i]])], args.translate_t])
    with open(os.path.join(args.out_dir, 'sample_args.json'), 'w') as f:
        json.dump(dict(vars(args), generator_step=ck['step']), f, indent=2)
    print(f'wrote {n} samples to {args.out_dir}')


if __name__ == '__main__':
    main()

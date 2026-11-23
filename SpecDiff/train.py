#!/usr/bin/env python3
"""Train the conditional spectrogram diffusion model on official-TRAIN cycles
(optionally minus the patient-grouped validation subset). The official test
set is never read here."""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F

from diffusion import training_loss
from model import N_CLASSES, SpecUNet

T_PAD = 512


def load_training_arrays(cache_npz, exclude_manifest=None):
    data = np.load(cache_npz, allow_pickle=False)
    keep = np.ones(len(data['label']), dtype=bool)
    if exclude_manifest:
        with open(exclude_manifest, newline='', encoding='utf-8') as f:
            excluded = {row['filename'] for row in csv.DictReader(f)}
        keep = np.asarray([name not in excluded for name in data['filename']])
        assert (~keep).sum() == len(excluded), 'validation manifest does not match the cache'
    return {k: data[k][keep] for k in ('fbank', 'label', 'device', 'duration')}


def load_unlabelled_arrays(cache_npz, include_manifest=None):
    """Transductive option: cycles used WITHOUT labels (class is always null).
    The label array of the cache is never read."""
    data = np.load(cache_npz, allow_pickle=False)
    keep = np.ones(len(data['device']), dtype=bool)
    if include_manifest:
        with open(include_manifest, newline='', encoding='utf-8') as f:
            wanted = {row['filename'] for row in csv.DictReader(f)}
        keep = np.asarray([name in wanted for name in data['filename']])
        assert keep.sum() == len(wanted), 'include manifest does not match the cache'
    return {k: data[k][keep] for k in ('fbank', 'device', 'duration')}


def to_model_space(fbank, mean, std):
    """[N, 498, 128] AST units -> [N, 1, 128, 512] diffusion units (std ~0.5)."""
    x = (fbank - mean) / (2.0 * std)
    x = x.transpose(1, 2).unsqueeze(1)
    return F.pad(x, (0, T_PAD - x.shape[-1], 0, 0), mode='replicate')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', required=True)
    p.add_argument('--exclude_manifest', default=None)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--steps', type=int, default=60000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--ema', type=float, default=0.9999)
    p.add_argument('--save_every', type=int, default=10000)
    p.add_argument('--resume_every', type=int, default=2000)
    p.add_argument('--gain_aug', type=float, default=0.07)
    p.add_argument('--base', type=int, default=96)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--shift_aug', type=int, default=0,
                   help='1 = random circular time shift of the 498 real frames (cycles are repeat-padded)')
    p.add_argument('--p_drop_class', type=float, default=0.15)
    p.add_argument('--cls_margin_w', type=float, default=0.0,
                   help='weight of a hinge term that asks for a lower denoising error under the true class than under a random wrong class')
    p.add_argument('--cls_margin', type=float, default=0.02)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--unlabelled_cache', default=None,
                   help='transductive: extra cycles trained with the null class (labels never read)')
    p.add_argument('--unlabelled_include', default=None,
                   help='manifest selecting the unlabelled rows of --unlabelled_cache (e.g. the validation patients)')
    p.add_argument('--unlabelled_frac', type=float, default=0.3,
                   help='fraction of each batch drawn from the unlabelled cycles')
    p.add_argument('--init_ckpt', default=None,
                   help='start model and EMA from this ema-*.pt (keeps its normalisation); ignored when resuming')
    p.add_argument('--time_budget_min', type=float, default=0,
                   help='stop cleanly (resumable) after this many minutes; 0 = no limit')
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from gpu_check import require_gpu
    require_gpu('SpecDiff generator training')
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device('cuda')
    start_time = time.time()

    arrays = load_training_arrays(args.cache, args.exclude_manifest)
    fb = torch.from_numpy(arrays['fbank'])
    mean, std = float(fb.mean()), float(fb.std())
    init = torch.load(args.init_ckpt, map_location='cpu') if args.init_ckpt else None
    if init is not None:
        mean, std = init['mean'], init['std']
    x_all = to_model_space(fb, mean, std).to(dev)
    cls_all = torch.from_numpy(arrays['label']).to(dev)
    dev_all = torch.from_numpy(arrays['device']).to(dev)
    dur_all = torch.from_numpy(np.minimum(arrays['duration'], 5.0) / 5.0).float().to(dev)
    n = x_all.shape[0]
    n_u = 0
    if args.unlabelled_cache:
        ua = load_unlabelled_arrays(args.unlabelled_cache, args.unlabelled_include)
        xu_all = to_model_space(torch.from_numpy(ua['fbank']), mean, std).to(dev)
        devu_all = torch.from_numpy(ua['device']).to(dev)
        duru_all = torch.from_numpy(np.minimum(ua['duration'], 5.0) / 5.0).float().to(dev)
        n_u = xu_all.shape[0]
        print(f'unlabelled cycles={n_u} devices={np.bincount(ua["device"], minlength=4).tolist()}', flush=True)
    print(f'training cycles={n} class counts={np.bincount(arrays["label"]).tolist()} mean={mean:.4f} std={std:.4f}', flush=True)

    if init is not None:
        cfg = dict(init['model_config'])
        cfg['dropout'] = args.dropout
        model = SpecUNet(**cfg).to(dev)
        model.load_state_dict(init['ema'])
    else:
        model = SpecUNet(base=args.base, dropout=args.dropout).to(dev)
    ema = deepcopy(model).eval().requires_grad_(False)
    print(f'parameters={sum(q.numel() for q in model.parameters()) / 1e6:.1f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    step = 0
    resume_path = os.path.join(args.out_dir, 'resume.pt')
    if os.path.isfile(resume_path):
        ck = torch.load(resume_path, map_location=dev)
        model.load_state_dict(ck['model'])
        ema.load_state_dict(ck['ema'])
        opt.load_state_dict(ck['opt'])
        step = ck['step']
        print(f'resumed from step {step}', flush=True)
    meta = dict(mean=mean, std=std, model_config=model.config, n_train=n, n_unlabelled=n_u,
                exclude_manifest=args.exclude_manifest, args=vars(args))
    with open(os.path.join(args.out_dir, 'train_args.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    def save(path, full):
        payload = dict(ema=ema.state_dict(), step=step, mean=mean, std=std, model_config=model.config)
        if full:
            payload.update(model=model.state_dict(), opt=opt.state_dict())
        torch.save(payload, path + '.tmp')
        os.replace(path + '.tmp', path)

    log = open(os.path.join(args.out_dir, 'loss.csv'), 'a')
    running, tic = [], time.time()
    perm, pos = torch.randperm(n, device=dev), 0
    model.train()
    b_u = int(round(args.batch_size * args.unlabelled_frac)) if n_u else 0
    b_l = args.batch_size - b_u
    while step < args.steps:
        if pos + b_l > n:
            perm, pos = torch.randperm(n, device=dev), 0
        idx = perm[pos:pos + b_l]
        pos += b_l
        x0, c_b, d_b, u_b = x_all[idx], cls_all[idx], dev_all[idx], dur_all[idx]
        if b_u:
            iu = torch.randint(0, n_u, (b_u,), device=dev)
            x0 = torch.cat([x0, xu_all[iu]])
            c_b = torch.cat([c_b, torch.full((b_u,), N_CLASSES, device=dev, dtype=c_b.dtype)])
            d_b = torch.cat([d_b, devu_all[iu]])
            u_b = torch.cat([u_b, duru_all[iu]])
        if args.shift_aug:
            shift = torch.randint(0, 498, (x0.shape[0], 1), device=dev)
            col = (torch.arange(498, device=dev)[None, :] + shift) % 498
            x0 = torch.gather(x0, 3, col[:, None, None, :].expand(-1, 1, x0.shape[2], -1))
            x0 = F.pad(x0, (0, T_PAD - 498, 0, 0), mode='replicate')
        if args.gain_aug > 0:
            x0 = x0 + (torch.rand(x0.shape[0], 1, 1, 1, device=dev) * 2 - 1) * args.gain_aug
        for group in opt.param_groups:
            group['lr'] = args.lr * min(1.0, (step + 1) / args.warmup)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = training_loss(model, x0, c_b, d_b, u_b,
                                 p_drop_class=args.p_drop_class,
                                 cls_margin_w=args.cls_margin_w, cls_margin=args.cls_margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        decay = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        running.append(loss.item())
        if step % 200 == 0:
            rate = 200 / (time.time() - tic)
            tic = time.time()
            print(f'step {step} loss {np.mean(running):.5f} {rate:.2f} it/s', flush=True)
            log.write(f'{step},{np.mean(running):.6f}\n')
            log.flush()
            running = []
        if step % args.save_every == 0 or step == args.steps:
            save(os.path.join(args.out_dir, f'ema-{step}.pt'), full=False)
        if step % args.resume_every == 0 or step == args.steps:
            save(resume_path, full=True)
        if args.time_budget_min and (time.time() - start_time) / 60 > args.time_budget_min:
            save(resume_path, full=True)
            print(f'time budget reached at step {step}; resumable', flush=True)
            break
    print(f'finished at step {step}', flush=True)


if __name__ == '__main__':
    main()

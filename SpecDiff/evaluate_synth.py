#!/usr/bin/env python3
"""Quality gate for a synthetic log-mel pool, using a real-only AST probe.

Reported per class:
  intended-class accuracy of the probe on synthetic samples, next to the
    probe's recall on real HELD-OUT cycles (the realistic ceiling);
  KID (unbiased polynomial-kernel MMD^2) between synthetic and held-out real
    probe features, next to the same KID for real-train vs real-held-out;
  memorisation ratio: nearest-train-neighbour distance of synthetic samples
    divided by that of real held-out samples (about 1 = as novel as unseen
    real data, near 0 = copies of training cycles).

Also writes per-sample probe outputs (pred, p_intended, nn_dist) to
<pool_dir>/probe.csv so mixes can filter on them.

The reference split must be the validation subset during selection. The
official test set may be used only for the final, locked diagnostic.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'Classification'))
from models.ast import ASTModel  # noqa: E402

CLASSES = ['normal', 'crackle', 'wheeze', 'both']


def load_probe(path, dev):
    state = torch.load(path, map_location='cpu', weights_only=False)
    cwd = os.getcwd()
    os.chdir(HERE.parent / 'Classification')
    model = ASTModel(label_dim=4, input_fdim=498, input_tdim=128, imagenet_pretrain=False,
                     audioset_pretrain=False, verbose=False)
    os.chdir(cwd)
    model_state = {k.replace('module.', ''): v for k, v in state['model'].items()}
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    assert not [k for k in missing if not k.startswith(('mlp_head', 'domain_mlp_head'))], missing
    model.mlp_head.load_state_dict(state['classifier'])
    return model.to(dev).eval()


@torch.no_grad()
def run_probe(model, fbank, dev, batch=128):
    feats, probs = [], []
    for start in range(0, len(fbank), batch):
        x = torch.from_numpy(np.asarray(fbank[start:start + batch], dtype=np.float32)).unsqueeze(1).to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            f = model(x)
            logits = model.mlp_head(f)
        feats.append(f.float().cpu())
        probs.append(torch.softmax(logits.float(), dim=-1).cpu())
    return torch.cat(feats).numpy().astype(np.float64), torch.cat(probs).numpy()


def kid(x, y):
    d = x.shape[1]
    k = lambda a, b: (a @ b.T / d + 1.0) ** 3
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    m, n = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def nn_dist(query, base):
    q = query / np.linalg.norm(query, axis=1, keepdims=True)
    b = base / np.linalg.norm(base, axis=1, keepdims=True)
    return 1.0 - (q @ b.T).max(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool_dir', required=True)
    p.add_argument('--probe', required=True)
    p.add_argument('--train_cache', required=True)
    p.add_argument('--ref', choices=['val', 'test'], required=True)
    p.add_argument('--val_manifest', default=None)
    p.add_argument('--test_cache', default=None)
    args = p.parse_args()
    dev = torch.device('cuda')
    model = load_probe(args.probe, dev)

    train = np.load(args.train_cache, allow_pickle=False)
    if args.ref == 'val':
        with open(args.val_manifest, newline='', encoding='utf-8') as f:
            val_names = {row['filename'] for row in csv.DictReader(f)}
        is_ref = np.asarray([name in val_names for name in train['filename']])
        ref_fb, ref_y = train['fbank'][is_ref], train['label'][is_ref]
        base_fb, base_y = train['fbank'][~is_ref], train['label'][~is_ref]
    else:
        test = np.load(args.test_cache, allow_pickle=False)
        ref_fb, ref_y = test['fbank'], test['label']
        base_fb, base_y = train['fbank'], train['label']

    pool = np.load(os.path.join(args.pool_dir, 'pool.npy'), mmap_mode='r')
    with open(os.path.join(args.pool_dir, 'pool.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    syn_y = np.asarray([int(r['label']) for r in rows])
    syn_dev = np.asarray([r['device'] for r in rows])

    syn_f, syn_p = run_probe(model, pool, dev)
    ref_f, ref_p = run_probe(model, ref_fb, dev)
    base_f, _ = run_probe(model, base_fb, dev)
    syn_pred, ref_pred = syn_p.argmax(1), ref_p.argmax(1)
    syn_nn = nn_dist(syn_f, base_f)
    ref_nn = nn_dist(ref_f, base_f)

    report = {'pool': args.pool_dir, 'probe': args.probe, 'ref': args.ref, 'classes': {}}
    print(f'{"class":8s} {"n":>5s} {"syn_acc":>8s} {"real_ref_recall":>15s} {"KID_syn":>9s} {"KID_real":>9s} {"nn_ratio":>8s}')
    for c, name in enumerate(CLASSES):
        s, r, b = syn_y == c, ref_y == c, base_y == c
        entry = dict(
            n=int(s.sum()),
            syn_intended_acc=float((syn_pred[s] == c).mean()),
            real_ref_recall=float((ref_pred[r] == c).mean()),
            kid_syn_vs_ref=kid(syn_f[s], ref_f[r]),
            kid_realtrain_vs_ref=kid(base_f[b], ref_f[r]),
            nn_ratio=float(np.median(syn_nn[s]) / np.median(ref_nn[r])),
            syn_pred_hist=np.bincount(syn_pred[s], minlength=4).tolist(),
            syn_acc_by_device={d: float((syn_pred[s & (syn_dev == d)] == c).mean())
                               for d in sorted(set(syn_dev[s]))},
        )
        report['classes'][name] = entry
        print(f'{name:8s} {entry["n"]:5d} {100 * entry["syn_intended_acc"]:8.2f} {100 * entry["real_ref_recall"]:15.2f} '
              f'{entry["kid_syn_vs_ref"]:9.4f} {entry["kid_realtrain_vs_ref"]:9.4f} {entry["nn_ratio"]:8.3f}')
        print(f'         pred hist {entry["syn_pred_hist"]}  by device {entry["syn_acc_by_device"]}')
    report['syn_mean_intended_acc'] = float(np.mean([e['syn_intended_acc'] for e in report['classes'].values()]))
    report['mean_kid_syn'] = float(np.mean([e['kid_syn_vs_ref'] for e in report['classes'].values()]))
    print(f'mean intended acc {100 * report["syn_mean_intended_acc"]:.2f}  mean KID {report["mean_kid_syn"]:.4f}')

    with open(os.path.join(args.pool_dir, f'quality_{args.ref}.json'), 'w') as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(args.pool_dir, 'probe.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['index', 'label', 'pred', 'p_intended', 'nn_dist'])
        for i in range(len(syn_y)):
            w.writerow([i, int(syn_y[i]), int(syn_pred[i]), f'{syn_p[i, syn_y[i]]:.5f}', f'{syn_nn[i]:.5f}'])


if __name__ == '__main__':
    main()

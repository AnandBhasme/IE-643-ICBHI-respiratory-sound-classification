#!/usr/bin/env python3
"""Validation-subset probabilities of saved nov_sel runs (last.pth), for post-hoc analyses."""
import argparse, csv, os
import numpy as np
import torch
from evaluate_synth import load_probe, run_probe

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
p = argparse.ArgumentParser()
p.add_argument('--arms', nargs='+', required=True, help='ARM:nseeds')
p.add_argument('--out', required=True)
a = p.parse_args()
data = np.load(f'{ROOT}/artifacts/data/fbank_cache/train.npz')
with open(f'{ROOT}/artifacts/data/official_icbhi/splits/val_manifest.csv', newline='') as f:
    names = {r['filename'] for r in csv.DictReader(f)}
keep = np.asarray([n in names for n in data['filename']])
fb, lab, dev_ = data['fbank'][keep], data['label'][keep], data['device'][keep]
out = dict(label=lab, device=dev_)
gpu = torch.device('cuda')
for spec in a.arms:
    arm, n = spec.rsplit(':', 1)
    probs = []
    for s in range(1, int(n) + 1):
        path = f'{ROOT}/artifacts/runs_nov/icbhi_ast_ce_nov_sel_{arm}_seed{s}/last.pth'
        if os.path.exists(path):
            m = load_probe(path, gpu); probs.append(run_probe(m, fb, gpu)[1]); del m
    out[arm] = np.asarray(probs, dtype=np.float32)
    print(arm, len(probs), flush=True)
np.savez(a.out, **out)

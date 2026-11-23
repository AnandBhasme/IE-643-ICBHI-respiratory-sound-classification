#!/usr/bin/env python3
"""Build demo/assets from the cached training log-mels (artifacts/data/fbank_cache/train.npz).

Writes reference_logmels.npz (a few real cycles per class and stethoscope, float16, AST-normalised) and reference_meta.json,
plus time-averaged spectral profiles (mean / SD over ALL training cycles per class and stethoscope).
Deterministic: cycles are taken from different recordings, evenly spaced through the sorted file names, duration >= 2 s.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'SpecDiff'))
from model import CLASSES, DEVICES  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--cache', default=str(ROOT / 'artifacts/data/fbank_cache/train.npz'))
ap.add_argument('--out', default=str(ROOT / 'demo/assets'))
ap.add_argument('--per_group', type=int, default=3)
a = ap.parse_args()
d = np.load(a.cache, allow_pickle=False)
fb, y, dv, dur, fn = d['fbank'], d['label'], d['device'], d['duration'], d['filename']
sel, meta = [], []
mean = np.zeros((4, 4, 128), np.float32)
sd = np.zeros((4, 4, 128), np.float32)
for c in range(4):
    for k in range(4):
        rows = np.flatnonzero((y == c) & (dv == k))
        if len(rows) == 0:
            continue
        prof = fb[rows].mean(1)                       # (n, 128): time average per cycle
        mean[c, k], sd[c, k] = prof.mean(0), prof.std(0)
        good = [r for r in sorted(rows, key=lambda r: fn[r]) if dur[r] >= 2.0] or sorted(rows, key=lambda r: fn[r])
        seen, picks = set(), []
        step = max(1, len(good) // a.per_group)
        for r in good[::step]:
            rec = str(fn[r]).split('_')[0]
            if rec not in seen:
                seen.add(rec)
                picks.append(r)
            if len(picks) == a.per_group:
                break
        for r in picks:
            sel.append(fb[r].astype(np.float16))
            meta.append(dict(label=int(c), device=DEVICES[k], filename=str(fn[r]), duration=float(dur[r])))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / 'reference_logmels.npz', fbank=np.stack(sel), profile_mean=mean, profile_sd=sd)
json.dump(meta, open(out / 'reference_meta.json', 'w'), indent=1)
print(f'{len(meta)} reference cycles: ' + ', '.join(f'{CLASSES[c]}={sum(m["label"] == c for m in meta)}' for c in range(4)))

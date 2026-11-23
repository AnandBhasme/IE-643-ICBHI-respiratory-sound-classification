#!/usr/bin/env python3
"""Package the demo weights from finished runs into ./weights (fp16, CPU-loadable) and write weights/SHA256SUMS.

  classifier_M0_seed1.pth      real-only AST (artifacts/runs_nov/icbhi_ast_ce_nov_final_M0_seed1/last.pth)
  classifier_T_K3_seed1.pth    final arm (…_T_K3_seed1/last.pth)
  specdiff_generator.pt        non-transductive generator, EMA weights at 50k steps (artifacts/specdiff/gen_full/ema-50000.pt)
Upload the three files as assets of a GitHub release (they exceed the 100 MB file limit of a normal commit), then put the URLs
into weights/weights_manifest.json (scripts/download_weights.py reads it).
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'artifacts'
OUT = ROOT / 'weights'
OUT.mkdir(exist_ok=True)


def half(sd):
    return {k: (v.half() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


jobs = {}
for arm in ('M0', 'T_K3'):
    src = A / f'runs_nov/icbhi_ast_ce_nov_final_{arm}_seed1/last.pth'
    ck = torch.load(src, map_location='cpu', weights_only=False)
    torch.save({'model': half(ck['model']), 'classifier': half(ck['classifier'])}, OUT / f'classifier_{arm}_seed1.pth')
    jobs[f'classifier_{arm}_seed1.pth'] = str(src.relative_to(ROOT))
src = A / 'specdiff/gen_full/ema-50000.pt'
ck = torch.load(src, map_location='cpu', weights_only=False)
torch.save({'ema': half(ck['ema']), 'step': ck['step'], 'mean': ck['mean'], 'std': ck['std'], 'model_config': ck['model_config']},
           OUT / 'specdiff_generator.pt')
jobs['specdiff_generator.pt'] = str(src.relative_to(ROOT))
sums = {n: sha(OUT / n) for n in jobs}
(OUT / 'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k, v in sums.items()))
manifest = {n: dict(sha256=sums[n], url='', source=jobs[n], bytes=(OUT / n).stat().st_size) for n in jobs}
(OUT / 'weights_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))

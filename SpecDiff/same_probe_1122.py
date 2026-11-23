#!/usr/bin/env python3
"""Task C (22 Nov): score the old DiffWave clips and a SpecDiff pool with the SAME probes.

Diagnostic only; nothing selected. Probes: C0 seed 1 last.pth (used for the SpecDiff gate) and
E1 seed 1 best.pth (used for the old DiffWave gate). Both loaded with evaluate_synth.load_probe/run_probe.

Preprocessing of DiffWave wavs:
  cache_fbank : exactly cache_fbank.py's path for prepared cycles = repeat-pad/crop to 5 s (as the training
                cycles were prepared: util.icbhi_util.cut_pad_sample_torchaudio, pad_types='repeat') then
                util.icbhi_util.generate_fbank (Kaldi fbank, AST normalisation) -> (498, 128).
  orig_gate   : inference/predict.preprocess (fbank on the raw clip, bilinear resize to 498 x 128), i.e. what the
                original v1/v2 gate used; reported only as a reproduction check of the 6.0% / 2.4% numbers.
intended_acc: % of clips whose probe argmax is the intended class.
probe_real_test_recall: probe recall of that class on the official-test fbank cache.
"""
import argparse, csv, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / 'Classification'))
sys.path.insert(0, str(HERE.parent / 'Classification' / 'inference'))
from evaluate_synth import load_probe, run_probe  # noqa: E402
from util.icbhi_util import generate_fbank, cut_pad_sample_torchaudio  # noqa: E402
from predict import preprocess as predict_preprocess  # noqa: E402

CLASSES = ['normal', 'crackle', 'wheeze', 'both']
PADARGS = SimpleNamespace(sample_rate=16000, desired_length=5, pad_types='repeat')


def fb_cache(path):
    audio, sr = sf.read(str(path), dtype='float32')
    if audio.ndim == 2:
        audio = audio.mean(1)
    assert sr == 16000, (path, sr)
    x = cut_pad_sample_torchaudio(torch.from_numpy(audio).float().unsqueeze(0), PADARGS)
    fb = generate_fbank(SimpleNamespace(model='ast'), x, sr, n_mels=128)
    assert fb.shape == (498, 128, 1), fb.shape
    return fb[..., 0]


def fb_orig(path):
    return predict_preprocess(str(path))[0, 0].numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    R = Path(a.root); os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda')
    probes = {'C0_seed1_last': R / 'artifacts/runs/icbhi_ast_ce_wp3_C0_seed1_bf16/last.pth',
              'E1_seed1_best': R / 'artifacts/runs/icbhi_ast_ce_E1_seed1_official/best.pth'}
    gens = {'DiffWave_v1_20k': R / 'artifacts/synth/v1/raw', 'DiffWave_v2_33868': R / 'artifacts/diag/v2/synth_pool'}
    te = np.load(R / 'artifacts/data/fbank_cache/test.npz', allow_pickle=False)
    feats = {}
    for g, root in gens.items():
        for c in (2, 3):
            paths = sorted(p for p in root.rglob('*.wav') if f'class{c}' in p.parts)
            assert paths, (g, c)
            feats[(g, c)] = {'cache_fbank': np.stack([fb_cache(p) for p in paths]),
                             'orig_gate': np.stack([fb_orig(p) for p in paths])}
            print(g, c, len(paths), flush=True)
    pool = np.load(R / 'artifacts/specdiff/pools/full_s50000_w3.5_empirical/pool.npy', mmap_mode='r')
    with open(R / 'artifacts/specdiff/pools/full_s50000_w3.5_empirical/pool.csv', newline='') as f:
        py = np.array([int(r['label']) for r in csv.DictReader(f)])
    rows = []
    for pn, pp in probes.items():
        model = load_probe(str(pp), dev)
        _, pt = run_probe(model, te['fbank'], dev)
        rec = {c: float((pt.argmax(1)[te['label'] == c] == c).mean() * 100) for c in range(4)}
        print(pn, 'real test recall', rec, flush=True)
        for (g, c), d in feats.items():
            for mode, x in d.items():
                _, p = run_probe(model, x, dev)
                rows.append(dict(generator=f'{g}[{mode}]', probe=pn, cls=CLASSES[c], n=len(x),
                                 intended_acc=round(float((p.argmax(1) == c).mean() * 100), 2),
                                 probe_real_test_recall=round(rec[c], 2)))
        _, p = run_probe(model, pool, dev)
        for c in range(4):
            m = py == c
            rows.append(dict(generator='SpecDiff_full_s50000_w3.5_empirical', probe=pn, cls=CLASSES[c], n=int(m.sum()),
                             intended_acc=round(float((p.argmax(1)[m] == c).mean() * 100), 2),
                             probe_real_test_recall=round(rec[c], 2)))
        del model; torch.cuda.empty_cache()
    with open(os.path.join(a.out, 'gate_same_probe.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    print('done ->', a.out)


if __name__ == '__main__':
    main()

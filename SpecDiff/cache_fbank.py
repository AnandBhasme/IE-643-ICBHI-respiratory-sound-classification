#!/usr/bin/env python3
"""Cache the exact AST log-mel inputs (Kaldi fbank, AST normalisation) for the
prepared 5 s cycles, with class / stethoscope / duration metadata.

Output: <out_dir>/{train,test}.npz with
  fbank  float32 [N, 498, 128]   (time, mel) in AST-normalised units
  label, device int64 [N]; duration float32 [N] (seconds, uncropped)
  filename, patient str [N]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'Classification'))
from model import DEVICES  # noqa: E402
from util.icbhi_util import generate_fbank  # noqa: E402


class _Args:
    model = 'ast'


def build(manifest, wav_dir):
    with open(manifest, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    fbanks = []
    for row in rows:
        audio, sr = sf.read(os.path.join(wav_dir, row['filename']), dtype='float32')
        assert sr == 16000 and audio.ndim == 1, (row['filename'], sr, audio.shape)
        # Mirror ICBHIDataset: sf.read (float64 there) -> float tensor [1, n].
        fb = generate_fbank(_Args, torch.from_numpy(audio).float().unsqueeze(0), sr, n_mels=128)
        assert fb.shape == (498, 128, 1), fb.shape
        fbanks.append(fb[..., 0])
    return dict(
        fbank=np.stack(fbanks).astype(np.float32),
        label=np.asarray([int(r['label']) for r in rows], dtype=np.int64),
        device=np.asarray([DEVICES.index(r['device']) for r in rows], dtype=np.int64),
        duration=np.asarray([float(r['duration']) for r in rows], dtype=np.float32),
        filename=np.asarray([r['filename'] for r in rows]),
        patient=np.asarray([r['patient'] for r in rows]),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for split, sub in (('train', 'training'), ('test', 'test')):
        data = build(os.path.join(args.data_folder, 'splits', f'{split}_manifest.csv'),
                     os.path.join(args.data_folder, sub, 'real'))
        np.savez(os.path.join(args.out_dir, f'{split}.npz'), **data)
        fb = data['fbank']
        print(f'{split}: n={len(fb)} mean={fb.mean():.4f} std={fb.std():.4f} min={fb.min():.3f} max={fb.max():.3f} '
              f'labels={np.bincount(data["label"]).tolist()} devices={np.bincount(data["device"], minlength=4).tolist()}')


if __name__ == '__main__':
    main()

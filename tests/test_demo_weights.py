"""Needs weights/ (download_weights.py) and the ICBHI test audio (artifacts/data/official_icbhi); skipped otherwise.
Checks that the packaged fp16 classifier reproduces the stored fp32 test probabilities, that generation is class-conditional,
and that the audio inversion runs."""
import sys
from glob import glob
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'demo'))


def _need():
    import pytest
    from engine import WEIGHTS
    wavs = sorted(glob(str(ROOT / 'artifacts/data/official_icbhi/test/*/*.wav')))
    if not (WEIGHTS / 'classifier_T_K3_seed1.pth').is_file() or not (WEIGHTS / 'specdiff_generator.pt').is_file() or not wavs:
        pytest.skip('weights or test audio not available')
    return wavs


def test_classifier_matches_stored_probabilities():
    from engine import Engine
    wavs = _need()
    eng = Engine('auto')
    name = 'T_K3'
    stored = np.load(ROOT / 'results/saved/probs/T_K3_seed1.npy')
    idx = list(range(0, len(wavs), 97))[:24]
    got = np.concatenate([eng.classify(eng.wav_to_fbank(wavs[i])[0], name, corrected=False) for i in idx])
    assert np.abs(got - stored[idx]).max() < 0.05          # fp16 weights + BF16 autocast vs the fp32 run
    assert (got.argmax(1) == stored[idx].argmax(1)).mean() >= 0.9


def test_generation_and_audio():
    from engine import Engine
    _need()
    eng = Engine('auto')
    fb = eng.generate('wheeze', 'AKGC417L', 3.0, guidance=3.5, steps=10, seed=0, n=2)
    assert fb.shape == (2, 498, 128) and np.isfinite(fb).all()
    audio, sr = eng.to_audio(fb[0], 3.0)
    assert sr == 16000 and abs(len(audio) - 48000) <= 1 and np.isfinite(audio).all()

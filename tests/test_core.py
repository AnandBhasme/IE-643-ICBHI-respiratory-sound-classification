"""CPU unit tests (no data, no GPU). Run:  python -m pytest tests -q   (or: python tests/test_core.py)"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ('SpecDiff', 'Classification'):
    sys.path.insert(0, str(ROOT / sub))


def test_balance_counts():
    from build_mix import extra_counts
    assert extra_counts('balance', [2063, 1215, 501, 363]) == [0, 848, 1562, 1700]
    assert extra_counts('fill:1300:2+3', [2063, 1215, 501, 363]) == [0, 0, 799, 937]
    assert extra_counts('mult:1', [2, 3, 4, 5]) == [2, 3, 4, 5]


def test_prior_weights():
    from prior_correct import mix_prior, pr_real
    w = pr_real / mix_prior('T_K3')
    assert np.allclose(mix_prior('T_K3'), 0.25)                      # balanced training mix
    assert np.allclose(w, pr_real / 0.25) and abs(pr_real.sum() - 1) < 1e-9
    assert w[0] > w[3]                                                # normal is up-weighted, 'both' down-weighted


def test_saved_probabilities_reproduce_headline():
    """No training: the shipped last-epoch probabilities give the numbers reported in the paper."""
    os.environ.setdefault('IE643_NO_SAVED', '0')
    from ensemble_nov import load_probs, score
    from prior_correct import mix_prior, pr_real
    from test_labels import test_order
    _, lab, _ = test_order()
    assert len(lab) == 2756
    m0 = [score(load_probs('M0', s).argmax(1), lab)[2] for s in range(1, 11)]
    w = pr_real / mix_prior('T_K3')
    tk = [score((load_probs('T_K3', s) * w).argmax(1), lab)[2] for s in range(1, 11)]
    assert abs(np.mean(m0) - 58.46) < 0.01 and abs(np.std(m0, ddof=1) - 0.96) < 0.01
    assert abs(np.mean(tk) - 59.57) < 0.01 and abs(np.std(tk, ddof=1) - 0.64) < 0.01
    assert sum(a > b for a, b in zip(tk, m0)) == 8


def test_unet_and_sampler_cpu():
    from diffusion import ddim_sample, training_loss
    from model import SpecUNet
    torch.manual_seed(0)
    net = SpecUNet(base=32, mults=(1, 2), num_res=1, attn_levels=(1,)).eval()
    x = torch.randn(2, 1, 128, 512)
    cls, dev, dur = torch.tensor([2, 3]), torch.tensor([0, 2]), torch.tensor([0.6, 1.0])
    loss = training_loss(net.train(), x, cls, dev, dur)
    assert torch.isfinite(loss)
    out = ddim_sample(net.eval(), (2, 1, 128, 512), cls, dev, dur, steps=2, guidance=2.0, autocast_dtype=None)
    assert out.shape == (2, 1, 128, 512) and torch.isfinite(out).all()


def test_audio_roundtrip():
    """log-mel -> audio -> log-mel stays close to the original (the audio is an approximate inversion)."""
    from audio_synth import audio_to_fbank, log_mel_to_audio
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 3) / 16000
    wav = (0.3 * np.sin(2 * np.pi * 400 * t) * (1 + np.sin(2 * np.pi * 1.2 * t)) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    fb = audio_to_fbank(wav)                      # (498, 128), repeat-padded to 5 s
    audio, sr = log_mel_to_audio(fb, n_iter=48)
    fb2 = audio_to_fbank(audio, sr)
    corr = np.corrcoef(fb.ravel(), fb2.ravel())[0, 1]
    assert sr == 16000 and audio.shape[0] > 70000 and np.isfinite(audio).all()
    assert corr > 0.9, corr


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)

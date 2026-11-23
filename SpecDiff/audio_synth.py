"""Approximate audio from an AST log-mel (illustration only).

The generator works on the classifier's input representation (Kaldi fbank, 128 mel bins, 25 ms Hann window, 10 ms shift,
AST normalisation), so there is no waveform. To listen to a sample we invert the representation:
  1. undo the AST normalisation -> natural-log mel energies of the pre-emphasised signal's power spectrum;
  2. estimate the power spectrum with multiplicative non-negative updates through the Kaldi mel filterbank;
  3. Griffin-Lim phase reconstruction (hop 160, window 400, FFT 512);
  4. undo the 0.97 pre-emphasis and peak-normalise.
Phase is not recovered, so the result sounds rough ("phasey"); it is meant for listening and for the round-trip check in
tests/test_audio_roundtrip.py, not as a vocoder. It is NOT the DiffWave vocoder of the first pipeline.
"""
import numpy as np
import torch
import torchaudio
from scipy.signal import lfilter

AST_MEAN, AST_STD = -4.2677393, 4.5689974
SR, N_FFT, WIN, HOP, N_MELS, PREEMPH = 16000, 512, 400, 160, 128, 0.97


def kaldi_mel_matrix():
    """(128, 257) Kaldi mel filterbank with the same frequency limits as generate_fbank."""
    bins, _ = torchaudio.compliance.kaldi.get_mel_banks(N_MELS, N_FFT, float(SR), 20.0, 0.0, 100.0, -500.0, 1.0)
    return torch.nn.functional.pad(bins, (0, 1)).float()


def log_mel_to_power(fbank_norm, n_iter=60):
    """fbank_norm (frames, 128) in AST-normalised units -> power spectrogram (257, frames)."""
    fb = torch.as_tensor(np.asarray(fbank_norm, dtype=np.float32)) * (2.0 * AST_STD) + AST_MEAN
    energy = fb.exp().T.clamp(min=1e-10)                      # (128, T)
    m = kaldi_mel_matrix()                                     # (128, 257)
    s = (m.T @ energy) / m.sum(0, keepdim=True).T.clamp(min=1e-8)
    s = s.clamp(min=1e-10)
    denom = m.T @ torch.ones_like(energy)
    for _ in range(n_iter):
        s = s * (m.T @ (energy / (m @ s).clamp(min=1e-10))) / denom.clamp(min=1e-10)
    return s


def log_mel_to_audio(fbank_norm, duration_s=None, n_iter=64, seed=0):
    power = log_mel_to_power(fbank_norm)
    mag = power.sqrt()
    window = torch.hann_window(WIN)
    n_samples = (mag.shape[1] - 1) * HOP
    torch.manual_seed(seed)
    wav = torchaudio.functional.griffinlim(mag, window, N_FFT, HOP, WIN, 1.0, n_iter, 0.99, n_samples, True)
    y = lfilter([1.0], [1.0, -PREEMPH], wav.numpy().astype(np.float64)).astype(np.float32)
    if duration_s is not None:
        y = y[:int(round(duration_s * SR))]
    peak = float(np.abs(y).max())
    return (0.9 * y / peak if peak > 0 else y), SR


def audio_to_fbank(audio, sr=SR):
    """Same computation as util.icbhi_util.generate_fbank for a mono float array (pads/crops to 5 s)."""
    x = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
    if sr != SR:
        x = torchaudio.functional.resample(x, sr, SR)
    target = 5 * SR
    if x.shape[-1] > target:
        x = x[..., :target]
    else:
        ratio = int(np.ceil(target / x.shape[-1]))
        x = x.repeat(1, ratio)[..., :target]
    fb = torchaudio.compliance.kaldi.fbank(x, htk_compat=True, sample_frequency=SR, use_energy=False, window_type='hanning',
                                           num_mel_bins=N_MELS, dither=0.0, frame_shift=10)
    return ((fb - AST_MEAN) / (AST_STD * 2)).numpy()

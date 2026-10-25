"""FFT-bank Morlet scalograms for the pre-registered ICBHI WP4 view."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def morlet_filterbank(n: int, sample_rate: int = 16_000, n_scales: int = 128,
                      f_min: float = 50.0, f_max: float = 4_000.0,
                      w0: float = 6.0, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return an analytic, positive-frequency Morlet bank with shape (F, n)."""
    dev = torch.device(device)
    omega = 2.0 * math.pi * torch.fft.fftfreq(n, d=1.0 / sample_rate, device=dev)
    frequencies = torch.logspace(math.log10(f_min), math.log10(f_max), n_scales,
                                 device=dev, dtype=torch.float32)
    scales = w0 / (2.0 * math.pi * frequencies)
    scaled = scales[:, None] * omega[None, :] - w0
    bank = math.pi ** -0.25 * torch.exp(-0.5 * scaled.square())
    bank = bank * (omega[None, :] > 0).to(bank.dtype)
    bank = bank * torch.sqrt(2.0 * math.pi * scales[:, None] * sample_rate)
    return bank


@torch.no_grad()
def scalogram(waveform: torch.Tensor, filterbank: torch.Tensor, sample_rate: int = 16_000,
              hop: int = 160, frames: int = 498) -> torch.Tensor:
    """Compute log-power CWT output as a float32 tensor in (T, F) order."""
    wav = torch.as_tensor(waveform, dtype=torch.float32, device=filterbank.device).reshape(-1)
    n = filterbank.shape[-1]
    if wav.numel() > n:
        wav = wav[:n]
    elif wav.numel() < n:
        if wav.numel() == 0:
            raise ValueError("Cannot compute a scalogram from an empty waveform")
        wav = wav.repeat(math.ceil(n / wav.numel()))[:n]
    spectrum = torch.fft.fft(wav)
    energy = torch.fft.ifft(spectrum[None, :] * filterbank, dim=-1).abs().square()
    pooled = F.avg_pool1d(energy.unsqueeze(0), kernel_size=hop, stride=hop).squeeze(0)
    logged = 10.0 * torch.log10(pooled.clamp_min(1e-10))
    if logged.shape[-1] < frames:
        logged = F.pad(logged, (0, frames - logged.shape[-1]), mode="replicate")
    return logged[:, :frames].transpose(0, 1).contiguous().float()

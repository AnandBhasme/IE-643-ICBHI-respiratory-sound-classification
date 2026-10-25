"""Offline inference for one ICBHI respiratory-cycle clip.

The input should be one cycle. Full-recording segmentation is intentionally left
to the caller because it requires the ICBHI annotation file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from models.ast import ASTModel
from util.scalogram import morlet_filterbank, scalogram


CLASSES = ("normal", "crackle", "wheeze", "both")
SAMPLE_RATE = 16_000
DESIRED_LENGTH_SEC = 5
FBANK_MEAN = -4.2677393
FBANK_STD = 4.5689974
_model = None
_device = None
_model_view = None
_checkpoint_key = None


def _load_audio(wav_or_array: Union[str, Path, np.ndarray], sample_rate: int | None):
    if isinstance(wav_or_array, (str, Path)):
        audio, sr = sf.read(str(wav_or_array), dtype="float32", always_2d=False)
    else:
        if sample_rate is None:
            raise ValueError("sample_rate is required when input is an array")
        audio, sr = np.asarray(wav_or_array, dtype=np.float32), int(sample_rate)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError(f"Expected a non-empty mono or multichannel waveform, got {audio.shape}")
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    return waveform


def _fixed_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Repeat-pad/crop to the five-second CWT cache convention."""
    target = SAMPLE_RATE * DESIRED_LENGTH_SEC
    if waveform.shape[-1] > target:
        return waveform[..., :target]
    repeat_count = math.ceil(target / waveform.shape[-1])
    return waveform.repeat(1, repeat_count)[..., :target]


def preprocess(wav_or_array, sample_rate: int | None = None, view: str = "mel",
               scalo_stats_json: str | Path | None = None) -> torch.Tensor:
    """Return normalized AST features with shape (1, channels, 498, 128)."""
    if view not in ("mel", "scalo", "both"):
        raise ValueError(f"Unsupported input view: {view}")
    waveform = _load_audio(wav_or_array, sample_rate)
    channels = []
    if view in ("mel", "both"):
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=SAMPLE_RATE,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10,
        )
        # Match training: compute fbank on the original cycle, then resize the
        # time/frequency grid to (498, 128) in the dataset transform.
        fbank = (fbank - FBANK_MEAN) / (FBANK_STD * 2)
        fbank = fbank.unsqueeze(0).unsqueeze(0)
        if fbank.shape[-2:] != (498, 128):
            fbank = F.interpolate(fbank, size=(498, 128), mode="bilinear",
                                  align_corners=False, antialias=True)
        channels.append(fbank)
    if view in ("scalo", "both"):
        if scalo_stats_json is None:
            raise ValueError("scalo/both inference requires the cache stats.json used in training")
        import json
        stats = json.loads(Path(scalo_stats_json).read_text(encoding="utf-8"))
        bank = morlet_filterbank(80_000, sample_rate=SAMPLE_RATE, n_scales=128,
                                 f_min=50.0, f_max=4_000.0, w0=6.0, device=waveform.device)
        fixed_waveform = _fixed_waveform(waveform)
        feature = scalogram(fixed_waveform[0], bank, sample_rate=SAMPLE_RATE, hop=160, frames=498)
        feature = (feature - float(stats["train_mean_db"])) / max(float(stats["train_std_db"]), 1e-8)
        channels.append(feature.unsqueeze(0).unsqueeze(0))
    return torch.cat(channels, dim=1)


def load_model(checkpoint: Union[str, Path], device: str = "cpu", view: str = "mel"):
    """Load an AST encoder and its trained four-class head from a repo checkpoint."""
    global _model, _device, _model_view, _checkpoint_key
    target = torch.device(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = ASTModel(
        label_dim=4,
        input_fdim=498,
        input_tdim=128,
        imagenet_pretrain=False,
        audioset_pretrain=False,
        verbose=False,
    )
    if view == "both":
        old = model.v.patch_embed.proj
        new = torch.nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                              stride=old.stride, padding=old.padding, dilation=old.dilation,
                              groups=old.groups, bias=old.bias is not None,
                              padding_mode=old.padding_mode)
        with torch.no_grad():
            new.weight[:, 0].copy_(old.weight[:, 0])
            new.weight[:, 1].zero_()
            if old.bias is not None:
                new.bias.copy_(old.bias)
        model.v.patch_embed.proj = new
    model_state = state["model"]
    classifier_state = state["classifier"]
    if model_state and next(iter(model_state)).startswith("module."):
        model_state = {key.removeprefix("module."): value for key, value in model_state.items()}
    model.load_state_dict(model_state)
    classifier = model.mlp_head
    classifier.load_state_dict(classifier_state)
    model = model.to(target).eval()
    classifier = classifier.to(target).eval()
    _model = (model, classifier)
    _device = target
    _model_view = view
    _checkpoint_key = str(Path(checkpoint).resolve())
    return _model


def predict(wav_or_array, checkpoint: Union[str, Path], device: str = "cpu", sample_rate: int | None = None,
            view: str = "mel", scalo_stats_json: str | Path | None = None):
    """Return class probabilities for one cycle WAV or one waveform array."""
    global _model, _device, _model_view, _checkpoint_key
    requested_checkpoint = str(Path(checkpoint).resolve())
    if (_model is None or _device != torch.device(device) or _model_view != view
            or _checkpoint_key != requested_checkpoint):
        load_model(checkpoint, device=device, view=view)
    features = preprocess(wav_or_array, sample_rate=sample_rate, view=view,
                          scalo_stats_json=scalo_stats_json).to(_device)
    with torch.inference_mode():
        logits = _model[1](_model[0](features))
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
    return {name: float(probability) for name, probability in zip(CLASSES, probabilities)}


def predict_ensemble(wavs, checkpoints, device: str = "cpu", sample_rate: int | None = None,
                     view: str = "mel", scalo_stats_json: str | Path | None = None):
    """Average probabilities from checkpoints while keeping only one AST in memory."""
    if not checkpoints:
        raise ValueError("At least one checkpoint is required")
    if view != "mel":
        raise ValueError("Streaming ensemble inference currently supports the mel view")
    target = torch.device(device)
    features = [preprocess(wav, sample_rate=sample_rate, view=view,
                            scalo_stats_json=scalo_stats_json).to(target) for wav in wavs]
    totals = torch.zeros((len(features), len(CLASSES)), device=target)
    for checkpoint in checkpoints:
        model, classifier = load_model(checkpoint, device=device, view=view)
        with torch.inference_mode():
            for index, feature in enumerate(features):
                totals[index] += torch.softmax(classifier(model(feature)), dim=-1)[0]
        del model, classifier
        if target.type == "cuda":
            torch.cuda.empty_cache()
    totals /= len(checkpoints)
    return [{name: float(probability) for name, probability in zip(CLASSES, row)}
            for row in totals.cpu().tolist()]

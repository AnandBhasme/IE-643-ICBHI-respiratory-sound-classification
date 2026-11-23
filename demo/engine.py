"""Model access for the demo and the command line: classifiers, generator, reference spectrograms, plots.

Weights are looked up in <repo>/weights (override with IE643_WEIGHTS). Missing files give an explicit message that points
to `python scripts/download_weights.py`.
"""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for sub in ('SpecDiff', 'Classification'):
    sys.path.insert(0, str(ROOT / sub))

from audio_synth import log_mel_to_audio  # noqa: E402
from diffusion import ddim_sample  # noqa: E402
from evaluate_synth import load_probe  # noqa: E402
from model import CLASSES, DEVICES, SpecUNet  # noqa: E402
from prior_correct import mix_prior, pr_real  # noqa: E402
from catalog import BY_ID, MODELS, installed, member_paths, missing_hint  # noqa: E402
from util.icbhi_util import cut_pad_sample_torchaudio, generate_fbank  # noqa: E402

WEIGHTS = Path(os.environ.get('IE643_WEIGHTS', ROOT / 'weights'))
ASSETS = HERE / 'assets'
T_PAD = 512
GENERATOR_FILE = 'specdiff_generator.pt'
DEVICE_NAMES = {'AKGC417L': 'AKG C417L microphone', 'Meditron': 'Meditron stethoscope', 'Litt3200': 'Littmann 3200 stethoscope',
                'LittC2SE': 'Littmann C2 SE (rare)'}


def torch_device(choice='auto'):
    if choice == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(choice)


class Engine:
    def __init__(self, device='auto'):
        self.device = torch_device(device)
        self._single, self._gen = None, None
        refs = np.load(ASSETS / 'reference_logmels.npz', allow_pickle=False)
        self.ref = {k: refs[k] for k in refs.files}
        self.ref_meta = json.load(open(ASSETS / 'reference_meta.json'))

    # ------------------------------------------------------------------ weights
    @staticmethod
    def available_classifiers():
        """Ids of the catalogue models whose checkpoint files are present, in catalogue order."""
        return [m['id'] for m in MODELS if installed(m)]

    @staticmethod
    def classifier_ids():
        return [m['id'] for m in MODELS]

    @staticmethod
    def generator_available():
        return (WEIGHTS / GENERATOR_FILE).is_file()

    def free_memory(self):
        """Release every cached model and the generator, and return the CUDA cache. Returns a status line."""
        self._single = None
        self._gen = None
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return self.memory_status()

    def memory_status(self):
        if self.device.type != 'cuda':
            return 'running on CPU'
        mib = 1024 ** 2
        return (f'GPU memory: {torch.cuda.memory_allocated() / mib:.0f} MiB allocated, '
                f'{torch.cuda.memory_reserved() / mib:.0f} MiB reserved, peak {torch.cuda.max_memory_allocated() / mib:.0f} MiB')

    def generator(self):
        if self._gen is None:
            f = WEIGHTS / GENERATOR_FILE
            if not f.is_file():
                raise self._missing(f.name)
            ck = torch.load(f, map_location='cpu', weights_only=False)
            model = SpecUNet(**ck['model_config'])
            model.load_state_dict({k: v.float() for k, v in ck['ema'].items()})
            self._gen = (model.to(self.device).eval(), ck['mean'], ck['std'])
        return self._gen

    # ------------------------------------------------------------------ classification
    @staticmethod
    def wav_to_fbank(path):
        """Exactly the training preprocessing: mono 16 kHz, repeat-pad / crop to 5 s with fade, Kaldi fbank, AST norm."""
        audio, sr = sf.read(str(path), dtype='float32', always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(1)
        x = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
        if sr != 16000:
            x = torchaudio.functional.resample(x, sr, 16000)
        x = cut_pad_sample_torchaudio(x, SimpleNamespace(sample_rate=16000, desired_length=5, pad_types='repeat'))
        fb = generate_fbank(SimpleNamespace(model='ast'), x, 16000, n_mels=128)
        return fb[..., 0], x.shape[-1] / 16000.0

    def _single_model(self, path):
        """One single model stays in memory at a time (the last one used)."""
        if self._single is not None and self._single[0] == path:
            return self._single[1]
        self._single = None
        gc.collect()
        model = load_probe(str(path), self.device)
        self._single = (path, model)
        return model

    @staticmethod
    def _probs(model, fb, device):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            feats = model(fb.unsqueeze(1).to(device))
            logits = model.mlp_head(feats)
        return torch.softmax(logits.float(), -1).cpu().numpy()

    @torch.no_grad()
    def classify(self, fbank, mid, corrected=True):
        """fbank (498, 128) or (N, 498, 128) -> probabilities (N, 4) for catalogue model `mid`.

        Single models keep one checkpoint in memory. Ensembles load one member at a time, average the softmax outputs
        with equal weight and free each member before loading the next, so memory use stays at one AST.
        Prior correction (p * pi_real / pi_mix) applies only to models trained with a balanced synthetic mix.
        """
        m = BY_ID[mid]
        if not installed(m):
            raise FileNotFoundError(missing_hint(m))
        fb = torch.as_tensor(np.asarray(fbank, dtype=np.float32))
        if fb.dim() == 2:
            fb = fb[None]
        paths = member_paths(m)
        if m['kind'] == 'single':
            p = self._probs(self._single_model(paths[0]), fb, self.device)
        else:
            self._single = None
            gc.collect()
            total = 0.0
            for path in paths:
                model = load_probe(str(path), self.device)
                total = total + self._probs(model, fb, self.device)
                del model
                gc.collect()
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            p = total / len(paths)
        arm = m.get('arm')
        if corrected and arm not in (None, 'M0'):
            p = p * (pr_real / mix_prior(arm))
            p = p / p.sum(1, keepdims=True)
        return p

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, cls, device, duration_s, guidance=3.5, steps=50, seed=0, n=1):
        model, mean, std = self.generator()
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        ci = torch.full((n,), CLASSES.index(cls), device=self.device)
        di = torch.full((n,), DEVICES.index(device), device=self.device)
        du = torch.full((n,), min(float(duration_s), 5.0) / 5.0, device=self.device)
        x = ddim_sample(model, (n, 1, 128, T_PAD), ci, di, du, steps=int(steps), guidance=float(guidance), generator=gen,
                        autocast_dtype=torch.bfloat16 if self.device.type == 'cuda' else None)
        fb = x[:, 0, :, :498].transpose(1, 2) * (2.0 * std) + mean
        return fb.cpu().numpy().astype(np.float32)

    @staticmethod
    def to_audio(fbank, duration_s):
        return log_mel_to_audio(fbank, duration_s=duration_s)

    # ------------------------------------------------------------------ real references
    def references(self, cls, device=None):
        """Real training log-mels shipped with the demo for (class, stethoscope)."""
        keep = [i for i, m in enumerate(self.ref_meta) if m['label'] == CLASSES.index(cls) and (device is None or m['device'] == device)]
        return self.ref['fbank'][keep].astype(np.float32), [self.ref_meta[i] for i in keep]

    def nearest_real(self, fbank, cls, device):
        refs, meta = self.references(cls, device)
        if len(refs) == 0:
            refs, meta = self.references(cls)
        a = fbank.reshape(1, -1) / np.linalg.norm(fbank)
        b = refs.reshape(len(refs), -1)
        b = b / np.linalg.norm(b, axis=1, keepdims=True)
        i = int((a @ b.T).argmax())
        return refs[i], meta[i]

    def profile(self, cls, device):
        """Time-averaged real log-mel spectrum (mean and SD over training cycles) for (class, stethoscope)."""
        p = self.ref['profile_mean'][CLASSES.index(cls), DEVICES.index(device)]
        s = self.ref['profile_sd'][CLASSES.index(cls), DEVICES.index(device)]
        return p, s


# ---------------------------------------------------------------------- plots
def comparison_figure(syn, real, real_meta, profile, cls, device):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]), constrained_layout=True)
    vmin, vmax = np.percentile(np.concatenate([syn.ravel(), real.ravel()]), [1, 99])
    for a, img, title in ((ax[0], syn, f'Synthetic {cls} ({device})'),
                          (ax[1], real, f'Nearest real {CLASSES[real_meta["label"]]} ({real_meta["device"]})')):
        im = a.imshow(img.T, origin='lower', aspect='auto', cmap='magma', vmin=vmin, vmax=vmax)
        a.set_title(title, fontsize=9)
        a.set_xlabel('frame (10 ms)')
        a.set_ylabel('mel bin')
    fig.colorbar(im, ax=ax[:2], fraction=0.02, pad=0.01)
    diff = syn.mean(0) - real.mean(0)
    ax[2].imshow((syn - real).T, origin='lower', aspect='auto', cmap='coolwarm', vmin=-1.0, vmax=1.0)
    ax[2].set_title('Synthetic − real (frame by frame)', fontsize=9)
    ax[2].set_xlabel('frame (10 ms)')
    bins = np.arange(128)
    pm, ps = profile
    ax[3].fill_between(bins, pm - ps, pm + ps, color='#c9c8c3', label='real class/device mean ± SD')
    ax[3].plot(bins, syn.mean(0), color='#eb6834', label='synthetic')
    ax[3].plot(bins, real.mean(0), color='#2a78d6', lw=1, label='nearest real')
    ax[3].set_xlabel('mel bin')
    ax[3].set_ylabel('mean log-mel')
    ax[3].set_title('Spectral profile', fontsize=9)
    ax[3].legend(fontsize=6.5, frameon=False)
    return fig


def probability_figure(rows):
    """rows: list of (title, probs[4])."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(1.6 + 2.4 * len(rows), 2.8), constrained_layout=True)
    width = 0.8 / len(rows)
    colors = ['#2a78d6', '#eb6834', '#1baf7a', '#9a9893']
    for k, (title, p) in enumerate(rows):
        bars = ax.bar(np.arange(4) + k * width, p, width * 0.92, color=colors[k % 4], label=title)
        ax.bar_label(bars, fmt='%.2f', fontsize=6.5, padding=1)
    ax.set_xticks(np.arange(4) + width * (len(rows) - 1) / 2, CLASSES)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('probability')
    ax.legend(fontsize=7, frameon=False, loc='upper center', ncol=len(rows))
    return fig


def logmel_figure(fb, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3), constrained_layout=True)
    im = ax.imshow(fb.T, origin='lower', aspect='auto', cmap='magma')
    ax.set_xlabel('frame (10 ms)')
    ax.set_ylabel('mel bin')
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return fig

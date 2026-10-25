"""Gradio demo for one ICBHI respiratory-cycle WAV clip (CUDA or CPU)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = HERE.parent
for candidate in (
    ROOT / "Code/Code/AFT/Classification",
    REPO / "github/Classification",
    REPO / "Classification",
):
    if (candidate / "inference/predict.py").is_file():
        sys.path.insert(0, str(candidate))
        break
else:
    raise RuntimeError("Cannot locate Classification/inference/predict.py")

from inference.predict import CLASSES, predict, preprocess  # noqa: E402

torch.set_num_threads(max(1, int(os.environ.get("IE643_CPU_THREADS", "4"))))
DEFAULT_CHECKPOINT = Path(os.environ.get(
    "IE643_DEMO_CHECKPOINT", HERE / "models/C4_seed1_last.pth"
))


def resolve_device(choice: str | None = None) -> torch.device:
    """Choose CUDA when available, otherwise CPU; allow explicit override."""
    choice = (choice or os.environ.get("IE643_DEVICE", "auto")).lower()
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but PyTorch cannot see a CUDA GPU. Check the GPU driver and PyTorch install.")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError("Device must be one of: auto, cpu, cuda")


def predict_clip(wav_path: str | os.PathLike, checkpoint: str | os.PathLike | None = None,
                 device: str | None = None):
    """Return (probability plot, normalized log-mel plot, probability mapping)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise FileNotFoundError(wav_path)
    ckpt = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
    if not ckpt.is_file():
        raise FileNotFoundError(f"Demo checkpoint missing: {ckpt}. Set IE643_DEMO_CHECKPOINT.")
    target_device = resolve_device(device)
    probabilities = predict(wav_path, ckpt, device=str(target_device), view="mel")
    features = preprocess(wav_path, view="mel")[0, 0].cpu().numpy()

    fig_prob, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    vals = [probabilities[name] for name in CLASSES]
    bars = ax.bar(CLASSES, vals, color="#4776a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title(f"C4 seed 1 prediction ({target_device.type.upper()})")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    fig_mel, mx = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    im = mx.imshow(features.T, origin="lower", aspect="auto", cmap="magma")
    mx.set_xlabel("Frame")
    mx.set_ylabel("Mel bin")
    mx.set_title("Normalized log-mel input")
    fig_mel.colorbar(im, ax=mx, fraction=0.025, pad=0.02)
    return fig_prob, fig_mel, probabilities


def classify_upload(wav_path, device_choice):
    return predict_clip(wav_path, device=device_choice)


def build_demo():
    import gradio as gr
    with gr.Blocks(title="IE 643 respiratory-cycle demo") as demo:
        gr.Markdown("## ICBHI respiratory-cycle classifier\nUpload one respiratory-cycle WAV. Device `auto` uses an available CUDA GPU, otherwise CPU.")
        audio = gr.Audio(label="Cycle WAV", sources=["upload"], type="filepath")
        device = gr.Dropdown(["auto", "cuda", "cpu"], value="auto", label="Inference device")
        run = gr.Button("Classify", variant="primary")
        with gr.Row():
            probs = gr.Plot(label="Four-class probabilities")
            mel = gr.Plot(label="Mel spectrogram")
        details = gr.Label(label="Probabilities", num_top_classes=4)
        run.click(classify_upload, inputs=[audio, device], outputs=[probs, mel, details])
    return demo


if __name__ == "__main__":
    build_demo().launch()

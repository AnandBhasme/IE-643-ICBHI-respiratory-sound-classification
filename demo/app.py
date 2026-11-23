#!/usr/bin/env python3
"""Gradio demo: classify respiratory cycles with the project's models, and generate and compare synthetic cycles.

    python demo/app.py                       # http://127.0.0.1:7860 (CPU or GPU, chosen automatically)
    python demo/app.py --share               # also create a public Gradio link

Models come from demo/catalog.py. A model whose files are missing is listed as "not installed", and its card says what to install.
Research prototype for a course project. Not a medical device; do not use it for diagnosis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import gradio as gr
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from catalog import BY_ID, MODELS, installed  # noqa: E402
from engine import CLASSES, Engine, comparison_figure, logmel_figure, probability_figure  # noqa: E402

ROOT = HERE.parent
SAMPLES = json.loads((HERE / 'samples.json').read_text())
DATA_DIR = ROOT / 'artifacts/data/official_icbhi'


def sample_path(sample):
    """Where a preselected clip lives: the test or the training folder of the prepared data."""
    return DATA_DIR / sample['split'] / 'real' / sample['file']
DEVICE_CHOICES = ['AKGC417L', 'Meditron', 'Litt3200', 'LittC2SE']
DEFAULT_MODEL = 'T_K3'
DISCLAIMER = ('Research prototype for a course project (IE 643). Not a medical device. The best model reaches about 59.6 Score '
              '((Se + Sp) / 2) on the official ICBHI test split, so many cycles are misclassified.')

CSS = """
.ie-label { display: flex; align-items: center; gap: 6px; font-weight: 600; font-size: 0.92rem; margin: 10px 0 4px 0; cursor: help; }
.ie-q { display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px; border-radius: 50%;
        background: #3274d9; color: #fff; font-size: 11px; font-weight: 700; }
.ie-card { padding: 12px 16px; border-radius: 8px; border-left: 4px solid #3274d9; background: var(--background-fill-secondary);
           margin: 6px 0 10px 0; font-size: 0.95rem; line-height: 1.5; }
.ie-card .ie-meta { opacity: 0.75; font-size: 0.85rem; margin-top: 4px; }
.ie-section { font-size: 1.1rem; font-weight: 600; margin: 14px 0 4px 0; }
"""


def tip(text, help_text):
    """A field name that explains itself on hover. The native tooltip is on the name and on the ? badge."""
    safe = help_text.replace('"', '&quot;')
    return gr.HTML(f'<div class="ie-label" title="{safe}">{text} <span class="ie-q" title="{safe}">?</span></div>')


def header(runtime):
    return ('<div class="ie-banner"><h1>Respiratory sound classifier and synthetic cycle generator</h1>'
            '<p>ICBHI 2017 four-class classification (normal, crackle, wheeze, both) with the Audio Spectrogram Transformer, and '
            'SpecDiff, a diffusion model that generates the classifier\'s own input.</p>'
            f'<span class="ie-runtime">{runtime}</span></div>')


def model_card(mid):
    m = BY_ID[mid]
    if installed(m):
        status = 'installed'
    else:
        status = ('**not installed**. Run `python scripts/download_weights.py` for the final models, or set '
                  '`IE643_INTENSIVE_DIR` to the folder with the intensive-phase runs.')
    return (f'<div class="ie-card"><b>{m["label"]}</b><br>{m["desc"]}'
            f'<div class="ie-meta">Official Score (last epoch): {m["score"]} &nbsp;·&nbsp; Status: {status}</div></div>')


def runtime_text():
    if torch.cuda.is_available():
        gib = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        return f'Running on the GPU: {torch.cuda.get_device_name(0)}, {gib:.1f} GiB'
    return 'Running on the CPU (no CUDA GPU found)'


def build():
    engines: dict[str, Engine] = {}

    def engine_for(device):
        if device == 'cuda' and not torch.cuda.is_available():
            raise gr.Error('CUDA was selected, but PyTorch cannot see a CUDA GPU. Choose auto or cpu.')
        if device not in engines:
            engines[device] = Engine(device)
        return engines[device]

    def release_memory():
        lines = [e.free_memory() for e in engines.values()]
        return '**Memory released.** ' + (lines[0] if lines else 'Nothing was loaded.')

    model_choices = [(m['label'] + ('' if installed(m) else '  (not installed)'), m['id']) for m in MODELS]
    generator_ok = Engine.generator_available()
    sample_files = [(str(sample_path(s)), s) for s in SAMPLES if sample_path(s).is_file()]

    # ------------------------------------------------------------------ classify
    def classify_fn(wav, mid, device, corrected):
        if wav is None:
            raise gr.Error('Upload a WAV file, or click one of the preselected cycles below.')
        if not installed(BY_ID[mid]):
            raise gr.Error(f'{mid} is not installed. See its card for what to install.')
        eng = engine_for(device)
        t0 = time.time()
        fb, secs = eng.wav_to_fbank(wav)
        p = eng.classify(fb, mid, corrected)[0]
        status = (f'**Done.** {BY_ID[mid]["label"]} on {eng.device.type.upper()} in {time.time() - t0:.2f} s. '
                  f'Clip used: {min(secs, 5.0):.2f} s (padded or cut to 5 s, as in training). {eng.memory_status()}.')
        return ({CLASSES[i]: float(p[i]) for i in range(4)},
                probability_figure([(mid, p)]),
                logmel_figure(fb, 'Log-mel input (AST normalisation)'),
                status)

    def show_model(mid):
        return model_card(mid)

    def sample_table():
        head = '| File | True class | Stethoscope | Taken from |\n|---|---|---|---|\n'
        where = {'test': 'official test list', 'training': 'training recordings'}
        return head + ''.join(f'| `{s["file"]}` | {s["class_name"]} | {s["device"]} | {where[s["split"]]} |\n' for s in SAMPLES)

    # ------------------------------------------------------------------ generate and compare
    def generate_fn(cls, device_name, duration, guidance, steps, seed, mid, corrected):
        if not generator_ok:
            raise gr.Error('The SpecDiff generator is not installed. Run `python scripts/download_weights.py`.')
        if not installed(BY_ID[mid]):
            raise gr.Error(f'{mid} is not installed. Choose another classifier.')
        eng = engine_for('auto')
        t0 = time.time()
        syn = eng.generate(cls, device_name, duration, guidance, steps, seed, n=1)[0]
        real, meta = eng.nearest_real(syn, cls, device_name)
        fig = comparison_figure(syn, real, meta, eng.profile(cls, device_name), cls, device_name)
        audio, sr = eng.to_audio(syn, duration)
        p_syn = eng.classify(syn, mid, corrected)[0]
        p_real = eng.classify(real, mid, corrected)[0]
        pfig = probability_figure([(f'{mid}: synthetic', p_syn), (f'{mid}: nearest real', p_real)])
        verdict = 'recognised' if int(p_syn.argmax()) == CLASSES.index(cls) else 'not recognised'
        info = (f'**Generated in {time.time() - t0:.1f} s on {eng.device.type.upper()}.** Nearest real reference: '
                f'{CLASSES[meta["label"]]}, {meta["device"]}, recording {meta["filename"]}.\n\n'
                f'{mid} says the synthetic cycle is **{CLASSES[int(p_syn.argmax())]}** ({p_syn.max():.2f}); the requested class '
                f'{cls} is {verdict}. For the nearest real cycle it says **{CLASSES[int(p_real.argmax())]}** ({p_real.max():.2f}).')
        return fig, (sr, audio), pfig, info

    # ------------------------------------------------------------------ generator check
    def check_fn(n, guidance, steps, seed, mid, corrected, progress=gr.Progress()):
        if not generator_ok:
            raise gr.Error('The SpecDiff generator is not installed. Run `python scripts/download_weights.py`.')
        if not installed(BY_ID[mid]):
            raise gr.Error(f'{mid} is not installed. Choose another probe.')
        eng = engine_for('auto')
        rows, total = [], 0.0
        for ci, cls in enumerate(CLASSES):
            progress(ci / 4, desc=f'generating {cls}')
            fb = eng.generate(cls, 'AKGC417L', 3.0, guidance, steps, int(seed) + ci, n=int(n))
            pred = eng.classify(fb, mid, corrected).argmax(1)
            acc = float((pred == ci).mean())
            total += acc
            rows.append([cls, int(n), f'{100 * acc:.0f}%', ', '.join(f'{CLASSES[k]}: {int((pred == k).sum())}' for k in range(4))])
        rows.append(['mean', '', f'{100 * total / 4:.0f}%', ''])
        return rows

    # ------------------------------------------------------------------ layout
    with gr.Blocks(title='ICBHI Respiratory Sound Classifier & Generator', css=CSS) as demo:
        gr.Markdown(f'# 🩺 ICBHI Respiratory Sound Classifier & Generator\n{DISCLAIMER}\n\n*{runtime_text()}*')
        if not Engine.available_classifiers():
            gr.Markdown('**No classifier is installed.** Run `python scripts/download_weights.py`, then restart the demo.')

        with gr.Tabs():
            # ---------------------------------------------------- classify
            with gr.Tab('Classify a recording'):
                with gr.Row():
                    with gr.Column(scale=5):
                        tip('Respiratory cycle (WAV)',
                            'The recording to classify: one respiratory cycle, 1 to 5 seconds, as a WAV file. Upload one, or click a '
                            'preselected cycle below. Clips are converted to 16 kHz mono and padded or cut to 5 seconds, as in training.')
                        wav = gr.Audio(type='filepath', sources=['upload'], show_label=False)
                        tip('Model',
                            'The trained classifier that makes the prediction. A single model loads one checkpoint. An ensemble averages '
                            'several checkpoints with equal weight and loads them one at a time, so it needs little extra memory. The '
                            'card below shows the model\'s official score.')
                        model = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, show_label=False, interactive=True)
                        card = gr.HTML(model_card(DEFAULT_MODEL))
                        tip('Inference device',
                            'Where the model runs. auto uses the GPU when one is visible and the CPU otherwise. The CPU always works, '
                            'but it is slower.')
                        device = gr.Dropdown(choices=['auto', 'cuda', 'cpu'], value='auto', show_label=False)
                        tip('Prior correction',
                            'Undoes the class balancing of the synthetic training mix, so the probabilities match the real class '
                            'frequencies. It has no tuned parameter. Only models trained with a balanced synthetic mix change; the '
                            'others ignore this setting.')
                        corr = gr.Checkbox(True, label='On', show_label=True)
                        with gr.Row():
                            run = gr.Button('Classify', variant='primary', scale=3)
                            free = gr.Button('Free GPU memory', variant='secondary', scale=2)
                        status = gr.Markdown('**Ready.**')
                    with gr.Column(scale=5):
                        label = gr.Label(num_top_classes=4, label='Prediction')
                        probs = gr.Plot(label='Probabilities')
                        mel = gr.Plot(label='Input spectrogram')

                gr.HTML('<div class="ie-section">Preselected cycles</div>')
                if sample_files:
                    gr.Markdown('Click a cycle to load it, then press **Classify**. Two cycles per class. Four are from the official '
                                'test list and four from the training recordings, as the table shows.')
                    gr.Examples(examples=[[p] for p, _ in sample_files], inputs=[wav], label=None, examples_per_page=8)
                    gr.Markdown(sample_table())
                else:
                    gr.Markdown('The preselected cycles appear here once the data stage has run: `python scripts/download_data.py --zip '
                                '...` and then `python scripts/reproduce.py --stages data`.')

                model.change(show_model, [model], [card])
                run.click(classify_fn, [wav, model, device, corr], [label, probs, mel, status])
                free.click(release_memory, [], [status])

            # ---------------------------------------------------- generate
            with gr.Tab('Generate and compare'):
                with gr.Row():
                    with gr.Column(scale=4):
                        tip('Class to generate',
                            'The class of the cycle to generate: normal, crackle, wheeze or both. Generation is conditioned on the class, '
                            'so the same seed gives a different cycle for each class.')
                        cls = gr.Radio(CLASSES, value='wheeze', show_label=False)
                        tip('Stethoscope',
                            'The recording device the sample imitates. The generator is conditioned on it, because class and device are '
                            'correlated in ICBHI: the AKG microphone supplies most crackles.')
                        dev = gr.Dropdown(choices=DEVICE_CHOICES, value='AKGC417L', show_label=False)
                        tip('Cycle duration (seconds)',
                            'The length of the real cycle the sample imitates. The generator is told this, so it repeats the cycle with '
                            'the same period as real repeat-padded clips.')
                        dur = gr.Slider(1.0, 5.0, value=3.0, step=0.1, show_label=False)
                        tip('Guidance',
                            'How strongly sampling is pushed towards the chosen class. Higher values give more class-typical cycles and less '
                            'variety. The report used 3.5 for the main pool and 2.0 for the transductive pool.')
                        guid = gr.Slider(1.0, 6.0, value=3.5, step=0.5, show_label=False)
                        tip('Sampling steps',
                            'Denoising steps used to make the cycle. More steps take longer. 50 is the report\'s value; 10 to 20 is enough '
                            'for a quick look.')
                        steps = gr.Slider(10, 100, value=50, step=10, show_label=False)
                        tip('Seed', 'The random seed. The same settings and seed always give the same cycle.')
                        seed = gr.Number(0, precision=0, show_label=False)
                        tip('Classifier',
                            'The model that judges the generated cycle and the nearest real cycle. Its output is shown next to them.')
                        gmodel = gr.Dropdown(choices=model_choices, value=DEFAULT_MODEL, show_label=False)
                        tip('Prior correction',
                            'As on the classify tab: applied only to models trained with a balanced synthetic mix.')
                        gcorr = gr.Checkbox(True, label='On', show_label=True)
                        gbtn = gr.Button('Generate and compare', variant='primary')
                    with gr.Column(scale=6):
                        cmp_fig = gr.Plot(label='Synthetic cycle next to the nearest real cycle')
                        audio = gr.Audio(label='Approximate audio (Griffin-Lim inversion of the log-mel; for listening only)',
                                         type='numpy')
                        pfig = gr.Plot(label='Classifier output')
                        ginfo = gr.Markdown()
                gbtn.click(generate_fn, [cls, dev, dur, guid, steps, seed, gmodel, gcorr], [cmp_fig, audio, pfig, ginfo])

            # ---------------------------------------------------- generator check
            with gr.Tab('Generator check'):
                gr.Markdown('Generates cycles of every class on the AKG microphone (3 s) and reports how often the probe recognises the '
                            'requested class. The report\'s figures (89% wheeze, 83% both) come from probe C0, which is not shipped here, '
                            'so each probe gives its own numbers.')
                with gr.Row():
                    with gr.Column(scale=4):
                        tip('Cycles per class',
                            'How many cycles to generate for each class. Each cycle is one sampling run, so more cycles take longer.')
                        n = gr.Slider(2, 32, value=8, step=2, show_label=False)
                        tip('Guidance', 'As on the generate tab.')
                        g3 = gr.Slider(1.0, 6.0, value=3.5, step=0.5, show_label=False)
                        tip('Sampling steps', 'As on the generate tab.')
                        s3 = gr.Slider(10, 100, value=50, step=10, show_label=False)
                        tip('Seed', 'The first seed. Each class uses the next integer.')
                        sd3 = gr.Number(0, precision=0, show_label=False)
                        tip('Probe classifier',
                            'The model that decides whether a generated cycle is recognised as the requested class. A different probe gives '
                            'different numbers, so say which one you used.')
                        m3 = gr.Dropdown(choices=model_choices, value='M0', show_label=False)
                        tip('Prior correction', 'As on the classify tab.')
                        c3 = gr.Checkbox(False, label='On', show_label=True)
                        bb = gr.Button('Run check', variant='primary')
                    with gr.Column(scale=6):
                        tbl = gr.Dataframe(headers=['class', 'n', 'recognised as requested', 'predictions'], interactive=False)
                bb.click(check_fn, [n, g3, s3, sd3, m3, c3], tbl)

            # ---------------------------------------------------- about
            with gr.Tab('About'):
                table = ['| Model | Official Score (last epoch) | Installed |', '|---|---|---|']
                table += [f'| {m["label"]} | {m["score"]} | {"yes" if installed(m) else "no"} |' for m in MODELS]
                gr.Markdown((HERE / 'ABOUT.md').read_text() + '\n\n### Models in this demo\n\n' + '\n'.join(table))

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7860)
    ap.add_argument('--share', action='store_true', help='also create a public Gradio link')
    a = ap.parse_args()
    print(f'classifiers installed: {Engine.available_classifiers()} | generator: {Engine.generator_available()}', flush=True)
    demo = build()
    demo.queue().launch(server_name=a.host, server_port=a.port, share=a.share,
                        allowed_paths=[str(DATA_DIR)] if DATA_DIR.is_dir() else None)


if __name__ == '__main__':
    main()

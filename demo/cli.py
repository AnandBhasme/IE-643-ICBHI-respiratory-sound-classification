#!/usr/bin/env python3
"""Command-line version of the demo (no Gradio needed).

    python demo/cli.py classify recording.wav [--model M0] [--no-correction]
    python demo/cli.py generate --cls wheeze --device AKGC417L --duration 3 --out out/ [--steps 50 --guidance 3.5 --seed 0]
    python demo/cli.py check --n 8                     # how often generated cycles are recognised as the requested class
`generate` writes <out>/<class>_<device>_<seed>.{png,wav,json}: comparison figure, approximate audio, classifier outputs.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import CLASSES, Engine, comparison_figure, probability_figure  # noqa: E402


def pick(engine, key):
    names = engine.available_classifiers()
    if not names:
        raise SystemExit('no classifier weights found: python scripts/download_weights.py')
    return next((n for n in names if n.startswith(key)), names[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='auto')
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('classify')
    c.add_argument('wav')
    c.add_argument('--model', default='T_K3')
    c.add_argument('--no-correction', action='store_true')
    g = sub.add_parser('generate')
    g.add_argument('--cls', choices=CLASSES, required=True)
    g.add_argument('--device-name', dest='dev', default='AKGC417L', choices=['AKGC417L', 'LittC2SE', 'Meditron', 'Litt3200'])
    g.add_argument('--duration', type=float, default=3.0)
    g.add_argument('--guidance', type=float, default=3.5)
    g.add_argument('--steps', type=int, default=50)
    g.add_argument('--seed', type=int, default=0)
    g.add_argument('--out', default='demo_out')
    k = sub.add_parser('check')
    k.add_argument('--n', type=int, default=8)
    k.add_argument('--guidance', type=float, default=3.5)
    k.add_argument('--steps', type=int, default=50)
    k.add_argument('--model', default='M0')
    a = ap.parse_args()
    eng = Engine(a.device)
    if a.cmd == 'classify':
        name = pick(eng, a.model)
        fb, secs = eng.wav_to_fbank(a.wav)
        p = eng.classify(fb, name, not a.no_correction)[0]
        print(json.dumps({'model': name, 'prior_correction': not a.no_correction, 'seconds_used': round(min(secs, 5.0), 2),
                          'prediction': CLASSES[int(p.argmax())], 'probabilities': {CLASSES[i]: round(float(p[i]), 4) for i in range(4)}}, indent=1))
    elif a.cmd == 'generate':
        import soundfile as sf
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        syn = eng.generate(a.cls, a.dev, a.duration, a.guidance, a.steps, a.seed, n=1)[0]
        real, meta = eng.nearest_real(syn, a.cls, a.dev)
        stem = f'{a.cls}_{a.dev}_{a.seed}'
        fig = comparison_figure(syn, real, meta, eng.profile(a.cls, a.dev), a.cls, a.dev)
        fig.savefig(out / f'{stem}.png', dpi=130)
        audio, sr = eng.to_audio(syn, a.duration)
        sf.write(out / f'{stem}.wav', audio, sr)
        res = {'requested': a.cls, 'device': a.dev, 'nearest_real': meta, 'classifiers': {}}
        for name in eng.available_classifiers():
            ps, pr = eng.classify(syn, name)[0], eng.classify(real, name)[0]
            res['classifiers'][name] = {'synthetic': {CLASSES[i]: round(float(ps[i]), 4) for i in range(4)},
                                        'nearest_real': {CLASSES[i]: round(float(pr[i]), 4) for i in range(4)},
                                        'synthetic_recognised_as_requested': bool(ps.argmax() == CLASSES.index(a.cls))}
        (out / f'{stem}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        print('wrote', out / f'{stem}.png', out / f'{stem}.wav')
    else:
        name = pick(eng, a.model)
        total = 0.0
        for ci, cls in enumerate(CLASSES):
            fb = eng.generate(cls, 'AKGC417L', 3.0, a.guidance, a.steps, ci, n=a.n)
            pred = eng.classify(fb, name, False).argmax(1)
            acc = float((pred == ci).mean())
            total += acc / 4
            print(f'{cls:8s} recognised as {cls}: {100 * acc:5.1f}%  (n={a.n}; predictions {np.bincount(pred, minlength=4).tolist()})')
        print(f'mean {100 * total:.1f}%  [probe: {name}]')


if __name__ == '__main__':
    main()

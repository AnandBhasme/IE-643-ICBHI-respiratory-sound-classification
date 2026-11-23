#!/usr/bin/env python3
"""Choose the demo's preselected clips: two official test cycles per class, taken from results/saved/test_order.csv.

    python scripts/make_demo_samples.py

Writes demo/samples.json. The clips themselves are not copied into the repository (ICBHI audio is not redistributed); the demo
reads them from artifacts/data/official_icbhi/test/real/ after the data stage has run.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['normal', 'crackle', 'wheeze', 'both']


def main():
    rows = list(csv.DictReader(open(ROOT / 'results/saved/test_order.csv', newline='')))
    picks = []
    for c, name in enumerate(CLASSES):
        cycles = sorted((r for r in rows if int(r['label']) == c), key=lambda r: r['filename'])
        n = len(cycles)
        for frac in (1 / 3, 2 / 3):
            r = cycles[int(n * frac)]
            picks.append(dict(file=f"{r['filename']}.wav", label=c, class_name=name, device=r['device'], split='official test'))
    out = ROOT / 'demo/samples.json'
    out.write_text(json.dumps(picks, indent=1) + '\n')
    print(f'wrote {len(picks)} clips to {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()

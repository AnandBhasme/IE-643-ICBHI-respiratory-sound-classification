#!/usr/bin/env python3
"""Write demo/samples.json: the eight preselected cycles of the demo, two per class.

    python scripts/make_demo_samples.py

The eight clips were chosen by hand for the earlier demo so that every class is shown twice. Four of them come from the
official test list and four from the training recordings; the split of each clip is read from results/saved/test_order.csv
and reported in the demo, not assumed. The clips are not copied into the repository (ICBHI audio is not redistributed); the
demo reads them from artifacts/data/official_icbhi/{test,training}/real/ after the data stage has run.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['normal', 'crackle', 'wheeze', 'both']
CHOSEN = [  # (file, label) as in the earlier demo's labels.csv
    ('151_2p2_Ar_mc_AKGC417L_event_3_label_0.wav', 0),
    ('130_2p5_Tc_mc_AKGC417L_event_9_label_0.wav', 0),
    ('138_1p3_Pl_mc_AKGC417L_event_4_label_1.wav', 1),
    ('130_2p3_Pl_mc_AKGC417L_event_9_label_1.wav', 1),
    ('118_1b1_Al_sc_Litt3200_event_5_label_2.wav', 2),
    ('178_1b2_Pr_mc_AKGC417L_event_7_label_2.wav', 2),
    ('218_1b1_Al_sc_Meditron_event_10_label_3.wav', 3),
    ('130_3p4_Tc_mc_AKGC417L_event_0_label_3.wav', 3),
]


def main():
    rows = {r['filename']: r for r in csv.DictReader(open(ROOT / 'results/saved/test_order.csv', newline=''))}
    out = []
    for file, label in CHOSEN:
        name = file[:-4]
        assert int(name.rsplit('_', 1)[1]) == label, file
        row = rows.get(name)
        split = 'test' if row else 'training'
        if row:
            assert int(row['label']) == label, file
        device = name.split('_')[4]
        out.append(dict(file=file, split=split, label=label, class_name=CLASSES[label], device=device))
    out.sort(key=lambda s: (s['label'], s['file']))
    (ROOT / 'demo/samples.json').write_text(json.dumps(out, indent=1) + '\n')
    print(f'wrote {len(out)} clips: {sum(s["split"] == "test" for s in out)} from the test list, '
          f'{sum(s["split"] == "training" for s in out)} from the training recordings')


if __name__ == '__main__':
    main()

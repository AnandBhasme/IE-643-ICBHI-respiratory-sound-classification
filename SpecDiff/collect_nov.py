#!/usr/bin/env python3
"""Summarise novelty-phase AST runs (stdlib only).

Reads <runs_dir>/**/metrics.csv for run folders tagged nov_<phase>_<arm>_seed<k>
and prints, per arm, the last-epoch Se / Sp / Score (mean and sample SD over
seeds), the mean over the last five epochs, and per-class recall.
In phase 'sel' the numbers are on the validation subset; in phase 'final' they
are on the official test set.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st


def mean_sd(values):
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs_dir', required=True)
    p.add_argument('--phase', required=True)
    p.add_argument('--out_csv', default=None)
    args = p.parse_args()
    pattern = re.compile(rf'nov_{args.phase}_(.+)_seed(\d+)$')
    arms = {}
    for path in sorted(glob.glob(os.path.join(args.runs_dir, '**', 'metrics.csv'), recursive=True)):
        m = pattern.search(os.path.basename(os.path.dirname(path)))
        if not m:
            continue
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        # a run counts only when it reached the number of epochs it was configured for
        try:
            with open(os.path.join(os.path.dirname(path), 'train_args.json')) as f:
                need = int(json.load(f).get('epochs', 20))
        except (OSError, ValueError):
            need = 20
        if len(rows) < need:
            continue
        last, tail = rows[-1], rows[-5:]
        arms.setdefault(m.group(1), []).append(dict(
            seed=int(m.group(2)), split=last['eval_split'],
            Se=float(last['Se']), Sp=float(last['Sp']), Score=float(last['Score']),
            Score5=st.mean(float(r['Score']) for r in tail),
            Se5=st.mean(float(r['Se']) for r in tail), Sp5=st.mean(float(r['Sp']) for r in tail),
            recalls=[float(last[f'recall_{c}']) for c in ('normal', 'crackle', 'wheeze', 'both')]))
    out_rows = []
    print(f'{"arm":14s} n  {"Se":>13s} {"Sp":>13s} {"Score":>13s} | last-5 Se/Sp/Score      | recall N/C/W/B')
    for arm in sorted(arms):
        runs = arms[arm]
        se, sp, sc = (mean_sd([r[k] for r in runs]) for k in ('Se', 'Sp', 'Score'))
        se5, sp5, sc5 = (st.mean(r[k] for r in runs) for k in ('Se5', 'Sp5', 'Score5'))
        rec = [st.mean(r['recalls'][i] for r in runs) for i in range(4)]
        print(f'{arm:14s} {len(runs)}  {se[0]:6.2f}±{se[1]:5.2f} {sp[0]:6.2f}±{sp[1]:5.2f} {sc[0]:6.2f}±{sc[1]:5.2f} | '
              f'{se5:6.2f} {sp5:6.2f} {sc5:6.2f} | ' + ' '.join(f'{v:5.1f}' for v in rec)
              + f'   [{runs[0]["split"]}] seeds={sorted(r["seed"] for r in runs)}')
        out_rows.append([arm, len(runs), se[0], se[1], sp[0], sp[1], sc[0], sc[1], se5, sp5, sc5] + rec)
    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['arm', 'n', 'Se', 'Se_sd', 'Sp', 'Sp_sd', 'Score', 'Score_sd', 'Se_last5', 'Sp_last5',
                        'Score_last5', 'recall_normal', 'recall_crackle', 'recall_wheeze', 'recall_both'])
            w.writerows(out_rows)


if __name__ == '__main__':
    main()

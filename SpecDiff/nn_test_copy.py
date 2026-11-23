#!/usr/bin/env python3
"""Report-control: are T_K3's synthetic cycles near-copies of official-test cycles?

GPU job, minutes. Located in SpecDiff/ (uses evaluate_synth.load_probe/run_probe).
Writes ONLY to --out (never into the pool directory). Diagnostic only: nothing is selected or tuned.

For each pool, distances are cosine nearest-neighbour distances (same definition as evaluate_synth.nn_dist)
in two spaces: (a) probe features (real-only AST C0 seed 1) and (b) raw log-mel, 2x2 average pooled.
  syn->test   : each synthetic sample to its nearest official-test cycle
  train->test : each real TRAINING cycle to its nearest official-test cycle (baseline: unrelated real data)
  syn->train  : each synthetic sample to its nearest training cycle
ratio_test = median(syn->test) / median(train->test). Well below the non-transductive pool's value
means the transductive generator moved its samples onto the test cycles.
near_copy_p1 / p5: share of synthetic samples closer to a test cycle than the 1st / 5th percentile of
train->test distances.
"""
import argparse, csv, json, os
import numpy as np
import torch
from evaluate_synth import load_probe, run_probe, nn_dist

CLASSES = ['normal', 'crackle', 'wheeze', 'both']


def pool2(x):
    x = np.asarray(x, dtype=np.float32)
    n, t, f = x.shape
    x = x[:, :t - t % 2, :f - f % 2].reshape(n, t // 2, 2, f // 2, 2).mean((2, 4))
    return x.reshape(n, -1)


def nn_chunked(q, b, chunk=2048):
    return np.concatenate([nn_dist(q[i:i + chunk], b) for i in range(0, len(q), chunk)])


def nn_idx(q, b, chunk=2048):
    qn = q / np.linalg.norm(q, axis=1, keepdims=True); bn = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.concatenate([(qn[i:i + chunk] @ bn.T).argmax(1) for i in range(0, len(q), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pools', nargs='+', required=True, help='pool dirs (T_K3 pool first, then controls)')
    ap.add_argument('--teacher', default='', help='teacher npy (soft labels) for the FIRST pool, optional')
    ap.add_argument('--probe', required=True)
    ap.add_argument('--train_cache', required=True)
    ap.add_argument('--test_cache', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda')
    model = load_probe(a.probe, dev)
    tr = np.load(a.train_cache, allow_pickle=False); te = np.load(a.test_cache, allow_pickle=False)
    te_y = te['label']
    feats = {'probe': {}, 'logmel': {}}
    feats['probe']['train'], _ = run_probe(model, tr['fbank'], dev)
    feats['probe']['test'], _ = run_probe(model, te['fbank'], dev)
    feats['logmel']['train'], feats['logmel']['test'] = pool2(tr['fbank']), pool2(te['fbank'])
    rows, detail = [], {}
    for k, pdir in enumerate(a.pools):
        pool = np.load(os.path.join(pdir, 'pool.npy'), mmap_mode='r')
        with open(os.path.join(pdir, 'pool.csv'), newline='') as f:
            meta = list(csv.DictReader(f))
        y = np.asarray([int(r['label']) for r in meta])
        name = os.path.basename(os.path.normpath(pdir))
        sp, _ = run_probe(model, pool, dev)
        spaces = {'probe': sp, 'logmel': pool2(pool)}
        for space, s in spaces.items():
            base_tr, base_te = feats[space]['train'], feats[space]['test']
            d_st, d_tt, d_sr = nn_chunked(s, base_te), nn_chunked(base_tr, base_te), nn_chunked(s, base_tr)
            p1, p5 = np.percentile(d_tt, [1, 5])
            for c in [None, 0, 1, 2, 3]:
                m = np.ones(len(y), bool) if c is None else (y == c)
                rows.append(dict(pool=name, space=space, cls='all' if c is None else CLASSES[c], n=int(m.sum()),
                                 med_syn_to_test=float(np.median(d_st[m])), med_train_to_test=float(np.median(d_tt)),
                                 ratio_test=float(np.median(d_st[m]) / np.median(d_tt)),
                                 med_syn_to_train=float(np.median(d_sr[m])),
                                 near_copy_p1=float((d_st[m] < p1).mean()), near_copy_p5=float((d_st[m] < p5).mean())))
            if k == 0 and space == 'probe':
                idx = nn_idx(s, base_te)
                info = dict(nearest_test_label_matches_intended=float((te_y[idx] == y).mean()))
                close = d_st < p5
                info['n_close_p5'] = int(close.sum())
                if close.any():
                    info['close_nearest_test_label_matches_intended'] = float((te_y[idx][close] == y[close]).mean())
                if a.teacher:
                    t = np.load(a.teacher)
                    assert len(t) == len(y), ('teacher not aligned with pool', t.shape, len(y))
                    ta = t.argmax(1)
                    info['teacher_argmax_matches_intended'] = float((ta == y).mean())
                    info['teacher_argmax_matches_nearest_test_label'] = float((ta == te_y[idx]).mean())
                    if close.any():
                        info['close_teacher_argmax_matches_nearest_test_label'] = float((ta[close] == te_y[idx][close]).mean())
                detail[name] = info
                order = np.argsort(d_st)[:6]
                np.savez_compressed(os.path.join(a.out, 'closest_pairs.npz'), syn=np.asarray(pool[order], np.float16),
                                    test=te['fbank'][idx[order]].astype(np.float16), syn_label=y[order],
                                    test_label=te_y[idx[order]], dist=d_st[order])
    with open(os.path.join(a.out, 'nn_test_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(a.out, 'pseudo_label_check.json'), 'w') as f:
        json.dump(detail, f, indent=2)
    print('done ->', a.out)


if __name__ == '__main__':
    main()

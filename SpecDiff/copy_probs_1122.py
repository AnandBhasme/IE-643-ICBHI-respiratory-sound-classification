#!/usr/bin/env python3
"""Task A step 2 (22 Nov): copy RAW probs_test_last.npy for M0/T_K3/M3 (1-10), M1b/K3 (1-5) and write test_order.csv."""
import csv, os, shutil, sys
from test_labels import test_order
from ensemble_nov import run_dir

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
names, lab, dev = test_order()
n = 0
for arm, seeds in (('M0', 10), ('T_K3', 10), ('M3', 10), ('M1b', 5), ('K3', 5)):
    for s in range(1, seeds + 1):
        d = run_dir(arm, s)
        assert d and os.path.basename(d) == f'icbhi_ast_ce_nov_final_{arm}_seed{s}', (arm, s, d)
        shutil.copyfile(os.path.join(d, 'probs_test_last.npy'), os.path.join(out, f'{arm}_seed{s}.npy'))
        n += 1
with open(os.path.join(out, 'test_order.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['index', 'filename', 'label', 'device'])
    for i, (a, b, c) in enumerate(zip(names, lab, dev)):
        w.writerow([i, a, int(b), c])
print(f'copied {n} prob files + test_order.csv ({len(names)} rows) -> {out}')

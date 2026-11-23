"""Labels of the official test set in the order used by the classifier's test loader
(sorted glob of <data_folder>/test/*/*.wav; label = last '_' token of the file name).

If the audio is not on disk (for example when only the saved probabilities in results/saved are used),
the identical order is read from results/saved/test_order.csv (columns index, filename, label, device).
"""
import csv
import os
from glob import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '../artifacts/data/official_icbhi')
SAVED_ORDER = os.path.join(HERE, '../results/saved/test_order.csv')


def test_order(data=DATA):
    paths = sorted(glob(os.path.join(data, 'test', '*', '*.wav')))
    if paths:
        names = [os.path.basename(p)[:-4] for p in paths]
        labels = np.array([int(n.split('_')[-1]) for n in names])
        devices = [n.split('_')[4] for n in names]
        return names, labels, devices
    with open(SAVED_ORDER, newline='') as f:
        rows = list(csv.DictReader(f))
    return [r['filename'] for r in rows], np.array([int(r['label']) for r in rows]), [r['device'] for r in rows]

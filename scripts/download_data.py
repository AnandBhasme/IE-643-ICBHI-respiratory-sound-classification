#!/usr/bin/env python3
"""Get the data and the checkpoint the pipeline needs, and verify them. Standard library only.

What ends up where (relative to the repository root):
  dataset/ICBHI_final_database/*.wav, *.txt                      ICBHI 2017: 920 recordings and their cycle annotations
  dataset/ICBHI_final_database/ICBHI_challenge_train_test.txt    official recording split: 539 train, 381 test
  Classification/pretrained_models/audioset_10_10_0.4593.pth     AudioSet-pretrained AST checkpoint (about 345 MB)

Usual calls
  python scripts/download_data.py --icbhi --ast                     # official ICBHI host first, then the AST checkpoint
  python scripts/download_data.py --zip PATH/ICBHI_final_database.zip --ast   # a copy of the zip you already have
  python scripts/download_data.py --verify                          # only check what is in place (no download)

If the official host cannot be reached (it is often blocked from clusters and proxied campus networks), download
ICBHI_final_database.zip in a browser and pass it with --zip. The split file is then rebuilt from
results/saved/test_order.csv, which lists the official test cycles. The rebuilt file is checked: it must give 539 train
and 381 test recordings, and its test recordings must be exactly the ones in test_order.csv. Pass --split-file to use
an official split file instead.

Every step is verified at the end and the script exits non-zero if a check fails.
"""
import argparse
import csv
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ICBHI_BASE = 'https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database'
ICBHI_ZIP_URL = f'{ICBHI_BASE}/ICBHI_final_database.zip'
SPLIT_URL = f'{ICBHI_BASE}/ICBHI_challenge_train_test.txt'
AST_URL = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
AST_FILE = ROOT / 'Classification/pretrained_models/audioset_10_10_0.4593.pth'
AST_MIN_BYTES = 300_000_000           # the checkpoint is about 345 MB; anything much smaller is a failed download
TEST_ORDER = ROOT / 'results/saved/test_order.csv'
N_RECORDINGS, N_TRAIN, N_TEST = 920, 539, 381
ALIASES = {'226_1b1_Pl_sc_Meditron': '226_1b1_Pl_sc_LittC2SE'}   # same recording, two instrument suffixes (as in prepare_official_split.py)


def fetch(url, dest):
    """Download url to dest. Returns True on success; prints the reason and returns False otherwise."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + '.part')
    print(f'downloading {url}')
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(part, 'wb') as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except (urllib.error.URLError, OSError) as e:
        part.unlink(missing_ok=True)
        print(f'  could not download: {e}')
        return False
    part.replace(dest)
    return True


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def data_dir_of(dataset_dir):
    return dataset_dir / 'ICBHI_final_database'


def extract(zip_path, dataset_dir):
    print(f'extracting {zip_path} into {dataset_dir}')
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dataset_dir)


def recordings(data_dir):
    return {p.stem for p in data_dir.glob('*.wav')}


def read_test_order():
    with open(TEST_ORDER, newline='', encoding='utf-8') as f:
        return {row['filename'].split('_event_')[0] for row in csv.DictReader(f)}


def rebuild_split(data_dir):
    """Write ICBHI_challenge_train_test.txt from test_order.csv, using the recordings present in data_dir."""
    if not TEST_ORDER.is_file():
        raise SystemExit(f'cannot rebuild the split: {TEST_ORDER} is missing')
    test = read_test_order()
    names = sorted(recordings(data_dir))
    out = data_dir / 'ICBHI_challenge_train_test.txt'
    with open(out, 'w', encoding='utf-8', newline='') as f:
        for name in names:
            f.write(f'{name}\t{"test" if name in test else "train"}\n')
    print(f'rebuilt split from {rel(TEST_ORDER)} -> {rel(out)}')
    return out


def read_split(path):
    split = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.reader(f, delimiter='\t'):
            if not row or not row[0].strip():
                continue
            name = ALIASES.get(Path(row[0].strip()).stem, Path(row[0].strip()).stem)
            split[name] = row[1].strip().lower()
    return split


def check(label, ok, detail=''):
    print(f'  [{"ok" if ok else "FAIL"}] {label}{(" - " + detail) if detail else ""}')
    return ok


def verify(dataset_dir, need_ast):
    data_dir = data_dir_of(dataset_dir)
    print('verification')
    good = True
    wavs = recordings(data_dir) if data_dir.is_dir() else set()
    good &= check(f'{N_RECORDINGS} recordings (.wav)', len(wavs) == N_RECORDINGS, f'found {len(wavs)}')
    ann = {p.stem for p in data_dir.glob('*.txt')} & wavs if data_dir.is_dir() else set()
    good &= check(f'{N_RECORDINGS} annotation files', len(ann) == N_RECORDINGS, f'found {len(ann)}')
    split_file = data_dir / 'ICBHI_challenge_train_test.txt'
    if check('split file present', split_file.is_file(), rel(split_file) if split_file.is_file() else ''):
        split = read_split(split_file)
        n_train = sum(v == 'train' for v in split.values())
        n_test = sum(v == 'test' for v in split.values())
        good &= check(f'{N_TRAIN} train / {N_TEST} test recordings', (n_train, n_test) == (N_TRAIN, N_TEST),
                      f'found {n_train} / {n_test}')
        good &= check('split covers every recording', set(split) == wavs, f'{len(set(split) ^ wavs)} differ' if set(split) != wavs else '')
        if TEST_ORDER.is_file():
            test_set = {n for n, v in split.items() if v == 'test'}
            good &= check('test recordings equal results/saved/test_order.csv', test_set == read_test_order())
    else:
        good = False
    if need_ast:
        ok = AST_FILE.is_file() and AST_FILE.stat().st_size >= AST_MIN_BYTES
        good &= check('AST checkpoint', ok, rel(AST_FILE) if AST_FILE.is_file() else 'missing')
    return good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--icbhi', action='store_true', help='download the ICBHI zip from the official host if not in place')
    ap.add_argument('--zip', type=Path, help='use a local copy of ICBHI_final_database.zip')
    ap.add_argument('--split-file', type=Path, help='use this official split file instead of the one from the host or test_order.csv')
    ap.add_argument('--ast', action='store_true', help='download the AudioSet AST checkpoint')
    ap.add_argument('--verify', action='store_true', help='only verify what is in place')
    ap.add_argument('--dataset-dir', type=Path, default=ROOT / 'dataset', help='where the ICBHI folder goes (default: dataset/)')
    a = ap.parse_args()

    dataset_dir = a.dataset_dir
    data_dir = data_dir_of(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dataset_dir / 'ICBHI_final_database.zip'

    if not a.verify:
        complete = data_dir.is_dir() and len(recordings(data_dir)) == N_RECORDINGS
        if a.zip:
            if not a.zip.is_file():
                raise SystemExit(f'--zip: no such file {a.zip}')
            if not complete:
                extract(a.zip, dataset_dir)
        elif a.icbhi and not complete:
            if not zip_path.is_file() and not fetch(ICBHI_ZIP_URL, zip_path):
                print('\nThe official ICBHI host could not be downloaded (see the error above; a certificate error means its\n'
                      'TLS certificate failed verification on this machine).\n'
                      'Download ICBHI_final_database.zip in a browser (https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)\n'
                      'and run:  python scripts/download_data.py --zip /path/to/ICBHI_final_database.zip')
            elif zip_path.is_file():
                extract(zip_path, dataset_dir)

        if complete or a.zip or (a.icbhi and data_dir.is_dir()):
            if a.split_file:
                shutil.copy(a.split_file, data_dir / 'ICBHI_challenge_train_test.txt')
                print(f'copied split file {a.split_file}')
            elif not (data_dir / 'ICBHI_challenge_train_test.txt').is_file():
                if not (a.icbhi and fetch(SPLIT_URL, data_dir / 'ICBHI_challenge_train_test.txt')):
                    rebuild_split(data_dir)

        if a.ast and not (AST_FILE.is_file() and AST_FILE.stat().st_size >= AST_MIN_BYTES):
            if not fetch(AST_URL, AST_FILE):
                print('AST checkpoint not downloaded; place it at', rel(AST_FILE))

    ok = verify(dataset_dir, need_ast=a.ast or AST_FILE.exists())
    print('\nall checks passed' if ok else '\nsome checks failed (see FAIL lines above)')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()

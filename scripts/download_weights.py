#!/usr/bin/env python3
"""Download the demo weights listed in weights/weights_manifest.json into ./weights and verify their SHA-256.

    python scripts/download_weights.py                         # everything still missing, then verify all
    python scripts/download_weights.py --only specdiff_generator.pt
    python scripts/download_weights.py --verify-only           # check the files in place, download nothing

The manifest points to the GitHub releases `weights-v1` (final models and generator) and `weights-intensive-v1`
(the 15 intensive-phase checkpoints used by the ensembles). Use --release to fetch only one of them. A file that already exists is verified and kept.
A failed download is reported and the script exits non-zero; no partial file is left behind.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / 'weights'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest, attempts=3):
    """Download url to dest through a .part file. Retries because large transfers over some networks fail intermittently."""
    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, attempts + 1):
        print(f'downloading {url} (attempt {attempt}/{attempts})', flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(part, 'wb') as f:
                while block := r.read(1 << 20):
                    f.write(block)
        except (urllib.error.URLError, OSError) as e:
            part.unlink(missing_ok=True)
            print(f'  download failed: {e}', file=sys.stderr, flush=True)
            time.sleep(5 * attempt)
            continue
        part.replace(dest)
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', help='asset names from the manifest (e.g. classifier_M0_seed1.pth)')
    ap.add_argument('--release', choices=['weights-v1', 'weights-intensive-v1'], help='only the files of this release')
    ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    man = json.loads((W / 'weights_manifest.json').read_text())
    bad = 0
    for name, m in man.items():
        if a.only and name not in a.only:
            continue
        if a.release and m.get('release') != a.release:
            continue
        dst = W / m.get('dest', name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.is_file() and not a.verify_only:
            if not m.get('url'):
                print(f'{name}: no URL in weights_manifest.json', file=sys.stderr)
                bad += 1
                continue
            if not download(m['url'], dst):
                bad += 1
                continue
        if not dst.is_file():
            print(f'{name}: missing (run without --verify-only)')
            bad += 1
            continue
        ok = sha256(dst) == m['sha256']
        print(f'{name}: {"OK" if ok else "CHECKSUM MISMATCH"}')
        if not ok:
            bad += 1
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

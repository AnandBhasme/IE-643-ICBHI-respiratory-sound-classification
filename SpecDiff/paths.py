"""Repository-relative paths used by the pipeline.

All data, runs, generators and pools live under <repo>/artifacts (git-ignored). To keep them on another disk,
make `artifacts` a symbolic link to that disk.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ARTIFACTS = os.path.join(ROOT, 'artifacts')
DATA = os.path.join(ARTIFACTS, 'data', 'official_icbhi')
CACHE = os.path.join(ARTIFACTS, 'data', 'fbank_cache')
RUNS = os.path.join(ARTIFACTS, 'runs_nov')
SPECDIFF = os.path.join(ARTIFACTS, 'specdiff')

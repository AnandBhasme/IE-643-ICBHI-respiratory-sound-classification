#!/usr/bin/env python3
"""Train one classifier arm (AST, final phase: full official train set, report on the official test set).

    python scripts/train_arm.py --arm T_K3 --seed 1 [--epochs 20] [--dry-run]

Builds the training-mix manifest (SpecDiff/build_mix.py, deterministic) and calls Classification/main.py with the recipe of the
report: Adam 5e-5, 20 epochs, cosine schedule with 1 warm-up epoch, batch 32, BF16, AudioSet-pretrained AST, 5 s repeat-padded
cycles. Output: artifacts/runs_nov/icbhi_ast_ce_nov_final_<arm>_seed<k>/ (last.pth, probs_test_last.npy, metrics.csv, ...).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'artifacts'
DATA = A / 'data/official_icbhi'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--amp', default='bf16', help='bf16 on Ampere or newer GPUs; use fp16 or none on older ones')
    ap.add_argument('--quick', action='store_true', help='smoke test: tiny pools (prefix quick_) and mode fill:1300:2+3')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not a.dry_run:
        from gpu_check import require_gpu
        require_gpu(f'AST training for arm {a.arm}')
    arms = {k: v for k, v in json.load(open(ROOT / 'configs/arms.json')).items() if not k.startswith('_')}
    cfg = arms[a.arm]
    py = sys.executable
    extra = []
    if cfg['mix'] != 'none':
        mix = A / f'specdiff/mixes/final_{a.arm}.csv'
        cmd = [py, 'build_mix.py', '--kind', cfg['mix'], '--mode', 'fill:1300:2+3' if a.quick else cfg['mode'], '--real_manifest', str(DATA / 'splits/train_manifest.csv'),
               '--out', str(mix)]
        if cfg['mix'] == 'synth':
            cmd += ['--pool_dir', str(A / 'specdiff/pools' / (('quick_' if a.quick else '') + cfg['pool']))]
            if cfg.get('soft'):
                cmd += ['--teacher_tag', 'finalM0']
        print('+', ' '.join(cmd), flush=True)
        if not a.dry_run:
            subprocess.run(cmd, cwd=ROOT / 'SpecDiff', check=True)
        extra = ['--extra_manifest', str(mix)]
        if cfg.get('soft'):
            extra += ['--soft_extra', '--soft_beta', '1.0']
    cmd = [py, 'main.py', '--tag', f'nov_final_{a.arm}_seed{a.seed}', '--save_dir', str(A / 'runs_nov'), '--data_folder', str(DATA),
           '--dataset', 'icbhi', '--seed', str(a.seed), '--train_real_only', '--class_split', 'lungsound', '--n_cls', '4',
           '--epochs', str(a.epochs), '--batch_size', str(a.batch_size), '--num_workers', str(a.num_workers), '--optimizer', 'adam',
           '--learning_rate', '5e-5', '--weight_decay', '1e-6', '--desired_length', '5', '--cosine', '--warmup_epochs', '1',
           '--model', 'ast', '--test_fold', 'official', '--pad_types', 'repeat', '--resz', '1', '--n_mels', '128', '--from_sl_official',
           '--audioset_pretrained', '--method', 'ce', '--amp', a.amp, '--print_freq', '100', '--save_probs'] + extra
    print('+', ' '.join(cmd), flush=True)
    if a.dry_run:
        return
    done = A / f'runs_nov/icbhi_ast_ce_nov_final_{a.arm}_seed{a.seed}/probs_test_last.npy'
    if done.exists():
        print(f'{done} exists; skipping (delete the run folder to retrain)')
        return
    subprocess.run(cmd, cwd=ROOT / 'Classification', check=True)


if __name__ == '__main__':
    main()

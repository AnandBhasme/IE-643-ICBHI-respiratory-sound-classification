#!/usr/bin/env python3
"""Reproduce the whole pipeline: data -> baselines -> generators -> synthetic pools -> teacher -> arms -> metrics.

    python scripts/reproduce.py --list                     # show the stages and their commands
    python scripts/reproduce.py --stages evaluate           # tables from the shipped probabilities, no GPU, ~1 minute
    python scripts/reproduce.py --stages all                # everything on this machine (needs 1 GPU, days of compute)
    python scripts/reproduce.py --stages all --backend slurm --partition a40 --qos a40   # same, as a dependent SLURM job graph
    python scripts/reproduce.py --stages all --quick        # tiny end-to-end smoke run (minutes) to check the wiring

Stages (in order):
  check      environment and file checks
  data       official split, validation split, log-mel cache                      (CPU)
  m0         real-only classifiers, seeds 1..N                                    (GPU, ~25 min each)
  generators gen_full (50k steps) then gen_full_tt (20k steps, transductive)      (GPU, ~4.5 h + ~1.5 h)
  pools      sample both synthetic pools and run the quality gate                 (GPU)
  teacher    soft labels for the pools from the real-only seeds 1..5              (GPU)
  arms       M1b, M3, K3, T_K3 classifiers (see configs/arms.json)                (GPU, ~25 min each)
  evaluate   prior correction, 10-seed paired comparison, ensembles, bias sweep, tables and figures   (CPU)
  nn_test    are the synthetic cycles near-copies of test cycles? (GPU, ~5 min; needs the pools)
Every step skips work that is already done, so an interrupted run can be restarted with the same command.
Defaults reproduce the report: 10 seeds for M0/M3/T_K3, 5 for M1b/K3. Use --seeds N to shorten.
The test set is used only for scoring, as in the report; see docs/PROTOCOL.md for the nine looks and their register.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'artifacts'
DATA, CACHE, RUNS, SD = A / 'data/official_icbhi', A / 'data/fbank_cache', A / 'runs_nov', A / 'specdiff'
PY = os.path.abspath(sys.executable)
GEN = json.load(open(ROOT / 'configs/generators.json'))
ARMS = {k: v for k, v in json.load(open(ROOT / 'configs/arms.json')).items() if not k.startswith('_')}
Q = shlex.quote


class Step:
    def __init__(self, name, cmd, cwd='.', gpu=True, time='01:00:00', mem='40G', deps=(), done=None, cpus=4, budget=0):
        self.name, self.cmd, self.cwd, self.gpu, self.time, self.mem = name, cmd, ROOT / cwd, gpu, time, mem
        self.budget = budget  # minutes after which train.py stops cleanly (resumable); used on SLURM only
        self.deps, self.done, self.cpus = list(deps), done, cpus

    def is_done(self):
        return self.done is not None and Path(self.done).exists()


def plan(args):
    q = args.quick
    n = lambda k: 1 if q else min(args.seeds or ARMS[k]['seeds'], ARMS[k]['seeds'])
    ep = '1' if q else '20'
    qf = ' --quick' if q else ''
    steps = []
    S = steps.append
    # ----------------------------------------------------------- data
    S(Step('data_split', f'{PY} Classification/prepare_official_split.py --data-dir dataset/ICBHI_final_database '
           f'--split-file dataset/ICBHI_final_database/ICBHI_challenge_train_test.txt --output-root {Q(str(DATA))} && '
           f'{PY} Classification/prepare_validation_split.py --split-dir {Q(str(DATA / "splits"))}',
           gpu=False, time='01:00:00', mem='16G', done=DATA / 'splits/val_manifest.csv'))
    S(Step('data_cache', f'{PY} SpecDiff/cache_fbank.py --data_folder {Q(str(DATA))} --out_dir {Q(str(CACHE))}', gpu=False, time='01:30:00',
           mem='24G', deps=['data_split'], done=CACHE / 'test.npz'))
    # ----------------------------------------------------------- real-only classifiers
    for s in range(1, n('M0') + 1):
        S(Step(f'arm_M0_s{s}', f'{PY} scripts/train_arm.py --arm M0 --seed {s} --epochs {ep}{qf}', time='01:00:00', deps=['data_split'],
               done=RUNS / f'icbhi_ast_ce_nov_final_M0_seed{s}/probs_test_last.npy'))
    # ----------------------------------------------------------- generators
    gf, gt = GEN['gen_full'], GEN['gen_full_tt']
    f_steps = 600 if q else gf['steps']
    t_steps = 200 if q else gt['steps']
    f_save = 300 if q else gf['save_every']
    t_save = 100 if q else gt['save_every']
    f_dir, t_dir = SD / 'gen_full', SD / 'gen_full_tt'
    f_ck = f'ema-{f_steps}.pt'
    init_ck = 'ema-300.pt' if q else 'ema-30000.pt'
    S(Step('gen_full', f'{PY} SpecDiff/train.py --cache {CACHE}/train.npz --out_dir {f_dir} --steps {f_steps} --save_every {f_save} '
           f'--time_budget_min {{BUDGET}} {gf["args"]}', gpu=True, time='08:00:00' if not q else '00:30:00', mem='48G', deps=['data_cache'],
           done=f_dir / f_ck, budget=450 if not q else 0))
    tt_args = gt['args'].format(A=A, C=CACHE).replace('ema-30000.pt', init_ck)
    if q:
        tt_args = tt_args.replace('--warmup 500', '--warmup 20')
    S(Step('gen_full_tt', f'{PY} SpecDiff/train.py --cache {CACHE}/train.npz --out_dir {t_dir} --steps {t_steps} --save_every {t_save} '
           f'--time_budget_min {{BUDGET}} {tt_args}', time='04:00:00' if not q else '00:30:00', mem='48G', deps=['gen_full'],
           done=t_dir / f'ema-{t_steps}.pt', budget=210 if not q else 0))
    # ----------------------------------------------------------- pools
    pool_names = {}
    for pname, p in GEN['pools'].items():
        step_no = f_steps if p['gen'] == 'gen_full' else t_steps
        name = pname if not q else f'quick_{pname}'
        pool_names[pname] = name
        per_class = p['per_class'] if not q else '0,0,800,940'
        nsteps = '50' if not q else '4'
        extra = p['extra'].format(A=A, C=CACHE)
        S(Step(f'pool_{pname}', f'{PY} SpecDiff/sample.py --ckpt {SD / p["gen"] / f"ema-{step_no}.pt"} --cache {CACHE}/train.npz '
               f'--out_dir {SD / "pools" / name} --per_class {per_class} --guidance {p["guidance"]} --device_policy {p["policy"]} '
               f'--steps {nsteps} --batch_size 100 {extra}', time='02:00:00' if not q else '00:20:00', mem='40G', deps=[p['gen']],
               done=SD / 'pools' / name / 'pool.npy'))
        S(Step(f'gate_{pname}', f'{PY} SpecDiff/evaluate_synth.py --pool_dir {SD / "pools" / name} --probe '
               f'{RUNS}/icbhi_ast_ce_nov_final_M0_seed1/last.pth --train_cache {CACHE}/train.npz --ref test --test_cache {CACHE}/test.npz',
               time='00:30:00', deps=[f'pool_{pname}', 'arm_M0_s1'], done=SD / 'pools' / name / 'quality_test.json'))
    # ----------------------------------------------------------- teacher
    t_seeds = [x for x in GEN['teacher_seeds'] if x <= n('M0')] if not q else [1]
    ck = ' '.join(str(RUNS / f'icbhi_ast_ce_nov_final_M0_seed{s}/last.pth') for s in t_seeds)
    pdirs = ' '.join(str(SD / 'pools' / v) for v in pool_names.values())
    S(Step('teacher', f'{PY} SpecDiff/teacher_probs.py --pool_dirs {pdirs} --ckpts {ck} --tag finalM0', time='00:45:00',
           deps=[f'pool_{p}' for p in pool_names] + [f'arm_M0_s{s}' for s in t_seeds],
           done=SD / 'pools' / list(pool_names.values())[-1] / 'teacher_finalM0.npy'))
    # ----------------------------------------------------------- arms
    for arm, cfg in ARMS.items():
        if arm == 'M0':
            continue
        deps = ['teacher'] if cfg['mix'] == 'synth' else ['data_split']
        for s in range(1, n(arm) + 1):
            S(Step(f'arm_{arm}_s{s}', f'{PY} scripts/train_arm.py --arm {arm} --seed {s} --epochs {ep}{qf}', time='01:00:00', deps=deps,
                   done=RUNS / f'icbhi_ast_ce_nov_final_{arm}_seed{s}/probs_test_last.npy'))
    # ----------------------------------------------------------- evaluate
    arm_steps = [x.name for x in steps if x.name.startswith('arm_')]
    out = ROOT / ('results/quick' if q else 'results/reproduced')
    S(Step('evaluate', f'{PY} SpecDiff/make_results.py {out} && ' + ('' if q else
           f'{PY} SpecDiff/look9.py > {out}/look9.txt && {PY} SpecDiff/bias_sweep.py --out {out}/bias_sweep > {out}/bias_sweep.txt && ')
           + 'true', gpu=False, time='00:30:00', mem='16G', deps=arm_steps))
    if not q:
        pools = ' '.join(str(SD / 'pools' / v) for v in pool_names.values())
        S(Step('nn_test', f'{PY} SpecDiff/nn_test_copy.py --pools {pools} --teacher {SD / "pools" / list(pool_names.values())[0] / "teacher_finalM0.npy"} '
               f'--probe {RUNS}/icbhi_ast_ce_nov_final_M0_seed1/last.pth --train_cache {CACHE}/train.npz --test_cache {CACHE}/test.npz '
               f'--out {out}/nn_test', time='00:40:00', deps=['teacher']))
    return steps


STAGES = {'check': [], 'data': ['data_'], 'm0': ['arm_M0'], 'generators': ['gen_full', 'gen_full_tt'], 'pools': ['pool_', 'gate_'],
          'teacher': ['teacher'], 'arms': ['arm_M1b', 'arm_M3', 'arm_K3', 'arm_T_K3'], 'evaluate': ['evaluate'], 'nn_test': ['nn_test']}


def select(steps, stages):
    if 'all' in stages:
        return steps
    pref = [p for s in stages for p in STAGES[s]]
    return [x for x in steps if any(x.name.startswith(p) for p in pref)]


def check():
    import importlib
    ok = True
    for mod in ('torch', 'torchaudio', 'timm', 'numpy', 'scipy', 'soundfile', 'matplotlib', 'sklearn', 'pandas'):
        try:
            m = importlib.import_module(mod)
            print(f'  {mod:12s} {getattr(m, "__version__", "")}')
        except ImportError:
            print(f'  {mod:12s} MISSING')
            ok = False
    try:
        import torch
        print('  cuda available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
    except Exception:
        pass
    from gpu_check import MIN_GB, describe
    print(f'  GPU: {describe()} (training stages need at least {MIN_GB} GiB)')
    for p, what in ((ROOT / 'dataset/ICBHI_final_database', 'ICBHI audio (scripts/download_data.py --icbhi)'),
                    (ROOT / 'Classification/pretrained_models/audioset_10_10_0.4593.pth', 'AST checkpoint (scripts/download_data.py --ast)')):
        print(f'  {"ok     " if p.exists() else "MISSING"} {p.relative_to(ROOT)}  {what if not p.exists() else ""}')
    print(f'  {"ok     " if (ROOT / "results/saved/probs").is_dir() else "MISSING"} results/saved/probs (shipped probabilities)')


def final_cmd(st, slurm):
    return st.cmd.replace('{BUDGET}', str(st.budget if slurm else 0))


def run_local(todo, args, all_steps):
    for st in todo:
        if st.is_done() and not args.force:
            print(f'[skip] {st.name} (output exists)')
            continue
        cmd = final_cmd(st, False)
        print(f'\n[run ] {st.name}\n       {cmd}', flush=True)
        if args.dry_run:
            continue
        r = subprocess.run(cmd, shell=True, cwd=st.cwd, executable='/bin/bash')
        if r.returncode != 0:
            sys.exit(f'step {st.name} failed with exit code {r.returncode}')


def run_slurm(todo, args, all_steps):
    """One job per step, chained with --dependency=afterok. Resources: one GPU per GPU job, no --exclusive/--nodelist.
    The command travels in the environment (STEP_CMD / STEP_CWD), which avoids sbatch's comma splitting of --export."""
    ids = {}
    for st in todo:
        if st.is_done() and not args.force:
            print(f'[skip] {st.name}')
            continue
        deps = [ids[d] for d in st.deps if d in ids]
        cmd = ['sbatch', '--parsable', f'--job-name=ie643_{st.name}'[:60], f'--time={st.time}', f'--mem={st.mem}',
               f'--cpus-per-task={st.cpus}', f'--partition={args.partition}', f'--qos={args.qos or args.partition}',
               '--output=hpc/logs/%x.%j.out', '--error=hpc/logs/%x.%j.err']
        if st.gpu:
            cmd.append('--gres=gpu:1')
        if deps:
            cmd.append('--dependency=afterok:' + ':'.join(deps))
        cmd.append(str(ROOT / 'hpc/slurm/step.slurm'))
        env = dict(os.environ, STEP_CMD=final_cmd(st, True), STEP_CWD=str(st.cwd))
        print(' '.join(shlex.quote(c) for c in cmd), '   # STEP_CMD=' + env['STEP_CMD'][:110], flush=True)
        if args.dry_run:
            ids[st.name] = f'<{st.name}>'
            continue
        out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        if out.returncode != 0:
            sys.exit(f'sbatch failed for {st.name}: {out.stderr.strip()}')
        ids[st.name] = out.stdout.strip().split(';')[0]
        print(f'  -> job {ids[st.name]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stages', nargs='+', default=['check'], choices=list(STAGES) + ['all'])
    ap.add_argument('--backend', choices=['local', 'slurm'], default='local')
    ap.add_argument('--seeds', type=int, default=0, help='cap on seeds per arm (default: the report values)')
    ap.add_argument('--quick', action='store_true', help='tiny smoke run: 1 seed, 1 epoch, few generator steps')
    ap.add_argument('--partition', default=None, help='slurm partition (required for --backend slurm)')
    ap.add_argument('--qos', default=None, help='slurm qos (default: same as the partition)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='re-run steps whose output already exists')
    ap.add_argument('--list', action='store_true')
    args = ap.parse_args()
    steps = plan(args)
    if args.list:
        for x in steps:
            print(f'{x.name:28s} gpu={int(x.gpu)} time={x.time} deps={x.deps}\n    {final_cmd(x, False)}')
        return
    if 'check' in args.stages:
        check()
    todo = select(steps, [s for s in args.stages if s != 'check'] if 'all' not in args.stages else ['all'])
    if args.backend == 'slurm':
        if not args.partition:
            sys.exit('--partition is required with --backend slurm (see `sinfo`; partition and qos must match)')
        run_slurm(todo, args, steps)
    else:
        if not args.dry_run:
            pending = [st for st in todo if st.gpu and not (st.is_done() and not args.force)]
            if pending:
                from gpu_check import require_gpu
                names = ', '.join(st.name for st in pending[:3]) + (' ...' if len(pending) > 3 else '')
                require_gpu(f'local GPU step(s) {names}')
        run_local(todo, args, steps)


if __name__ == '__main__':
    main()

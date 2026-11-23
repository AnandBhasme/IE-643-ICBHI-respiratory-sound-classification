"""Model catalogue for the demo: what each choice is, where its weights come from, and the numbers it is reported with.

Sources
  ('weights', file)   a file in <repo>/weights (release weights-v1)
  ('intensive', dir)  <dir>/last.pth, where dir is an intensive-phase run folder (C0-C4); the folder comes from
                      IE643_INTENSIVE_DIR (default <repo>/weights/intensive)

Scores are the official ICBHI Score (Se + Sp) / 2 on the official test set, last epoch, mean +- sample SD over the seeds
listed. They are copied from docs/PROJECT_HISTORY.md and results/final; they are not recomputed here.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = Path(os.environ.get('IE643_WEIGHTS', ROOT / 'weights'))
INTENSIVE = Path(os.environ.get('IE643_INTENSIVE_DIR', WEIGHTS / 'intensive'))

C4_SEEDS = ['icbhi_ast_patchmix_cl_wp3_C4_seed1_bf16_retry1', 'icbhi_ast_patchmix_cl_wp3_C4_seed2_bf16_retry2',
            'icbhi_ast_patchmix_cl_wp3_C4_seed3_bf16_retry2']


def _seeds(prefix):
    return [('intensive', f'icbhi_ast_ce_wp3_{prefix}_seed{s}_bf16') for s in (1, 2, 3)]


MODELS = [
    dict(id='T_K3', group='Final models (novelty phase)', kind='single', arm='T_K3',
         label='T_K3: final model (SpecDiff + soft labels)',
         source=('weights', 'classifier_T_K3_seed1.pth'), seeds='seed 1 of 10',
         score='59.57 ± 0.64 (10 seeds); 58.46 for M0',
         desc='The final model of the novelty phase. An AST fine-tuned on the real training cycles plus SpecDiff-generated cycles '
              'with soft labels. Its generator also trained on unlabelled test audio, so it is transductive. The prior correction '
              'is applied by default.'),
    dict(id='M0', group='Final models (novelty phase)', kind='single', arm='M0',
         label='M0: real data only (baseline)',
         source=('weights', 'classifier_M0_seed1.pth'), seeds='seed 1 of 10',
         score='58.46 ± 0.96 (10 seeds)',
         desc='The baseline: an AST fine-tuned on the 4,142 real training cycles only. Every novelty comparison is against this model.'),
    dict(id='C4', group='Intensive-phase candidates (single seed)', kind='single', arm='M0',
         label='C4: Patch-Mix contrastive + EMA',
         source=('weights', 'classifier_C4_seed1.pth'), seeds='seed 1 of 3',
         score='58.74 ± 0.86 (3 seeds)',
         desc='Intensive-phase candidate C4: Patch-Mix contrastive loss with an exponential moving average of the weights, real data only.'),
    dict(id='C0', group='Intensive-phase candidates (single seed)', kind='single', arm='M0',
         label='C0: cross-entropy baseline',
         source=('intensive', 'icbhi_ast_ce_wp3_C0_seed1_bf16'), seeds='seed 1 of 3',
         score='58.14 ± 0.95 (3 seeds)',
         desc='Intensive-phase control C0: plain cross-entropy fine-tuning of the AudioSet-pretrained AST, real data only.'),
    dict(id='C1', group='Intensive-phase candidates (single seed)', kind='single', arm='M0',
         label='C1: cross-entropy + EMA',
         source=('intensive', 'icbhi_ast_ce_wp3_C1_seed1_bf16'), seeds='seed 1 of 3',
         score='58.53 ± 0.90 (3 seeds)',
         desc='Intensive-phase candidate C1: C0 plus an exponential moving average of the weights.'),
    dict(id='C2', group='Intensive-phase candidates (single seed)', kind='single', arm='M0',
         label='C2: + label smoothing and stochastic depth',
         source=('intensive', 'icbhi_ast_ce_wp3_C2_seed1_bf16'), seeds='seed 1 of 3',
         score='58.60 ± 1.11 (3 seeds)',
         desc='Intensive-phase candidate C2: C1 plus label smoothing and stochastic depth.'),
    dict(id='ENS-AST', group='Ensembles (streamed one member at a time)', kind='ensemble',
         label='Ens-AST: 15-model headline ensemble',
         members=[('intensive', d) for d in
                  [f'icbhi_ast_ce_wp3_C{c}_seed{s}_bf16' for c in range(4) for s in (1, 2, 3)]] + [('intensive', d) for d in C4_SEEDS],
         score='60.39 (15 models, last epoch)',
         desc='Equal-weight average of the 15 intensive-phase checkpoints (C0 to C4, seeds 1 to 3). This was the intensive-phase '
              'headline. Its C3 members duplicate C1 (C3\'s learning-rate groups were erased by a scheduler bug), so the ensemble '
              'contains C1 outputs twice.'),
    dict(id='ENS-C1', group='Ensembles (streamed one member at a time)', kind='ensemble',
         label='Ens-C1: 3-seed ensemble',
         members=_seeds('C1'), score='60.18 (3 seeds)',
         desc='Equal-weight average of C1 over seeds 1 to 3.'),
    dict(id='ENS-C4', group='Ensembles (streamed one member at a time)', kind='ensemble',
         label='Ens-C4: 3-seed Patch-Mix ensemble',
         members=[('intensive', d) for d in C4_SEEDS], score='59.93 (3 seeds)',
         desc='Equal-weight average of C4 over seeds 1 to 3.'),
    dict(id='ENS-C0', group='Ensembles (streamed one member at a time)', kind='ensemble',
         label='Ens-C0: 3-seed cross-entropy ensemble',
         members=_seeds('C0'), score='59.57 (3 seeds)',
         desc='Equal-weight average of C0 over seeds 1 to 3.'),
]

BY_ID = {m['id']: m for m in MODELS}


def member_paths(model):
    """Checkpoint files that a model needs (one for a single model, several for an ensemble)."""
    out = []
    for kind, name in ([model['source']] if model['kind'] == 'single' else model['members']):
        out.append(WEIGHTS / name if kind == 'weights' else INTENSIVE / name / 'last.pth')
    return out


def installed(model):
    return all(p.is_file() for p in member_paths(model))


def missing_hint(model):
    missing = [p for p in member_paths(model) if not p.is_file()]
    return (f'{model["id"]} is not installed ({len(missing)} of {len(member_paths(model))} checkpoint files missing). '
            f'Run `python scripts/download_weights.py` for the final models, or set IE643_INTENSIVE_DIR to the folder that holds '
            f'the intensive-phase runs.')

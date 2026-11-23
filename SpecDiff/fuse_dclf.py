#!/usr/bin/env python3
"""Fuse the diffusion (generative) classifier with the AST seed ensemble.
dclf logit_c = -(mean_t err_c) / (mean_t err_null) / tau, AST = log mean-probability.
fused = log p_AST + w * log_softmax(dclf logit). Prints Score for a grid of w and tau."""
import argparse
import numpy as np


def score(pred, label):
    ab = label > 0
    se, sp = (pred[ab] == label[ab]).mean() * 100, (pred[~ab] == 0).mean() * 100
    return se, sp, (se + sp) / 2


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def dclf_logp(err, tau):
    rel = err[:, :, :4].mean(1) / err[:, :, 4:5].mean(1)
    return log_softmax(-rel / tau)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('npz')
    p.add_argument('--w', nargs='+', type=float, default=[0, 0.25, 0.5, 1, 2, 4])
    p.add_argument('--tau', nargs='+', type=float, default=[0.005, 0.01, 0.02])
    a = p.parse_args()
    d = np.load(a.npz)
    label, err = d['label'], d['err']
    ast = np.log(d['ast_probs'].mean(0) + 1e-8)
    print('AST ensemble       Se %.2f Sp %.2f Score %.2f' % score(ast.argmax(1), label))
    print('dclf alone         Se %.2f Sp %.2f Score %.2f' % score(dclf_logp(err, 0.01).argmax(1), label))
    single = [score(np.log(pk + 1e-8).argmax(1), label)[2] for pk in d['ast_probs']]
    print('AST single models  mean Score %.2f' % np.mean(single))
    for tau in a.tau:
        g = dclf_logp(err, tau)
        row = ' '.join('w%.2g:%.2f' % (w, score((ast + w * g).argmax(1), label)[2]) for w in a.w)
        print(f'tau {tau}: {row}')
        # same for single AST models (mean over seeds)
        row = ' '.join('w%.2g:%.2f' % (w, np.mean([score((np.log(pk + 1e-8) + w * g).argmax(1), label)[2]
                                                     for pk in d['ast_probs']])) for w in a.w)
        print(f'   single-model mean: {row}')


if __name__ == '__main__':
    main()

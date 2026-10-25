"""ICBHI four-class evaluation metrics."""

import numpy as np


def evaluate_confusion(confusion, two_cls_eval=False):
    """Return metrics from a truth-row/prediction-column confusion matrix.

    Se, Sp, Score, recalls, accuracy, and binary sensitivity are fractions in
    [0, 1]. ICBHI Se counts only exact class matches among abnormal cycles.
    """
    cm = np.asarray(confusion, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or cm.shape[0] < 2:
        raise ValueError(f"Expected a square confusion matrix with at least two classes, got {cm.shape}")
    support = cm.sum(axis=1)
    if np.any(support == 0):
        raise ValueError("Every class must have at least one true example")

    specificity = cm[0, 0] / support[0]
    exact_abnormal_hits = np.trace(cm[1:, 1:])
    abnormal_support = support[1:].sum()
    sensitivity = cm[1:, 1:].sum() / abnormal_support if two_cls_eval else exact_abnormal_hits / abnormal_support
    recalls = cm.diagonal() / support
    total = float(cm.sum())
    tp = cm.diagonal().astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = support.astype(np.float64) - tp
    tn = total - tp - fp - fn
    micro_se = float(tp.sum() / total)
    micro_sp = float(tn.sum() / (tn + fp).sum())
    micro_accuracy = float((tp + tn).sum() / (cm.shape[0] * total))
    return {
        "Se": float(sensitivity),
        "Sp": float(specificity),
        "Score": float((sensitivity + specificity) / 2.0),
        "Se_micro": micro_se,
        "Sp_micro": micro_sp,
        "Score_micro": float((micro_se + micro_sp) / 2.0),
        "accuracy_micro": micro_accuracy,
        "recall": recalls.astype(float).tolist(),
        "accuracy": float(cm.trace() / cm.sum()),
        "se_binary": float(cm[1:, 1:].sum() / abnormal_support),
        "confusion": cm.tolist(),
    }

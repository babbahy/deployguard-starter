from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y_true, p_failure, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true)
    p_failure = np.asarray(p_failure)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bins[:-1], bins[1:]):
        if hi == 1.0:
            mask = (p_failure >= lo) & (p_failure <= hi)
        else:
            mask = (p_failure >= lo) & (p_failure < hi)
        if not mask.any():
            continue
        conf = p_failure[mask].mean()
        freq = y_true[mask].mean()
        ece += mask.mean() * abs(conf - freq)
    return float(ece)


def recall_at_fpr(y_true, p_failure, target_fpr: float) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, p_failure)
    valid = np.where(fpr <= target_fpr)[0]
    return float(tpr[valid].max()) if len(valid) else 0.0


def binary_metrics(y_true, p_failure, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    p_failure = np.asarray(p_failure).astype(float)
    y_pred = (p_failure >= threshold).astype(int)

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, p_failure)),
        "ece_15": expected_calibration_error(y_true, p_failure, 15),
        "recall_at_fpr_1pct": recall_at_fpr(y_true, p_failure, 0.01),
        "recall_at_fpr_5pct": recall_at_fpr(y_true, p_failure, 0.05),
    }

    # These require both classes in the evaluation set.
    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, p_failure))
        result["pr_auc"] = float(average_precision_score(y_true, p_failure))
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    return result

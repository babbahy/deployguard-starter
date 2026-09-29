from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve

from deployguard.evaluation.metrics import binary_metrics


def evaluation_report(y_true, p_failure, threshold: float = 0.5, n_bins: int = 15) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_failure, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise FloatingPointError("Evaluation requires finite probabilities in [0, 1]")
    report = {
        "failure_prevalence": float(y.mean()),
        **binary_metrics(y, p, threshold=threshold),
    }
    precision, recall, thresholds = precision_recall_curve(y, p)
    report["precision_at_recall"] = {}
    for target in (0.25, 0.5, 0.75, 0.9):
        eligible = precision[recall >= target]
        report["precision_at_recall"][str(target)] = (
            float(eligible.max()) if eligible.size else None
        )
    report["recall_at_precision"] = {}
    for target in (0.8, 0.9, 0.95):
        eligible = np.flatnonzero(precision[:-1] >= target)
        report["recall_at_precision"][str(target)] = (
            float(recall[eligible].max()) if eligible.size else 0.0
        )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = []
    for index, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (p >= lo) & ((p <= hi) if index == n_bins - 1 else (p < hi))
        if mask.any():
            reliability.append({
                "bin_low": float(lo), "bin_high": float(hi), "count": int(mask.sum()),
                "mean_probability": float(p[mask].mean()),
                "observed_failure_rate": float(y[mask].mean()),
            })
    report["reliability_bins"] = reliability

    predicted = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1.0 - p)
    selective = []
    for confidence_threshold in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        accepted = confidence >= confidence_threshold
        failures_accepted = accepted & (predicted == 1)
        tp = int(np.sum(failures_accepted & (y == 1)))
        total_failures = int(y.sum())
        selective.append({
            "confidence_threshold": confidence_threshold,
            "coverage": float(accepted.mean()),
            "review_rate": float(1.0 - accepted.mean()),
            "selective_error_rate": (float(np.mean(predicted[accepted] != y[accepted]))
                                     if accepted.any() else None),
            "selective_accuracy": (float(np.mean(predicted[accepted] == y[accepted]))
                                   if accepted.any() else None),
            "failure_precision": (float(tp / failures_accepted.sum())
                                  if failures_accepted.any() else 0.0),
            "failure_recall": (float(tp / total_failures) if total_failures else None),
        })
    report["selective_prediction_curve"] = selective
    return report

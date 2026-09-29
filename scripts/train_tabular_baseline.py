from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from deployguard.evaluation.metrics import binary_metrics

CATEGORICAL = ["event", "head_branch", "is_merge"]
NUMERIC = ["total_churn", "files_modified", "msg_len"]


def read_split(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    rows = pd.read_json(path, lines=True).to_dict("records")
    parsed = []
    for row in rows:
        fields = dict(line.split(": ", 1) for line in row["state"].splitlines())
        parsed.append({key: fields.get(key, "unknown") for key in CATEGORICAL + NUMERIC})
    frame = pd.DataFrame(parsed)
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col].replace("unknown", np.nan), errors="coerce")
    return frame, np.asarray([int(row["label"]) for row in rows])


def choose_threshold(y_true: np.ndarray, p_failure: np.ndarray, min_precision: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, p_failure)
    eligible = np.flatnonzero(precision[:-1] >= min_precision)
    if not len(eligible):
        return {"min_precision": min_precision, "threshold": 1.0,
                "validation_precision": None, "validation_recall": 0.0,
                "no_threshold_met_precision": True}
    best_recall = recall[eligible].max()
    best = eligible[recall[eligible] == best_recall]
    i = best[-1]
    return {"min_precision": min_precision, "threshold": float(thresholds[i]),
            "validation_precision": float(precision[i]),
            "validation_recall": float(recall[i]),
            "no_threshold_met_precision": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-precision", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.min_precision <= 1:
        raise ValueError("--min-precision must be in (0, 1]")

    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Dataset leakage audit must pass before training")
    train_x, train_y = read_split(args.data_dir / "train.jsonl")
    val_x, val_y = read_split(args.data_dir / "validation.jsonl")
    test_x, test_y = read_split(args.data_dir / "test.jsonl")

    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
    ])
    features = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL),
    ])
    model = Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
    ])
    model.fit(train_x, train_y)
    val_p = model.predict_proba(val_x)[:, 1]
    test_p = model.predict_proba(test_x)[:, 1]
    threshold = choose_threshold(val_y, val_p, args.min_precision)
    threshold_value = threshold["threshold"]
    test_metrics = binary_metrics(test_y, test_p, threshold=threshold_value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output_dir / "logistic_regression.joblib")
    report = {
        "model": "logistic_regression",
        "features": CATEGORICAL + NUMERIC,
        "train_rows": len(train_y), "validation_rows": len(val_y), "test_rows": len(test_y),
        "train_failure_rate": float(train_y.mean()),
        "validation_failure_rate": float(val_y.mean()), "test_failure_rate": float(test_y.mean()),
        "test_average_precision_prevalence_baseline": float(test_y.mean()),
        "threshold_selected_on_validation": threshold,
        "test_metrics_at_validation_threshold": test_metrics,
        "threshold_0_5_metrics": binary_metrics(test_y, test_p, threshold=0.5),
        "repository_split": manifest["split_strategy"],
        "data_manifest": str(args.data_dir / "manifest.json"),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

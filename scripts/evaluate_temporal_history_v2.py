from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from deployguard.data.history import HISTORY_FEATURES
from deployguard.evaluation.report import evaluation_report


CAT = ["repo", "event", "head_branch", "workflow_id", "is_merge"]
BASE_NUM = ["run_number", "additions", "deletions", "total_churn", "files_modified",
            "msg_len", "num_parents", "created_hour", "created_weekday"]
NUM = BASE_NUM + HISTORY_FEATURES


def load_splits(path: Path):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("benchmark_version") != "temporal-history-v2" or not manifest["audit"]["passed"]:
        raise ValueError("Expected audited temporal-history-v2")
    splits = {}
    for name, info in manifest["splits"].items():
        file = path / f"{name}.jsonl"
        if hashlib.sha256(file.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"{name} checksum mismatch")
        frame = pd.read_json(file, lines=True)
        if len(frame) != info["rows"] or frame.id.duplicated().any():
            raise ValueError(f"{name} identity/count mismatch")
        splits[name] = frame
    return splits, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU baselines for causal historical features v2")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal_history_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/temporal_history_v2_baselines"))
    args = parser.parse_args()
    splits, manifest = load_splits(args.data_dir)
    train, validation, test = (splits[n] for n in ("train", "validation", "test"))
    y_train, y_val, y_test = (part.label.to_numpy(dtype=int) for part in (train, validation, test))
    models = {
        "rich_logistic_with_causal_history": Pipeline([
            ("features", ColumnTransformer([
                ("text", TfidfVectorizer(max_features=50000, min_df=3, ngram_range=(1, 2),
                                          strip_accents="unicode"), "commit_message"),
                ("numeric", Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]), NUM),
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
                ]), CAT),
            ])),
            ("classifier", LogisticRegression(max_iter=1200, solver="liblinear", random_state=42)),
        ]),
        "hist_gradient_boosting_with_causal_history": Pipeline([
            ("features", ColumnTransformer([
                ("numeric", SimpleImputer(strategy="median"), NUM),
                ("categorical", Pipeline([
                    ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                               max_categories=255, encoded_missing_value=-1)),
                ]), CAT),
            ], sparse_threshold=0.0)),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=0.06, max_iter=120, max_leaf_nodes=15,
                min_samples_leaf=30, l2_regularization=2.0,
                categorical_features=[False] * len(NUM) + [True] * len(CAT),
                early_stopping=False, random_state=42,
            )),
        ]),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark": "temporal-history-v2", "base_benchmark_id": manifest["base_benchmark_id"],
        "test_rows": len(test), "test_failure_prevalence": float(y_test.mean()),
        "prediction_protocol": "prior outcomes count only after updated_at < current created_at; test updates are online-causal",
        "models": {},
    }
    all_preds = []
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(train, y_train)
        val_p = model.predict_proba(validation)[:, 1]
        test_p = model.predict_proba(test)[:, 1]
        if not np.isfinite(val_p).all() or not np.isfinite(test_p).all():
            raise FloatingPointError(f"{name} returned non-finite probabilities")
        rows = []
        for repo, part in test.assign(p_failure=test_p).groupby("repo", sort=True):
            rows.append({"model": name, "repo": repo,
                         **evaluation_report(part.label.to_numpy(dtype=int), part.p_failure.to_numpy())})
        prediction = pd.DataFrame({
            "model": name, "id": test.id, "repo": test.repo, "timestamp": test.created_at,
            "label": y_test, "p_failure": test_p,
            "predicted_label": (test_p >= 0.5).astype(int),
            "run_id": f"history-v2-{name}",
        })
        all_preds.append(prediction)
        report["models"][name] = {
            "fit_seconds": time.perf_counter() - started,
            "validation": evaluation_report(y_val, val_p),
            "test": evaluation_report(y_test, test_p),
            "per_repository": rows,
        }
        print(f"{name}: test AP={report['models'][name]['test']['pr_auc']:.4f}")
    pd.concat(all_preds, ignore_index=True).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([{"model": name, **result["test"]}
                  for name, result in report["models"].items()]).to_csv(
                      args.output_dir / "aggregate_metrics.csv", index=False
                  )
    pd.DataFrame([row for result in report["models"].values()
                  for row in result["per_repository"]]).to_csv(
                      args.output_dir / "per_repository_metrics.csv", index=False
                  )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

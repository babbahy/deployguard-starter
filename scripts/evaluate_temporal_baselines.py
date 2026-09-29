from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from deployguard.data.temporal import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE
from deployguard.evaluation.metrics import binary_metrics


def load_splits(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Temporal dataset leakage audit did not pass")
    result = {}
    for split, info in manifest["splits"].items():
        path = data_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"{split} checksum differs from temporal manifest")
        frame = pd.read_json(path, lines=True)
        if frame.id.duplicated().any():
            raise ValueError(f"Duplicate run identity in {split}")
        result[split] = frame
    return result, manifest


def build_logistic(
    categorical_columns: list[str] | None = None,
    numeric_columns: list[str] | None = None,
    include_text: bool = True,
) -> Pipeline:
    categorical_columns = CATEGORICAL_FEATURES if categorical_columns is None else categorical_columns
    numeric_columns = NUMERIC_FEATURES if numeric_columns is None else numeric_columns
    transformers = []
    if include_text:
        transformers.append(("text", TfidfVectorizer(
            max_features=50000, min_df=3, ngram_range=(1, 2), strip_accents="unicode"
        ), TEXT_FEATURE))
    if numeric_columns:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric, numeric_columns))
    if categorical_columns:
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ])
        transformers.append(("categorical", categorical, categorical_columns))
    preprocess = ColumnTransformer(transformers)
    return Pipeline([
        ("features", preprocess),
        ("classifier", LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=42
        )),
    ])


def build_hist_gradient_boosting() -> Pipeline:
    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("ordinal", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            max_categories=255, encoded_missing_value=-1,
        )),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ], sparse_threshold=0.0)
    category_mask = [False] * len(NUMERIC_FEATURES) + [True] * len(CATEGORICAL_FEATURES)
    return Pipeline([
        ("features", preprocess),
        ("classifier", HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=120, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=2.0,
            categorical_features=category_mask, early_stopping=False,
            random_state=42,
        )),
    ])


def threshold_for_precision(y_true: np.ndarray, probabilities: np.ndarray, minimum: float) -> dict:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    eligible = np.flatnonzero(precision[:-1] >= minimum)
    if not len(eligible):
        return {
            "minimum_precision": minimum, "threshold": None,
            "validation_precision": None, "validation_recall": None,
        }
    best_recall = recall[eligible].max()
    candidates = eligible[recall[eligible] == best_recall]
    index = candidates[-1]
    return {
        "minimum_precision": minimum,
        "threshold": float(thresholds[index]),
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
    }


def metrics_for_group(y_true: np.ndarray, p_failure: np.ndarray) -> dict:
    metrics = binary_metrics(y_true, p_failure, threshold=0.5)
    return {
        "runs": int(len(y_true)),
        "failures": int(np.sum(y_true)),
        "actual_failure_rate": float(np.mean(y_true)),
        "average_predicted_failure_probability": float(np.mean(p_failure)),
        **metrics,
    }


def per_repository_metrics(
    predictions: pd.DataFrame, output_path: Path
) -> pd.DataFrame:
    results = []
    for (model_name, repo), frame in predictions.groupby(["model", "repo"], sort=True):
        result = {"model": model_name, "repo": repo}
        result.update(metrics_for_group(
            frame.label.to_numpy(dtype=int),
            frame.p_failure.to_numpy(dtype=float),
        ))
        results.append(result)
    report = pd.DataFrame(results)
    report.to_csv(output_path, index=False)
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CPU baselines for the temporal CI benchmark")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/temporal_baselines_v2"))
    parser.add_argument("--minimum-validation-precision", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 < args.minimum_validation_precision <= 1:
        raise ValueError("minimum validation precision must be in (0, 1]")

    splits, manifest = load_splits(args.data_dir)
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    y_train = train.label.to_numpy(dtype=int)
    y_val = validation.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Training split must contain both outcomes")

    x_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
    x_train, x_val, x_test = train[x_cols], validation[x_cols], test[x_cols]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_rows = []
    prediction_rows = []
    report = {
        "dataset": str(args.data_dir),
        "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "train_failure_rate": float(y_train.mean()),
        "validation_failure_rate": float(y_val.mean()),
        "test_failure_rate": float(y_test.mean()),
        "random_ranking_pr_auc_reference": float(y_test.mean()),
        "minimum_validation_precision": args.minimum_validation_precision,
        "models": {},
        "per_repository_metrics": str(output_dir / "per_repository_metrics.csv"),
        "temporal_audit_passed": manifest["audit"]["passed"],
    }

    constants = {
        "majority_class": np.zeros(len(test), dtype=float),
        "train_prevalence": np.full(len(test), y_train.mean(), dtype=float),
    }
    for name, probabilities in constants.items():
        metrics = metrics_for_group(y_test, probabilities)
        report["models"][name] = {
            "fit": False, "test_at_0_5": metrics,
        }
        prediction_rows.append(pd.DataFrame({
            "model": name, "repo": test.repo.to_numpy(),
            "label": y_test, "p_failure": probabilities,
        }))

    models = {
        "logistic_tfidf_structured": build_logistic(),
        "logistic_tfidf_structured_no_repo": build_logistic(
            categorical_columns=[c for c in CATEGORICAL_FEATURES if c != "repo"]
        ),
        "logistic_repo_only": build_logistic(
            categorical_columns=["repo"], numeric_columns=[], include_text=False
        ),
        "hist_gradient_boosting_structured": build_hist_gradient_boosting(),
    }
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - started
        val_probability = model.predict_proba(x_val)[:, 1]
        test_probability = model.predict_proba(x_test)[:, 1]
        if not np.isfinite(val_probability).all() or not np.isfinite(test_probability).all():
            raise FloatingPointError(f"{name} produced non-finite probabilities")
        threshold = threshold_for_precision(
            y_val, val_probability, args.minimum_validation_precision
        )
        test_at_validation_threshold = None
        if threshold["threshold"] is not None:
            test_at_validation_threshold = binary_metrics(
                y_test, test_probability, threshold=threshold["threshold"]
            )
        report["models"][name] = {
            "fit": True,
            "training_seconds": train_seconds,
            "threshold_selected_on_validation": threshold,
            "test_at_0_5": metrics_for_group(y_test, test_probability),
            "test_at_validation_threshold": test_at_validation_threshold,
        }
        joblib.dump(model, output_dir / f"{name}.joblib")
        prediction_rows.append(pd.DataFrame({
            "model": name, "repo": test.repo.to_numpy(),
            "label": y_test, "p_failure": test_probability,
        }))
        print(f"{name}: {train_seconds:.1f}s; test AP={report['models'][name]['test_at_0_5']['pr_auc']:.4f}")

    predictions = pd.concat(prediction_rows, ignore_index=True)
    per_repository_metrics(predictions, output_dir / "per_repository_metrics.csv")
    repo_split_results = json.loads(Path("runs/tabular_full/metrics.json").read_text())
    modern_path = Path("runs/modernbert-l4-20260927T094219Z-56e25afb/test_metrics.json")
    modern_results = json.loads(modern_path.read_text()) if modern_path.exists() else None
    report["comparison_context"] = {
        "repository_disjoint_logistic": repo_split_results["threshold_0_5_metrics"],
        "repository_disjoint_logistic_test_prevalence": repo_split_results["test_failure_rate"],
        "repository_disjoint_modernbert": modern_results,
        "warning": "These datasets and feature sets differ; the comparison is descriptive, not a controlled model ablation.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "test_failure_rate": report["test_failure_rate"],
        "models": {
            name: values.get("test_at_0_5")
            for name, values in report["models"].items()
        },
        "output_dir": str(output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

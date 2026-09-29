from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from deployguard.evaluation.metrics import binary_metrics


from deployguard.evaluation.report import evaluation_report
TEXT = ["commit_message"]
REPO = ["repo"]
STRUCTURED_NUMERIC = [
    "run_number", "additions", "deletions", "total_churn", "files_modified",
    "msg_len", "num_parents", "created_hour", "created_weekday",
]
STRUCTURED_CATEGORICAL = ["event", "head_branch", "workflow_id", "is_merge"]
CHANGE_SIZE = ["additions", "deletions", "total_churn", "files_modified", "msg_len"]
WORKFLOW_PROXY_NUMERIC = ["run_number", "created_hour", "created_weekday"]
WORKFLOW_PROXY_CATEGORICAL = ["event", "workflow_id"]


def checked_splits(data_dir: Path) -> tuple[dict[str, pd.DataFrame], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    freeze = manifest.get("benchmark_freeze")
    if not freeze or freeze.get("version") != "temporal-v1" or not freeze.get("immutable"):
        raise ValueError("Temporal benchmark v1 has not been frozen")
    lock = json.loads((data_dir / "benchmark_lock.json").read_text())
    if lock != freeze:
        raise ValueError("Benchmark lock differs from the v1 manifest")
    splits = {}
    for name, info in manifest["splits"].items():
        path = data_dir / f"{name}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != freeze["split_sha256"][name]:
            raise ValueError(f"Frozen {name} bytes changed")
        frame = pd.read_json(path, lines=True)
        if len(frame) != info["rows"] or frame.id.duplicated().any():
            raise ValueError(f"{name} row count or identity check failed")
        splits[name] = frame
    return splits, freeze


def build_model(numeric: list[str], categorical: list[str], include_text: bool) -> Pipeline:
    transformers = []
    if include_text:
        transformers.append(("text", TfidfVectorizer(
            max_features=50000, min_df=3, ngram_range=(1, 2), strip_accents="unicode"
        ), "commit_message"))
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]), categorical))
    return Pipeline([
        ("features", ColumnTransformer(transformers)),
        ("classifier", LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)),
    ])


def repo_bootstrap_ci(frame: pd.DataFrame, iterations: int, seed: int) -> dict:
    """Percentile intervals resample repositories, keeping each repository intact."""
    rng = np.random.default_rng(seed)
    by_repo = {repo: part.index.to_numpy() for repo, part in frame.groupby("repo", sort=True)}
    repos = np.array(list(by_repo))
    y = frame.label.to_numpy(dtype=int)
    p = frame.p_failure.to_numpy(dtype=float)
    aps, rocs = [], []
    for _ in range(iterations):
        sampled = rng.choice(repos, size=len(repos), replace=True)
        indices = np.concatenate([by_repo[repo] for repo in sampled])
        if np.unique(y[indices]).size < 2:
            continue
        metrics = binary_metrics(y[indices], p[indices])
        aps.append(metrics["pr_auc"])
        rocs.append(metrics["roc_auc"])
    if not aps:
        return {"method": "repository-cluster percentile bootstrap", "iterations": iterations,
                "valid_replicates": 0, "pr_auc_95pct": None, "roc_auc_95pct": None}
    return {
        "method": "repository-cluster percentile bootstrap; resample 19 repositories with replacement",
        "iterations": iterations, "valid_replicates": len(aps),
        "pr_auc_95pct": [float(x) for x in np.quantile(aps, [0.025, 0.975])],
        "roc_auc_95pct": [float(x) for x in np.quantile(rocs, [0.025, 0.975])],
    }


def summarize(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "rows": len(y), "failures": int(y.sum()), "failure_prevalence": float(y.mean()),
        **binary_metrics(y, p, threshold=0.5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature ablations on frozen temporal benchmark v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/temporal_v1_ablations"))
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    splits, freeze = checked_splits(args.data_dir)
    train, val, test = (splits[n] for n in ("train", "validation", "test"))
    y_train = train.label.to_numpy(dtype=int)
    y_val = val.label.to_numpy(dtype=int)
    y_test = test.label.to_numpy(dtype=int)
    groups = {
        "commit_message_text_only": ([], [], True),
        "structured_metadata_only": (STRUCTURED_NUMERIC, STRUCTURED_CATEGORICAL, False),
        "change_size_only": (CHANGE_SIZE, [], False),
        "workflow_history_proxy_only": (WORKFLOW_PROXY_NUMERIC, WORKFLOW_PROXY_CATEGORICAL, False),
        "text_structured_no_repo": (STRUCTURED_NUMERIC, STRUCTURED_CATEGORICAL, True),
        "full_rich_state": (STRUCTURED_NUMERIC, STRUCTURED_CATEGORICAL + REPO, True),
        "repository_only": ([], REPO, False),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "benchmark_id": freeze["benchmark_id"],
        "test_rows": len(test), "test_failures": int(y_test.sum()),
        "test_failure_prevalence": float(y_test.mean()), "models": {},
        "test_set_used_for_selection": False,
    }
    predictions = []
    model_specs = {"prevalence_reference": None, **groups}
    for name, spec in model_specs.items():
        started = time.perf_counter()
        if spec is None:
            val_p = np.full(len(val), y_train.mean())
            test_p = np.full(len(test), y_train.mean())
        else:
            numeric, categorical, include_text = spec
            model = build_model(numeric, categorical, include_text)
            model.fit(train, y_train)
            val_p = model.predict_proba(val)[:, 1]
            test_p = model.predict_proba(test)[:, 1]
        if not np.isfinite(val_p).all() or not np.isfinite(test_p).all():
            raise FloatingPointError(f"{name} produced non-finite probabilities")
        pred = pd.DataFrame({
            "model": name, "id": test.id, "repo": test.repo,
            "timestamp": test.created_at, "label": y_test, "p_failure": test_p,
            "predicted_label": (test_p >= 0.5).astype(int),
            "run_id": f"temporal-v1-ablation-{name}",
        })
        predictions.append(pred)
        per_repo = []
        for repo, part in pred.groupby("repo", sort=True):
            per_repo.append({"model": name, "repo": repo, **summarize(
                part.label.to_numpy(dtype=int), part.p_failure.to_numpy(dtype=float)
            )})
        report["models"][name] = {
            "fit_seconds": time.perf_counter() - started,
            "validation": summarize(y_val, val_p),
            "test": summarize(y_test, test_p),
            "test_evaluation_report": evaluation_report(y_test, test_p),
            "test_cluster_bootstrap_ci": repo_bootstrap_ci(
                pred, args.bootstrap_iterations, seed=42
            ),
            "per_repository": per_repo,
        }
        print(f"{name}: test AP={report['models'][name]['test']['pr_auc']:.4f}")
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "predictions.csv", index=False)
    pd.DataFrame([
        {"model": name, **values["test"],
         "pr_auc_ci_low": (values["test_cluster_bootstrap_ci"]["pr_auc_95pct"] or [None, None])[0],
         "pr_auc_ci_high": (values["test_cluster_bootstrap_ci"]["pr_auc_95pct"] or [None, None])[1],
         "roc_auc_ci_low": (values["test_cluster_bootstrap_ci"]["roc_auc_95pct"] or [None, None])[0],
         "roc_auc_ci_high": (values["test_cluster_bootstrap_ci"]["roc_auc_95pct"] or [None, None])[1]}
        for name, values in report["models"].items()
    ]).to_csv(args.output_dir / "aggregate_metrics.csv", index=False)
    pd.DataFrame([
        row for values in report["models"].values() for row in values["per_repository"]
    ]).to_csv(args.output_dir / "per_repository_metrics.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

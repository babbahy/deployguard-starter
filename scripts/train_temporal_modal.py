"""Validation-selected ModernBERT sweep and three-seed evaluation on frozen v1."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "deployguard-training"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "transformers==4.57.6", "datasets==4.8.5", "accelerate==1.15.0",
        "pandas==2.3.3", "numpy==2.4.6", "scikit-learn==1.9.1",
    )
    .env({"PYTHONPATH": "/app/src", "HF_HOME": "/vol/hf-cache", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/app/src", ignore=["**/__pycache__/**", "**/*.egg-info/**"])
    .add_local_file(ROOT / "scripts/train_temporal_modernbert.py", "/app/train_temporal_modernbert.py")
)
app = modal.App("deployguard-temporal-modernbert")


@app.function(
    image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": volume},
    timeout=14400, retries=0, min_containers=0, max_containers=1, scaledown_window=2,
)
def train(dataset_id: str, run_id: str, learning_rate: float, epochs: float,
          seed: int, skip_test: bool) -> dict:
    import subprocess
    import sys
    import torch

    if not re.fullmatch(r"[a-f0-9]{64}", dataset_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid dataset/run identifier")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide CUDA; refusing to run on CPU")
    if learning_rate not in (1e-5, 2e-5, 5e-5) or epochs not in (2.0, 3.0, 5.0):
        raise ValueError("Configuration is outside the predeclared sweep")
    volume.reload()
    data_dir = Path("/vol/datasets") / dataset_id
    output = Path("/vol/runs") / run_id
    command = [
        sys.executable, "-u", "/app/train_temporal_modernbert.py",
        "--data-dir", str(data_dir), "--output-dir", str(output), "--run-id", run_id,
        "--learning-rate", str(learning_rate), "--epochs", str(epochs), "--seed", str(seed),
        "--batch-size", "16", "--max-length", "512", "--require-cuda",
    ]
    if skip_test:
        command.append("--skip-test")
    subprocess.run(command, check=True, timeout=13800)
    volume.commit()
    files = {name: (output / name).read_text() for name in (
        "run_info.json", "metrics.json", "validation_predictions.csv",
    )}
    if not skip_test:
        files["test_predictions.csv"] = (output / "test_predictions.csv").read_text()
    return {"run_id": run_id, "files": files}


@app.local_entrypoint()
def main(data_dir: str = "data/processed/cicd_temporal", sweep_seed: int = 2026):
    import csv
    from datetime import datetime, timezone

    directory = Path(data_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    freeze = manifest.get("benchmark_freeze")
    if not freeze or freeze.get("version") != "temporal-v1":
        raise ValueError("This launch script only accepts frozen temporal benchmark v1")
    lock_path = directory / "benchmark_lock.json"
    if json.loads(lock_path.read_text()) != freeze:
        raise ValueError("Temporal-v1 benchmark lock mismatch")
    for name, digest in freeze["split_sha256"].items():
        if hashlib.sha256((directory / f"{name}.jsonl").read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen {name} split changed")
    dataset_id = hashlib.sha256(json.dumps(freeze["split_sha256"], sort_keys=True).encode()).hexdigest()
    (ROOT / "runs/temporal_v1_modernbert").mkdir(parents=True, exist_ok=True)
    with volume.batch_upload(force=True) as upload:
        for name in ("manifest.json", "benchmark_lock.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
            upload.put_file(directory / name, f"/datasets/{dataset_id}/{name}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep = []
    for learning_rate in (1e-5, 2e-5, 5e-5):
        for epochs in (2.0, 3.0, 5.0):
            run_id = f"mb-v1-sweep-lr{learning_rate:g}-ep{int(epochs)}-s{sweep_seed}-{stamp}"
            result = train.remote(dataset_id, run_id, learning_rate, epochs, sweep_seed, True)
            files = result["files"]
            run_dir = ROOT / "runs/temporal_v1_modernbert" / "sweep" / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            for name, contents in files.items():
                (run_dir / name).write_text(contents)
            info = json.loads(files["run_info.json"])
            metrics = json.loads(files["metrics.json"])
            sweep.append({"run_id": run_id, "learning_rate": learning_rate,
                          "epochs": epochs, "validation": metrics["validation"],
                          "training_seconds": info["training_seconds"]})
            print(f"{run_id}: validation AP={metrics['validation']['pr_auc']:.5f}")
    sweep.sort(key=lambda item: (item["validation"]["pr_auc"], item["validation"]["roc_auc"]), reverse=True)
    selected = {"learning_rate": sweep[0]["learning_rate"], "epochs": sweep[0]["epochs"]}
    (ROOT / "runs/temporal_v1_modernbert/sweep_summary.json").write_text(json.dumps({
        "benchmark_id": freeze["benchmark_id"], "selection_metric": "validation PR-AUC",
        "sweep_seed": sweep_seed, "configurations": sweep, "selected": selected,
        "test_metrics_used_for_selection": False,
    }, indent=2) + "\n")
    final_runs = []
    for seed in (17, 42, 73):
        run_id = f"mb-v1-final-lr{selected['learning_rate']:g}-ep{int(selected['epochs'])}-s{seed}-{stamp}"
        result = train.remote(dataset_id, run_id, selected["learning_rate"], selected["epochs"], seed, False)
        run_dir = ROOT / "runs/temporal_v1_modernbert" / "final" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        for name, contents in result["files"].items():
            (run_dir / name).write_text(contents)
        metrics = json.loads(result["files"]["metrics.json"])
        final_runs.append({"run_id": run_id, "seed": seed, "test": metrics["test"]})
        print(f"{run_id}: test metrics saved at {run_dir}")
    metric_names = ("pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "ece_15",
                    "recall_at_fpr_1pct", "recall_at_fpr_5pct")
    summary = {}
    for name in metric_names:
        values = [run["test"][name] for run in final_runs if run["test"].get(name) is not None]
        mean = sum(values) / len(values)
        summary[name] = {"mean": mean,
                         "std": (sum((value - mean) ** 2 for value in values) / (len(values) - 1)) ** 0.5
                                 if len(values) > 1 else 0.0}
    (ROOT / "runs/temporal_v1_modernbert/final_summary.json").write_text(json.dumps({
        "benchmark_id": freeze["benchmark_id"], "selected": selected, "seeds": final_runs,
        "test_failure_prevalence": final_runs[0]["test"]["failure_prevalence"],
        "mean_plus_std": summary, "selection_used_test": False,
    }, indent=2) + "\n")

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/tmp/deployguard-upstream-laya")
VOLUME_NAME = "deployguard-training"
V1_ID = "temporal-v1-3e9cc5c049399d0b"
TEST_HASH = "1410c9dcdc2c87ab503178f3f9749a674bd34959b402ca04ba16d3a0e22af003"
UPSTREAM_COMMIT = "9d955671415fc19f069b9cc998928075c1f255ec"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
SWEEP_NAME = "sweep_retry_01"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.6", "safetensors==0.7.0", "huggingface_hub==0.36.2",
                 "numpy==2.4.6", "scikit-learn==1.9.1")
    .env({"PYTHONPATH": "/app/src:/app/laya_upstream", "HF_HOME": "/vol/hf-cache",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/app/src", ignore=["**/__pycache__/**", "**/*.egg-info/**"])
    .add_local_dir(UPSTREAM, "/app/laya_upstream",
                   ignore=[".git/**", "tests/**", "research/**", "notebooks/**", "benchmarks/**"])
    .add_local_file(ROOT / "scripts/train_laya_temporal_v1.py", "/app/train_laya_temporal_v1.py")
)
app = modal.App("deployguard-laya-temporal-v1")


def verify_local(typed_dir: Path, data_dir: Path, include_test: bool) -> tuple[dict, dict]:
    source_manifest = json.loads((data_dir / "manifest.json").read_text())
    freeze = source_manifest.get("benchmark_freeze")
    lock = json.loads((data_dir / "benchmark_lock.json").read_text())
    bridge = json.loads((typed_dir / "benchmark.json").read_text())
    if not freeze or lock != freeze or freeze.get("benchmark_id") != V1_ID:
        raise ValueError("Expected immutable temporal-v1 benchmark lock")
    if bridge.get("benchmark_id") != V1_ID or bridge.get("split_hashes") != freeze["split_sha256"]:
        raise ValueError("Typed bridge does not preserve the frozen temporal-v1 split hashes")
    if freeze["split_sha256"].get("test") != TEST_HASH:
        raise ValueError("Locked test hash mismatch")
    checked = ("train", "validation", "test") if include_test else ("train", "validation")
    from deployguard.data.temporal import rich_temporal_v1_state

    for split in checked:
        raw_path, typed_path = data_dir / f"{split}.jsonl", typed_dir / f"{split}.jsonl"
        if hashlib.sha256(raw_path.read_bytes()).hexdigest() != freeze["split_sha256"][split]:
            raise ValueError(f"Frozen {split} source split checksum mismatch")
        with raw_path.open(encoding="utf-8") as raw_file, typed_path.open(encoding="utf-8") as typed_file:
            for index, pair in enumerate(zip(raw_file, typed_file, strict=True)):
                raw, rendered = (json.loads(line) for line in pair)
                expected_label = "failure" if int(raw["label"]) == 1 else "success"
                record = rendered
                if (record.get("id") != raw["id"] or record.get("repo") != raw["repo"]
                        or record.get("timestamp") != raw["created_at"]
                        or record.get("state") != rich_temporal_v1_state(raw)
                        or record.get("gold", {}).get("outcome", {}).get("label") != expected_label):
                    raise ValueError(f"Typed bridge mismatch in {split} row {index}")
    return source_manifest, freeze


@app.function(image=image, gpu="L4", cpu=4, memory=32768, volumes={"/vol": volume},
              timeout=14400, retries=0, min_containers=0, max_containers=6, scaledown_window=2)
def train_one(dataset_id: str, phase: str, run_id: str, encoder_lr: float, epochs: int,
              seed: int, batch_size: int) -> dict:
    import re

    if phase not in {"sweep", "final"} or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid Laya run identifier or phase")
    if encoder_lr not in (1e-5, 2.5e-5, 5e-5) or epochs not in (2, 4) or batch_size != 8:
        raise ValueError("Laya run differs from the pre-registered configuration matrix")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Laya benchmark requires CUDA; refusing CPU fallback")
    volume.reload()
    data_dir = Path("/vol/datasets") / dataset_id
    test_path = data_dir / "test.jsonl"
    include_test = phase == "final"
    if test_path.exists() != include_test:
        raise ValueError("Test split must be physically absent during tuning")
    output = Path("/vol/runs/laya_temporal_v1") / phase / run_id
    command = [sys.executable, "-u", "/app/train_laya_temporal_v1.py",
               "--data-dir", str(data_dir), "--output-dir", str(output), "--run-id", run_id,
               "--encoder-learning-rate", str(encoder_lr), "--epochs", str(epochs),
               "--seed", str(seed), "--batch-size", str(batch_size)]
    if include_test:
        command.append("--include-test")
    try:
        subprocess.run(command, check=True, timeout=13800)
        result = {name: json.loads((output / name).read_text())
                  for name in ("run_info.json", "metrics.json")}
        result["validation_predictions"] = (output / "validation_predictions.csv").read_text()
        test_predictions = output / "test_predictions.csv"
        result["test_predictions"] = test_predictions.read_text() if test_predictions.exists() else None
        return result
    finally:
        volume.commit()


def _upload(typed_dir: Path, dataset_id: str, names: list[str]) -> None:
    with volume.batch_upload(force=True) as upload:
        for name in names:
            upload.put_file(typed_dir / name, f"/datasets/{dataset_id}/{name}")


@app.local_entrypoint(name="submit_sweep")
def submit_sweep(data_dir: str = "data/processed/cicd_temporal",
                 typed_dir: str = "data/typed/temporal_v1") -> None:
    source, freeze = verify_local(Path(typed_dir).resolve(), Path(data_dir).resolve(), include_test=False)
    typed = Path(typed_dir).resolve()
    names = ["benchmark.json", "train.jsonl", "validation.jsonl"]
    dataset_id = hashlib.sha256(b"".join((typed / name).read_bytes() for name in names) + b"laya-sweep").hexdigest()
    run_root = ROOT / "runs/laya_temporal_v1" / SWEEP_NAME
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "submitted_jobs.json"
    if plan_path.exists():
        raise FileExistsError(f"Laya sweep plan already exists: {plan_path}")
    _upload(typed, dataset_id, names)
    plan = {"benchmark_id": freeze["benchmark_id"], "dataset_id": dataset_id,
            "test_split_uploaded": False,
            "upstream_commit": UPSTREAM_COMMIT, "model_revision": MODEL_REVISION,
            "matrix": [{"encoder_learning_rate": lr, "epochs": ep, "head_learning_rate": 1e-4,
                        "seed": 2026, "batch_size": 8}
                       for lr in (1e-5, 2.5e-5, 5e-5) for ep in (2, 4)],
            "jobs": []}
    for config in plan["matrix"]:
        lr_slug = format(config["encoder_learning_rate"], "g").replace(".", "p")
        run_id = "laya-v1-sweep-lr{}-ep{}-s2026-{}-{}".format(
            lr_slug, config["epochs"],
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid4().hex[:5])
        call = train_one.spawn(dataset_id, "sweep", run_id, config["encoder_learning_rate"],
                               config["epochs"], 2026, config["batch_size"])
        plan["jobs"].append({**config, "run_id": run_id, "function_call_id": call.object_id,
                             "status": "submitted"})
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Submitted Laya validation-only config: {run_id}", flush=True)
    (run_root / "experiment_contract.json").write_text(json.dumps({
        "benchmark_id": freeze["benchmark_id"], "split_hashes": freeze["split_sha256"],
        "train_rows": source["splits"]["train"]["rows"],
        "validation_rows": source["splits"]["validation"]["rows"],
        "test_rows_accessible_to_tuning": 0, "matrix": plan["matrix"],
        "selection_metric": "validation average precision", "model_revision": MODEL_REVISION,
        "upstream_commit": UPSTREAM_COMMIT,
    }, indent=2) + "\n")
    print(json.dumps({"benchmark_id": freeze["benchmark_id"], "test_uploaded": False,
                      "jobs": plan["jobs"]}, indent=2))


@app.local_entrypoint(name="collect_sweep")
def collect_sweep() -> None:
    run_root = ROOT / "runs/laya_temporal_v1" / SWEEP_NAME
    plan_path = run_root / "submitted_jobs.json"
    plan = json.loads(plan_path.read_text())
    results, pending, failed = [], [], []
    for job in plan["jobs"]:
        if job["status"] == "collected":
            continue
        try:
            result = modal.FunctionCall.from_id(job["function_call_id"]).get(timeout=0)
        except TimeoutError:
            pending.append(job["run_id"])
            continue
        except Exception as exc:
            job["status"], job["error"] = "failed", str(exc)
            failed.append(job["run_id"])
            continue
        results.append((job, result))
    if pending or failed:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"pending": pending, "failed": failed}, indent=2))
        return
    for job, result in results:
        if result["metrics.json"].get("test") is not None or result["test_predictions"] is not None:
            raise ValueError("Laya tuning job unexpectedly accessed the test split")
        output = run_root / job["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        for name in ("run_info.json", "metrics.json"):
            (output / name).write_text(json.dumps(result[name], indent=2) + "\n")
        (output / "validation_predictions.csv").write_text(result["validation_predictions"])
        job["status"] = "collected"
    if any(job["status"] != "collected" for job in plan["jobs"]):
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        return
    rows = []
    for job in plan["jobs"]:
        result = json.loads((run_root / job["run_id"] / "metrics.json").read_text())
        info = json.loads((run_root / job["run_id"] / "run_info.json").read_text())
        validation = result["validation"]
        rows.append({"run_id": job["run_id"], "learning_rate": job["encoder_learning_rate"],
                     "epochs": job["epochs"], "validation_ap": validation["pr_auc"],
                     "validation_roc_auc": validation["roc_auc"], "validation_brier": validation["brier"],
                     "validation_ece": validation["ece_15"],
                     "validation_recall_at_fpr_1pct": validation["recall_at_fpr_1pct"],
                     "validation_recall_at_fpr_5pct": validation["recall_at_fpr_5pct"],
                     "training_seconds": info["training_seconds"],
                     "peak_gpu_memory_bytes": info["peak_gpu_memory_bytes"],
                     "numerical_health_pass": all(v is not False for v in info["numerical_health"].values()
                                                   if isinstance(v, bool))})
    winner = max(rows, key=lambda row: row["validation_ap"])
    summary = {"benchmark_id": plan["benchmark_id"], "selection_metric": "validation average precision",
               "test_used_for_selection": False, "winner": winner, "runs": rows}
    (run_root / "sweep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (run_root / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    md = ["# Laya Temporal-v1 Validation Sweep", "",
          "Selected solely by final validation AP; test split was not uploaded to tuning jobs.", "",
          "| Run ID | Encoder LR | Epochs | Val AP | ROC-AUC | Brier | ECE | Recall@1% FPR | Recall@5% FPR | Seconds | Peak VRAM GB | Health |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        md.append("| {run_id} | {learning_rate:g} | {epochs} | {validation_ap:.4f} | {validation_roc_auc:.4f} | {validation_brier:.4f} | {validation_ece:.4f} | {validation_recall_at_fpr_1pct:.4f} | {validation_recall_at_fpr_5pct:.4f} | {training_seconds:.0f} | {vram:.2f} | {health} |".format(
            **row, vram=row["peak_gpu_memory_bytes"] / 1024**3, health="pass" if row["numerical_health_pass"] else "fail"))
    (run_root / "sweep_summary.md").write_text("\n".join(md) + "\n")
    winner_path = run_root / "winner_config.json"
    winner_path.write_text(json.dumps({
        "benchmark_id": plan["benchmark_id"], "selection_metric": "validation average precision",
        "selected_run_id": winner["run_id"], "validation_ap": winner["validation_ap"],
        "encoder_learning_rate": winner["learning_rate"], "epochs": winner["epochs"],
        "head_learning_rate": 1e-4, "batch_size": 8, "seed": 2026,
        "upstream_commit": UPSTREAM_COMMIT, "model_revision": MODEL_REVISION,
    }, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"winner": winner, "summary": str(run_root / "sweep_summary.md")}, indent=2))


@app.local_entrypoint(name="submit_final")
def submit_final(data_dir: str = "data/processed/cicd_temporal",
                 typed_dir: str = "data/typed/temporal_v1") -> None:
    source, freeze = verify_local(Path(typed_dir).resolve(), Path(data_dir).resolve(), include_test=True)
    sweep_root = ROOT / "runs/laya_temporal_v1" / SWEEP_NAME
    winner = json.loads((sweep_root / "winner_config.json").read_text())
    if winner.get("benchmark_id") != freeze["benchmark_id"] or winner.get("selection_metric") != "validation average precision":
        raise ValueError("Laya winner config is not from this frozen validation-only sweep")
    lr, epochs = float(winner["encoder_learning_rate"]), int(winner["epochs"])
    if (lr, epochs) not in {(1e-5, 2), (1e-5, 4), (2.5e-5, 2), (2.5e-5, 4), (5e-5, 2), (5e-5, 4)}:
        raise ValueError("Laya selected config falls outside the pre-registered matrix")
    run_root = ROOT / "runs/laya_temporal_v1/final"
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "submitted_jobs.json"
    if plan_path.exists():
        raise FileExistsError(f"Laya final seed plan already exists: {plan_path}")
    typed = Path(typed_dir).resolve()
    names = ["benchmark.json", "train.jsonl", "validation.jsonl", "test.jsonl"]
    dataset_id = hashlib.sha256(b"".join((typed / name).read_bytes() for name in names) + b"laya-final").hexdigest()
    _upload(typed, dataset_id, names)
    plan = {"benchmark_id": freeze["benchmark_id"], "dataset_id": dataset_id,
            "selection": winner, "test_uploaded_after_selection": True, "jobs": []}
    for seed in (17, 42, 73):
        lr_slug = format(lr, "g").replace(".", "p")
        run_id = "laya-v1-final-lr{}-ep{}-s{}-{}-{}".format(
            lr_slug, epochs, seed, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            uuid4().hex[:5])
        call = train_one.spawn(dataset_id, "final", run_id, lr, epochs, seed, 8)
        plan["jobs"].append({"run_id": run_id, "seed": seed, "function_call_id": call.object_id,
                             "encoder_learning_rate": lr, "epochs": epochs, "status": "submitted"})
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Submitted Laya final seed {seed}: {run_id}", flush=True)
    (run_root / "selection.json").write_text(json.dumps(plan["selection"], indent=2) + "\n")
    print(json.dumps({"benchmark_id": freeze["benchmark_id"], "test_hash": freeze["split_sha256"]["test"],
                      "jobs": plan["jobs"]}, indent=2))


@app.local_entrypoint(name="collect_final")
def collect_final() -> None:
    run_root = ROOT / "runs/laya_temporal_v1/final"
    plan_path = run_root / "submitted_jobs.json"
    plan = json.loads(plan_path.read_text())
    results, pending, failed = [], [], []
    for job in plan["jobs"]:
        if job["status"] == "collected":
            continue
        try:
            result = modal.FunctionCall.from_id(job["function_call_id"]).get(timeout=0)
        except TimeoutError:
            pending.append(job["run_id"])
            continue
        except Exception as exc:
            job["status"], job["error"] = "failed", str(exc)
            failed.append(job["run_id"])
            continue
        results.append((job, result))
    if pending or failed:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"pending": pending, "failed": failed}, indent=2))
        return
    for job, result in results:
        if result["metrics.json"].get("test") is None or result["test_predictions"] is None:
            raise ValueError(f"Laya final test artifacts missing for {job['run_id']}")
        output = run_root / job["run_id"]
        output.mkdir(parents=True, exist_ok=True)
        for name in ("run_info.json", "metrics.json"):
            (output / name).write_text(json.dumps(result[name], indent=2) + "\n")
        (output / "validation_predictions.csv").write_text(result["validation_predictions"])
        (output / "test_predictions.csv").write_text(result["test_predictions"])
        job["status"] = "collected"
    if any(job["status"] != "collected" for job in plan["jobs"]):
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        return

    import numpy as np
    metric_paths = {
        "AP": ("pr_auc",), "ROC-AUC": ("roc_auc",), "Brier": ("brier",), "ECE": ("ece_15",),
        "Recall@1% FPR": ("recall_at_fpr_1pct",), "Recall@5% FPR": ("recall_at_fpr_5pct",),
        "Recall@80% precision": ("recall_at_precision", "0.8"),
        "Recall@90% precision": ("recall_at_precision", "0.9"),
        "Recall@95% precision": ("recall_at_precision", "0.95"),
    }
    records = []
    for job in plan["jobs"]:
        output = run_root / job["run_id"]
        info = json.loads((output / "run_info.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())["test"]
        predictions = list(csv.DictReader((output / "test_predictions.csv").open(encoding="utf-8")))
        y = np.asarray([int(row["label"]) for row in predictions])
        p = np.asarray([float(row["p_failure"]) for row in predictions])
        groups = {"safe": p < 0.10, "risky": p >= 0.90, "abstain": (p >= 0.10) & (p < 0.90)}
        breakdown = {}
        for name, mask in groups.items():
            count = int(mask.sum())
            breakdown[name] = {"count": count, "fraction": float(mask.mean()),
                               "observed_failure_rate": float(y[mask].mean()) if count else None}
        breakdown["risk_failure_recall"] = float(np.sum(groups["risky"] & (y == 1)) / y.sum())
        health = info["numerical_health"]
        records.append({
            "run_id": job["run_id"], "seed": job["seed"], "test": metrics,
            "training_seconds": info["training_seconds"],
            "inference_rows_per_second": info["inference_rows_per_second"],
            "peak_gpu_memory_bytes": info["peak_gpu_memory_bytes"],
            "calibration_temperature": info["calibration_temperature"],
            "numerical_health": health,
            "numerical_health_pass": all(value is not False for value in health.values()
                                          if isinstance(value, bool)),
            "safe_risky_abstain": breakdown,
        })
    summary = {"benchmark_id": plan["benchmark_id"],
               "test_prevalence": records[0]["test"]["failure_prevalence"],
               "selected_config": plan["selection"], "seeds": records, "mean_plus_minus_sd": {}}
    for label, path in metric_paths.items():
        values = []
        for record in records:
            value = record["test"]
            for key in path:
                value = value[key]
            values.append(float(value))
        summary["mean_plus_minus_sd"][label] = {"mean": statistics.mean(values),
                                                 "sd": statistics.stdev(values), "per_seed": values}
    for label, key in (("Training time seconds", "training_seconds"),
                       ("Inference rows per second", "inference_rows_per_second"),
                       ("Peak GPU VRAM bytes", "peak_gpu_memory_bytes")):
        values = [float(record[key]) for record in records]
        summary["mean_plus_minus_sd"][label] = {"mean": statistics.mean(values),
                                                 "sd": statistics.stdev(values), "per_seed": values}
    summary["numerical_health"] = {
        "all_seeds_pass": all(record["numerical_health_pass"] for record in records),
        "per_seed": [{"seed": record["seed"], "pass": record["numerical_health_pass"],
                      "diagnostics": record["numerical_health"]} for record in records],
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    baselines = json.loads((ROOT / "analysis/temporal_baselines/metrics.json").read_text())["models"]
    comparison = {"test_prevalence": summary["test_prevalence"], "models": {
        "Laya rich typed choice (3-seed mean)": {
            name: summary["mean_plus_minus_sd"][name]["mean"] for name in metric_paths},
        "Rich TF-IDF + structured logistic": baselines["logistic_tfidf_structured"]["test_at_0_5"],
        "Structured HistGradientBoosting": baselines["hist_gradient_boosting_structured"]["test_at_0_5"],
        "Rich-state ModernBERT (3-seed mean)": json.loads(
            (ROOT / "runs/temporal_v1_rich_modernbert/final/summary.json").read_text())[
                "mean_plus_minus_sd"],
    }}
    (run_root / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"summary": str(run_root / "summary.json"),
                      "comparison": str(run_root / "baseline_comparison.json")}, indent=2))

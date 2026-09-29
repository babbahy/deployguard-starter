from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[1]
VOLUME_NAME = "deployguard-training"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("transformers==4.57.6", "datasets==4.8.5", "accelerate==1.15.0",
                 "pandas==2.3.3", "numpy==2.4.6", "scikit-learn==1.9.1")
    .env({"PYTHONPATH": "/app/src", "HF_HOME": "/vol/hf-cache",
          "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(ROOT / "src", "/app/src", ignore=["**/__pycache__/**", "**/*.egg-info/**"])
    .add_local_file(ROOT / "scripts/train_temporal_modernbert.py", "/app/train_temporal_modernbert.py")
)
app = modal.App("deployguard-temporal-v1-modernbert")
MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def verify_local(data_dir: Path, include_test: bool = True) -> tuple[dict, dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    freeze = manifest.get("benchmark_freeze")
    lock = json.loads((data_dir / "benchmark_lock.json").read_text())
    if not freeze or lock != freeze or freeze.get("version") != "temporal-v1":
        raise ValueError("Expected locked temporal-v1 benchmark")
    checked_splits = ("train", "validation", "test") if include_test else ("train", "validation")
    for split in checked_splits:
        digest = hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if digest != freeze["split_sha256"].get(split):
            raise ValueError(f"Frozen {split} split hash mismatch")
    return manifest, freeze


@app.function(image=image, gpu="L4", cpu=2, memory=16384, volumes={"/vol": volume},
              timeout=14400, retries=0, min_containers=0, max_containers=1, scaledown_window=2)
def train_one(dataset_id: str, run_id: str, learning_rate: float, epochs: float,
              seed: int, batch_size: int, skip_test: bool,
              resume_checkpoint_step: int = 0) -> dict:
    import torch

    if not re.fullmatch(r"[a-f0-9]{64}", dataset_id) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid dataset or run identifier")
    if not 1e-7 <= learning_rate <= 1e-4 or not 1 <= epochs <= 8 or not 1 <= batch_size <= 16:
        raise ValueError("Invalid training configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing CPU fallback")
    volume.reload()
    output = Path("/vol/runs") / run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "/app/train_temporal_modernbert.py",
        "--data-dir", f"/vol/datasets/{dataset_id}", "--output-dir", str(output),
        "--run-id", run_id, "--model", MODEL, "--revision", REVISION,
        "--require-cuda", "--threads", "2", "--batch-size", str(batch_size),
        "--max-length", "512", "--epochs", str(epochs),
        "--learning-rate", str(learning_rate), "--seed", str(seed)]
    if skip_test:
        command.append("--skip-test")
    if resume_checkpoint_step:
        if not skip_test or resume_checkpoint_step < 1:
            raise ValueError("Checkpoint resumes are allowed only in validation-only mode")
        checkpoint = output / "trainer" / f"checkpoint-{resume_checkpoint_step}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Missing remote resume checkpoint: {checkpoint}")
        command.extend(["--resume-from-checkpoint", str(checkpoint)])
    try:
        subprocess.run(command, check=True, timeout=13800)
        result = {name: json.loads((output / name).read_text())
                  for name in ("run_info.json", "metrics.json")}
        result["validation_predictions"] = (output / "validation_predictions.csv").read_text()
        test_path = output / "test_predictions.csv"
        result["test_predictions"] = test_path.read_text() if test_path.exists() else None
        return result
    finally:
        volume.commit()


def _sweep_config_from_id(run_id: str):
    match = re.search(r"-lr(.+?)-ep([0-9.]+)-s", run_id)
    return (float(match.group(1)), float(match.group(2))) if match else None


def _save_sweep_result(run_root: Path, run_id: str, result: dict, completed: list[dict]) -> None:
    if result["test_predictions"] is not None or result["metrics.json"].get("test") is not None:
        raise ValueError("Sweep job unexpectedly accessed the test split")
    output = run_root / run_id
    output.mkdir(parents=True, exist_ok=True)
    for name in ("run_info.json", "metrics.json"):
        (output / name).write_text(json.dumps(result[name], indent=2) + "\n")
    (output / "validation_predictions.csv").write_text(result["validation_predictions"])
    if not any(item["run_id"] == run_id for item in completed):
        completed.append({"run_id": run_id, "metrics": result["metrics.json"]})
    (run_root / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")


@app.local_entrypoint(name="submit_remaining")
def submit_remaining(data_dir: str = "data/processed/cicd_temporal",
                     resume_run_id: str = "mb-rich-sweep-lr2e-05-ep5-s2026-20260927T215914Z-536d4",
                     resume_checkpoint_step: int = 2575, batch_size: int = 16,
                     seed: int = 2026) -> None:
    directory = Path(data_dir).resolve()
    _, freeze = verify_local(directory, include_test=False)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    dataset_id = hashlib.sha256(manifest_bytes + b"sweep").hexdigest()
    run_root = ROOT / "runs" / "temporal_v1_rich_modernbert" / "sweep"
    run_root.mkdir(parents=True, exist_ok=True)
    upload_names = ["manifest.json", "benchmark_lock.json", "train.jsonl", "validation.jsonl"]
    with volume.batch_upload(force=True) as upload:
        for name in upload_names:
            upload.put_file(directory / name, "/datasets/{}/{}".format(dataset_id, name))

    progress_path = run_root / "progress.json"
    completed = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed_configs = {_sweep_config_from_id(item["run_id"]) for item in completed}
    completed_configs.discard(None)
    resume_config = _sweep_config_from_id(resume_run_id)
    configs = [(lr, ep) for lr in (1e-5, 2e-5, 5e-5) for ep in (2, 3, 5)]
    if resume_config != (2e-5, 5.0) or resume_config in completed_configs:
        raise ValueError("The supplied resume run must be the unfinished 2e-5 x 5 config")
    if resume_checkpoint_step < 1:
        raise ValueError("A positive saved checkpoint step is required")

    plan_path = run_root / "submitted_jobs.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {
        "benchmark_id": freeze["benchmark_id"], "jobs": []
    }
    submitted_ids = {job["run_id"] for job in plan["jobs"]}
    submitted_configs = {_sweep_config_from_id(run_id) for run_id in submitted_ids}
    job_specs = []
    if resume_run_id not in submitted_ids:
        job_specs.append((resume_run_id, resume_config, resume_checkpoint_step))
    for config in configs:
        if config in completed_configs or config == resume_config or config in submitted_configs:
            continue
        lr, ep = config
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = "mb-rich-sweep-lr{}-ep{}-s{}-{}-{}".format(
            format(lr, "g"), format(ep, "g"), seed, stamp, uuid4().hex[:5]
        )
        job_specs.append((run_id, config, 0))

    for run_id, (lr, ep), checkpoint_step in job_specs:
        print("Submitting {}".format(run_id), flush=True)
        call = train_one.spawn(dataset_id, run_id, lr, ep, seed, batch_size, True, checkpoint_step)
        plan["jobs"].append({
            "run_id": run_id, "learning_rate": lr, "epochs": ep, "seed": seed,
            "resume_checkpoint_step": checkpoint_step,
            "function_call_id": call.object_id, "status": "submitted",
        })
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    print(json.dumps({"benchmark_id": freeze["benchmark_id"], "submitted": plan["jobs"]}, indent=2))
    print("Modal owns these jobs now; collect later with phase=collect.")


@app.local_entrypoint(name="collect_remaining")
def collect_remaining() -> None:
    run_root = ROOT / "runs" / "temporal_v1_rich_modernbert" / "sweep"
    plan_path = run_root / "submitted_jobs.json"
    if not plan_path.exists():
        raise FileNotFoundError("No submitted_jobs.json exists")
    plan = json.loads(plan_path.read_text())
    progress_path = run_root / "progress.json"
    completed = json.loads(progress_path.read_text()) if progress_path.exists() else []
    pending, failed, collected = [], [], []
    for job in plan["jobs"]:
        if job["status"] == "collected":
            continue
        call = modal.FunctionCall.from_id(job["function_call_id"])
        try:
            result = call.get(timeout=0)
        except TimeoutError:
            pending.append(job["run_id"])
            continue
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            failed.append(job["run_id"])
            continue
        _save_sweep_result(run_root, job["run_id"], result, completed)
        job["status"] = "collected"
        collected.append(job["run_id"])
    progress_path.write_text(json.dumps(completed, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"collected": collected, "pending": pending, "failed": failed}, indent=2))


@app.local_entrypoint(name="submit_final")
def submit_final(data_dir: str = "data/processed/cicd_temporal", batch_size: int = 16) -> None:
    directory = Path(data_dir).resolve()
    _, freeze = verify_local(directory, include_test=True)
    if freeze["benchmark_id"] != "temporal-v1-3e9cc5c049399d0b":
        raise ValueError("Final evaluation requires the locked temporal-v1 benchmark")
    if freeze["split_sha256"]["test"] != "1410c9dcdc2c87ab503178f3f9749a674bd34959b402ca04ba16d3a0e22af003":
        raise ValueError("Locked temporal-v1 test hash mismatch")

    sweep_path = ROOT / "runs/temporal_v1_rich_modernbert/sweep/progress.json"
    sweep = json.loads(sweep_path.read_text())
    winner = max(sweep, key=lambda item: item["metrics"]["validation"]["pr_auc"])
    expected_run = "mb-rich-sweep-lr5e-05-ep3-s2026-20260928T091147Z-b9bca"
    if winner["run_id"] != expected_run or abs(winner["metrics"]["validation"]["pr_auc"] - 0.4598572578989274) > 1e-12:
        raise ValueError("Validation sweep winner differs from the frozen selection")

    run_root = ROOT / "runs/temporal_v1_rich_modernbert/final"
    run_root.mkdir(parents=True, exist_ok=True)
    plan_path = run_root / "submitted_jobs.json"
    if plan_path.exists():
        raise FileExistsError(f"Final seed plan already exists: {plan_path}")
    selection = {
        "selection_rule": "highest final validation PR-AUC across the prespecified nine configs",
        "selected_by": "validation only",
        "winning_sweep_run_id": winner["run_id"],
        "winning_validation_ap": winner["metrics"]["validation"]["pr_auc"],
        "learning_rate": 5e-5, "epochs": 3, "max_length": 512,
        "batch_size": batch_size, "seeds": [17, 42, 73],
        "dataset_version": "temporal-v1", "benchmark_id": freeze["benchmark_id"],
        "split_sha256": freeze["split_sha256"], "model": MODEL, "model_revision": REVISION,
        "product_decomposition": {
            "safe": "p_failure < 0.10", "risky": "p_failure >= 0.90",
            "abstain": "0.10 <= p_failure < 0.90", "use_for_model_selection": False,
        },
    }
    selection_path = run_root / "selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
        raise ValueError("Existing final selection contract differs; refusing to overwrite")
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    manifest_bytes = (directory / "manifest.json").read_bytes()
    dataset_id = hashlib.sha256(manifest_bytes + b"final").hexdigest()
    upload_names = ["manifest.json", "benchmark_lock.json", "train.jsonl", "validation.jsonl", "test.jsonl"]
    with volume.batch_upload(force=True) as upload:
        for name in upload_names:
            upload.put_file(directory / name, f"/datasets/{dataset_id}/{name}")
    plan = {"benchmark_id": freeze["benchmark_id"], "dataset_id": dataset_id,
            "selection": selection, "jobs": []}
    for seed in (17, 42, 73):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"mb-rich-final-lr5e-05-ep3-s{seed}-{stamp}-{uuid4().hex[:5]}"
        call = train_one.spawn(dataset_id, run_id, 5e-5, 3, seed, batch_size, False)
        plan["jobs"].append({"run_id": run_id, "seed": seed, "function_call_id": call.object_id,
                             "learning_rate": 5e-5, "epochs": 3, "status": "submitted"})
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Submitted fixed final seed {seed}: {run_id}", flush=True)
    print(json.dumps({"benchmark_id": freeze["benchmark_id"], "jobs": plan["jobs"]}, indent=2))


@app.local_entrypoint(name="collect_final")
def collect_final() -> None:
    run_root = ROOT / "runs/temporal_v1_rich_modernbert/final"
    plan_path = run_root / "submitted_jobs.json"
    plan = json.loads(plan_path.read_text())
    resolved, pending, failed = [], [], []
    for job in plan["jobs"]:
        if job["status"] == "collected":
            continue
        try:
            result = modal.FunctionCall.from_id(job["function_call_id"]).get(timeout=0)
        except TimeoutError:
            pending.append(job["run_id"])
            continue
        except Exception as exc:
            job["status"] = "failed"
            job["error"] = str(exc)
            failed.append(job["run_id"])
            continue
        resolved.append((job, result))
    if pending or failed:
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps({"pending": pending, "failed": failed, "collected_this_call": []}, indent=2))
        return

    # Keep every seed fixed and finished before looking at any test result.
    for job, result in resolved:
        if result["metrics.json"].get("test") is None or result["test_predictions"] is None:
            raise ValueError(f"Final test artifacts missing for {job['run_id']}")
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
    import pandas as pd
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
        predictions = pd.read_csv(output / "test_predictions.csv")
        p, y = predictions["p_failure"].to_numpy(), predictions["label"].to_numpy()
        groups = {"safe": p < 0.10, "risky": p >= 0.90, "abstain": (p >= 0.10) & (p < 0.90)}
        decomposition = {}
        for name, mask in groups.items():
            count = int(mask.sum())
            decomposition[name] = {"count": count, "fraction": float(mask.mean()),
                                    "observed_failure_rate": float(y[mask].mean()) if count else None}
        decomposition["risk_failure_recall"] = float(np.sum(groups["risky"] & (y == 1)) / y.sum())
        health = info["numerical_health"]
        records.append({
            "run_id": job["run_id"], "seed": job["seed"], "test": metrics,
            "training_seconds": info["training_seconds"],
            "inference_rows_per_second": info["inference_rows_per_second"],
            "inference_seconds": info["inference_seconds"],
            "peak_gpu_memory_bytes": info["peak_gpu_memory_bytes"],
            "numerical_health": health,
            "numerical_health_pass": all(health[key] for key in
                                          ("finite_loss", "finite_gradients", "finite_parameters")),
            "safe_risky_abstain": decomposition,
        })
    summary = {"benchmark_id": plan["benchmark_id"], "test_prevalence": records[0]["test"]["failure_prevalence"],
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
        "ModernBERT rich temporal-v1 (3-seed mean)": {
            name: summary["mean_plus_minus_sd"][name]["mean"] for name in metric_paths},
        "Frozen rich TF-IDF + structured logistic": baselines["logistic_tfidf_structured"]["test_at_0_5"],
        "Frozen structured HistGradientBoosting": baselines["hist_gradient_boosting_structured"]["test_at_0_5"],
    }}
    (run_root / "baseline_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"collected": [record["run_id"] for record in records],
                      "summary": str(run_root / "summary.json"),
                      "comparison": str(run_root / "baseline_comparison.json")}, indent=2))
@app.local_entrypoint()
def main(phase: str = "sweep", data_dir: str = "data/processed/cicd_temporal",
         learning_rate: float = 0.0, epochs: float = 0.0, batch_size: int = 16,
         seed: int = 2026):
    if phase not in {"sweep", "final"}:
        raise ValueError("phase must be sweep or final")
    directory = Path(data_dir).resolve()
    _, freeze = verify_local(directory, include_test=phase == "final")
    manifest_bytes = (directory / "manifest.json").read_bytes()
    dataset_id = hashlib.sha256(manifest_bytes + phase.encode()).hexdigest()
    run_root = ROOT / "runs" / "temporal_v1_rich_modernbert" / phase
    run_root.mkdir(parents=True, exist_ok=True)
    upload_names = ["manifest.json", "benchmark_lock.json", "train.jsonl", "validation.jsonl"]
    if phase == "final":
        upload_names.append("test.jsonl")
    with volume.batch_upload(force=True) as upload:
        for name in upload_names:
            upload.put_file(directory / name, f"/datasets/{dataset_id}/{name}")
    if phase == "sweep":
        configs = [(lr, ep) for lr in (1e-5, 2e-5, 5e-5) for ep in (2, 3, 5)]
        seeds = (seed,)
    else:
        if learning_rate <= 0 or epochs <= 0:
            raise ValueError("final phase requires --learning-rate and --epochs")
        configs, seeds = [(learning_rate, epochs)], (17, 42, 73)
    completed = []
    for lr, ep in configs:
        for run_seed in seeds:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_id = f"mb-rich-{phase}-lr{lr:g}-ep{ep:g}-s{run_seed}-{stamp}-{uuid4().hex[:5]}"
            print(f"Starting {run_id} on {freeze['benchmark_id']}", flush=True)
            result = train_one.remote(dataset_id, run_id, lr, ep, run_seed,
                                      batch_size, phase == "sweep")
            output = run_root / run_id
            output.mkdir(parents=True, exist_ok=False)
            for name in ("run_info.json", "metrics.json"):
                (output / name).write_text(json.dumps(result[name], indent=2) + "\n")
            (output / "validation_predictions.csv").write_text(result["validation_predictions"])
            if result["test_predictions"] is not None:
                (output / "test_predictions.csv").write_text(result["test_predictions"])
            completed.append({"run_id": run_id, "metrics": result["metrics.json"]})
            (run_root / "progress.json").write_text(json.dumps(completed, indent=2) + "\n")
    print(json.dumps({"phase": phase, "benchmark_id": freeze["benchmark_id"],
                      "test_sha256": freeze["split_sha256"]["test"], "runs": completed}, indent=2))
    print(f"Artifacts: {run_root}")
    print(f"Modal checkpoints: volume {VOLUME_NAME}, /runs/<run_id>/best_model")

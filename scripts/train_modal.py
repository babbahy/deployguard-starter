"""Run the canonical ModernBERT trainer on one Modal L4 GPU."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "answerdotai/ModernBERT-base"
REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
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
    .add_local_file(ROOT / "scripts/train_modernbert.py", "/app/train_modernbert.py")
)
app = modal.App("deployguard-modernbert")


def validate_data(data_dir: Path) -> tuple[bytes, dict]:
    manifest_bytes = (data_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Dataset must pass the preprocessing leakage audit")
    if set(manifest.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test splits")
    for split, info in manifest["splits"].items():
        digest = hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"{split} changed since the audit; rerun preprocessing")
    return manifest_bytes, manifest


@app.function(
    image=image, gpu="L4", cpu=2, memory=16384, volumes={"/vol": volume},
    timeout=5400, retries=0, min_containers=0, max_containers=1, scaledown_window=2,
)
def train(dataset_id: str, run_id: str, max_steps: int, batch_size: int,
          learning_rate: float) -> dict:
    import re
    import subprocess
    import sys
    import torch

    if not re.fullmatch(r"[a-f0-9]{64}", dataset_id) or not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("Invalid dataset or run identifier")
    if not 1 <= max_steps <= 10000 or not 1 <= batch_size <= 16:
        raise ValueError("Training limits: 1-10,000 updates, batch size 1-16")
    if not 1e-7 <= learning_rate <= 1e-4:
        raise ValueError("learning rate must be between 1e-7 and 1e-4")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not provide a working CUDA GPU")
    volume.reload()
    output = Path("/vol/runs") / run_id
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable, "-u", "/app/train_modernbert.py",
        "--data-dir", f"/vol/datasets/{dataset_id}", "--output-dir", str(output),
        "--model", MODEL, "--revision", REVISION, "--require-cuda",
        "--threads", "2", "--batch-size", str(batch_size), "--max-length", "128",
        "--max-steps", str(max_steps), "--learning-rate", str(learning_rate),
    ]
    try:
        subprocess.run(command, check=True, timeout=5200)
        result = {name: json.loads((output / name).read_text())
                  for name in ("run_info.json", "test_metrics.json")}
        result["run_id"] = run_id
        return result
    finally:
        volume.commit()


@app.function(
    image=image, gpu="L4", cpu=2, memory=16384, volumes={"/vol": volume},
    timeout=300, retries=0, min_containers=0, max_containers=1, scaledown_window=2,
)
def verify_checkpoint(dataset_id: str, run_id: str) -> dict:
    import json
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from deployguard.evaluation.metrics import binary_metrics

    volume.reload()
    data_dir = Path("/vol/datasets") / dataset_id
    model_dir = Path("/vol/runs") / run_id / "best_model"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for split, info in manifest["splits"].items():
        import hashlib
        digest = hashlib.sha256((data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"{split} checksum mismatch during checkpoint verification")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, attn_implementation="sdpa", reference_compile=False,
    ).cuda().eval()
    rows = [json.loads(line) for line in (data_dir / "test.jsonl").read_text().splitlines()]
    state = [row["state"] for row in rows]
    probs = []
    for start in range(0, len(rows), 8):
        tokens = tokenizer(state[start:start + 8], return_tensors="pt", padding=True,
                           truncation=True, max_length=128).to("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**tokens).logits
            probs.extend(logits.float().softmax(-1)[:, 1].cpu().tolist())
    predictions = torch.tensor(probs)
    return {
        "run_id": run_id, "device": torch.cuda.get_device_name(0),
        "rows": len(rows), "finite_probabilities": bool(torch.isfinite(predictions).all()),
        "probability_min": float(predictions.min()), "probability_max": float(predictions.max()),
        "unique_probabilities": int(predictions.unique().numel()),
        "predicted_failures_at_0_5": int((predictions >= 0.5).sum()),
        "failure_recall": sum(int(p >= 0.5 and row["label"] == 1)
                               for p, row in zip(probs, rows)) / max(1, sum(row["label"] == 1 for row in rows)),
        "rescored_test_metrics": binary_metrics([row["label"] for row in rows], probs),
    }


@app.local_entrypoint()
def main(data_dir: str = "data/processed/cicd_smoke", max_steps: int = 100, batch_size: int = 8,
         learning_rate: float = 2e-5, verify_run: str = ""):
    if verify_run:
        submission_path = ROOT / "runs" / verify_run / "submission.json"
        submission = json.loads(submission_path.read_text())
        if submission["run_id"] != verify_run:
            raise ValueError("Run ID differs from submission metadata")
        print(json.dumps(verify_checkpoint.remote(submission["dataset_id"], verify_run), indent=2))
        return
    if not 1 <= max_steps <= 10000 or not 1 <= batch_size <= 16:
        raise ValueError("Training limits: 1-10,000 updates, batch size 1-16")
    if not 1e-7 <= learning_rate <= 1e-4:
        raise ValueError("learning rate must be between 1e-7 and 1e-4")
    directory = Path(data_dir).resolve()
    manifest_bytes, manifest = validate_data(directory)
    dataset_id = hashlib.sha256(manifest_bytes).hexdigest()
    run_id = "modernbert-l4-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "submission.json").write_text(json.dumps({
        "run_id": run_id, "volume": VOLUME_NAME, "dataset_id": dataset_id,
        "max_steps": max_steps, "batch_size": batch_size,
        "learning_rate": learning_rate, "timeout_seconds": 5400,
    }, indent=2) + "\n")
    print(f"Uploading audited splits: { {k: v['rows'] for k, v in manifest['splits'].items()} }")
    with volume.batch_upload(force=True) as upload:
        for filename in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
            upload.put_file(directory / filename, f"/datasets/{dataset_id}/{filename}")
    print(f"Starting {run_id}: one L4, {max_steps} updates, {batch_size} per batch")
    result = train.remote(dataset_id, run_id, max_steps, batch_size, learning_rate)
    for name in ("run_info.json", "test_metrics.json"):
        (output / name).write_text(json.dumps(result[name], indent=2) + "\n")
    print(json.dumps(result["test_metrics.json"], indent=2))
    print(f"Local reports: {output}")
    print(f"Checkpoint: Modal Volume {VOLUME_NAME}, /runs/{run_id}/best_model")
    print(f"Download: modal volume get {VOLUME_NAME} /runs/{run_id}/best_model {output}/best_model")

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Value, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from deployguard.evaluation.report import evaluation_report
from deployguard.evaluation.health import NumericalHealthCallback, check_finite_loss, checked_probabilities
from deployguard.evaluation.metrics import binary_metrics
from deployguard.data.temporal import TEMPORAL_V1_STATE_FIELDS, rich_temporal_v1_state


class GuardedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is not None:
            if not torch.isfinite(labels).all().item() or not torch.all((labels == 0) | (labels == 1)).item():
                raise ValueError("Labels must be finite binary class indices")
            inputs["labels"] = labels.long()
        outputs = model(**inputs)
        check_finite_loss(outputs.loss)
        if not torch.isfinite(outputs.logits).all().item():
            raise FloatingPointError("Non-finite logits during forward pass")
        return (outputs.loss, outputs) if return_outputs else outputs.loss


def verify_v1(data_dir: Path, include_test: bool = True) -> tuple[dict, dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    freeze = manifest.get("benchmark_freeze")
    lock_path = data_dir / "benchmark_lock.json"
    if not freeze or freeze.get("benchmark_id", "").startswith("temporal-v1-") is False:
        raise ValueError("Training requires frozen temporal benchmark v1")
    if not lock_path.exists() or json.loads(lock_path.read_text()) != freeze:
        raise ValueError("Benchmark lock and manifest differ")
    checked_splits = ("train", "validation", "test") if include_test else ("train", "validation")
    for split in checked_splits:
        path = data_dir / f"{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != freeze["split_sha256"].get(split):
            raise ValueError(f"Frozen temporal-v1 {split} checksum mismatch")
    dataset_digest = hashlib.sha256(json.dumps({"benchmark_id": freeze["benchmark_id"], "split_sha256": freeze["split_sha256"]}, sort_keys=True).encode()).hexdigest()
    return manifest, {**freeze, "dataset_sha256": dataset_digest}


def export_predictions(frame: pd.DataFrame, p_failure: np.ndarray, path: Path, run_id: str, model_id: str) -> None:
    if len(frame) != len(p_failure) or not np.isfinite(p_failure).all():
        raise FloatingPointError("Prediction export received invalid/non-finite probabilities")
    if np.any((p_failure < 0) | (p_failure > 1)):
        raise FloatingPointError("Probability outside [0, 1]")
    exported = pd.DataFrame({
        "id": frame.id.astype(str), "repo": frame.repo.astype(str),
        "timestamp": frame.created_at.astype(str), "label": frame.label.astype(int),
        "p_failure": p_failure.astype(float), "predicted_label": (p_failure >= 0.5).astype(int),
        "model_id": model_id, "run_id": run_id,
    })
    exported.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded ModernBERT training on frozen temporal benchmark v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--revision", default="8949b909ec900327062f0ebf497f51aef5e6f0c8")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA; refusing to fall back to CPU")
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        raise RuntimeError("ModernBERT benchmark runs require GPU")

    manifest, freeze = verify_v1(args.data_dir, include_test=not args.skip_test)
    if args.resume_from_checkpoint is not None and not args.resume_from_checkpoint.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume_from_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_from_checkpoint is not None)
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.cuda.reset_peak_memory_stats()
    loaded_splits = ("train", "validation", "test") if not args.skip_test else ("train", "validation")
    raw = load_dataset("json", data_files={
        name: str(args.data_dir / f"{name}.jsonl") for name in loaded_splits
    })
    raw = raw.cast_column("label", Value("int64"))
    source_frames = {name: raw[name].to_pandas() for name in raw}
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    def tokenize(batch):
        states = [
            rich_temporal_v1_state({name: batch[name][i] for name in TEMPORAL_V1_STATE_FIELDS})
            for i in range(len(batch["repo"]))
        ]
        return tokenizer(states, truncation=True, max_length=args.max_length)

    remove_columns = [column for column in raw["train"].column_names if column != "label"]
    tokenized = raw.map(tokenize, batched=True, remove_columns=remove_columns)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision, num_labels=2,
        problem_type="single_label_classification",
        id2label={0: "success", 1: "failure"},
        label2id={"success": 0, "failure": 1},
        attn_implementation="sdpa", reference_compile=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probabilities = checked_probabilities(torch.as_tensor(logits)).cpu().numpy()[:, 1]
        return binary_metrics(labels, probabilities)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "trainer"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        use_cpu=False,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        weight_decay=0.01,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=True,
    )
    health = NumericalHealthCallback()
    trainer = GuardedTrainer(
        model=model, args=training_args, train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        processing_class=tokenizer, compute_metrics=compute_metrics,
        callbacks=[health],
    )
    training_started = time.perf_counter()
    train_result = trainer.train(
        resume_from_checkpoint=(str(args.resume_from_checkpoint)
                                if args.resume_from_checkpoint is not None else None)
    )
    training_seconds = time.perf_counter() - training_started

    validation_result = trainer.predict(tokenized["validation"], metric_key_prefix="validation")
    validation_p = checked_probabilities(torch.as_tensor(validation_result.predictions)).cpu().numpy()[:, 1]
    validation_metrics = evaluation_report(validation_result.label_ids, validation_p)
    export_predictions(source_frames["validation"], validation_p,
                       args.output_dir / "validation_predictions.csv", args.run_id, args.model)
    test_metrics = None
    inference_seconds = None
    if not args.skip_test:
        inference_started = time.perf_counter()
        test_result = trainer.predict(tokenized["test"], metric_key_prefix="test")
        inference_seconds = time.perf_counter() - inference_started
        test_p = checked_probabilities(torch.as_tensor(test_result.predictions)).cpu().numpy()[:, 1]
        test_metrics = evaluation_report(test_result.label_ids, test_p)
        export_predictions(source_frames["test"], test_p,
                           args.output_dir / "test_predictions.csv", args.run_id, args.model)

    model_dir = args.output_dir / "best_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    run_info = {
        "run_id": args.run_id, "model": args.model, "model_revision": model.config._commit_hash,
        "git_commit": git_commit, "dataset_version": "temporal-v1",
        "benchmark_id": freeze["benchmark_id"], "split_hashes": freeze["split_sha256"],
        "dataset_sha256": freeze["dataset_sha256"],
        "seed": args.seed,
        "training_config": {key: (str(value) if isinstance(value, Path) else value)
                             for key, value in vars(args).items()},
        "train_metrics": train_result.metrics, "validation_metrics": validation_metrics,
        "test_metrics": test_metrics, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "inference_rows_per_second": (len(source_frames["test"]) / inference_seconds
                                       if inference_seconds else None),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "precision": "bf16" if torch.cuda.is_bf16_supported() else "fp16",
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "numerical_health": health.diagnostics,
        "versions": {p: version(p) for p in ("torch", "transformers", "datasets", "accelerate")},
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "train": train_result.metrics, "validation": validation_metrics, "test": test_metrics,
    }, indent=2) + "\n")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import hashlib
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from deployguard.evaluation.metrics import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    if args.require_cuda and (args.cpu or not torch.cuda.is_available()):
        raise RuntimeError("This run requires a CUDA GPU")
    use_cuda = torch.cuda.is_available() and not args.cpu
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()

    set_seed(42)
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Preprocessing leakage audit must pass before training")
    for split, info in manifest["splits"].items():
        digest = hashlib.sha256((args.data_dir / f"{split}.jsonl").read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"{split} changed after preprocessing audit")

    files = {
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
        "test": str(args.data_dir / "test.jsonl"),
    }
    ds = load_dataset("json", data_files=files)

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    def tokenize(batch):
        return tokenizer(
            batch["state"],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = ds.map(tokenize, batched=True, remove_columns=[c for c in ds['train'].column_names if c != 'label'])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        id2label={0: "success", 1: "failure"},
        label2id={"success": 0, "failure": 1},
        revision=args.revision,
        attn_implementation="sdpa",
        reference_compile=False,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        if not np.isfinite(logits).all():
            raise FloatingPointError("Evaluation produced non-finite logits; refusing to save this run")
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        return binary_metrics(labels, probs)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_args = TrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        use_cpu=args.cpu,
        optim="adamw_torch",
        seed=42,
        data_seed=42,
        save_total_limit=1,
        dataloader_pin_memory=not args.cpu,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1 if args.max_steps > 0 else 50,
        load_best_model_at_end=True,
        metric_for_best_model="pr_auc",
        greater_is_better=True,
        report_to="none",
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    initial_weight = model.classifier.weight.detach().clone()
    started = time.perf_counter()
    train_result = trainer.train()
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    run_info = {"model": args.model, "resolved_revision": model.config._commit_hash,
                "steps": trainer.state.global_step, "train_metrics": train_result.metrics,
                "elapsed_seconds": time.perf_counter() - started,
                "classifier_max_weight_change": float((model.classifier.weight.detach().cpu() - initial_weight.cpu()).abs().max()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if use_cuda else None,
                "precision": "bf16" if use_bf16 else ("fp16" if use_cuda else "fp32"),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "versions": {p: version(p) for p in ("torch", "transformers", "datasets", "accelerate")},
                "experiment_scale": "smoke" if args.max_steps < 1000 else (
                    "full_cohort" if manifest["eligible_rows"] > 10000 else "extended_subset"
                ), "data_manifest": manifest}
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from deployguard.evaluation.health import checked_probabilities, check_finite_loss
from deployguard.evaluation.report import evaluation_report


MODEL_ID = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
UPSTREAM_COMMIT = "9d955671415fc19f069b9cc998928075c1f255ec"
BENCHMARK_ID = "temporal-v1-3e9cc5c049399d0b"
EXPECTED_TEST_HASH = "1410c9dcdc2c87ab503178f3f9749a674bd34959b402ca04ba16d3a0e22af003"
MAX_LENGTH = 512
HEAD_LR = 1e-4
MICRO_BATCH = 8
GRAD_ACCUM = 8
GROUP_SIZE = 4
MAX_GRADIENT_ABS = 1e6
MAX_PARAMETER_ABS = 1e6
QUESTION = {
    "t": "choice",
    "ins": "Given only the pre-run CI/CD state, what will the workflow outcome be?",
    "crit": {
        "success": "The workflow completes successfully.",
        "failure": "The workflow fails.",
    },
}


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for index, row in enumerate(rows):
        if not {"id", "repo", "timestamp", "state", "questions", "gold"}.issubset(row):
            raise ValueError(f"typed row {index} is missing required fields")
        question = row["questions"].get("outcome")
        if question != {
            "type": "choice",
            "instructions": QUESTION["ins"],
            "criteria": {"success": QUESTION["crit"]["success"],
                         "failure": QUESTION["crit"]["failure"]},
        }:
            raise ValueError(f"typed row {index} does not match the frozen binary choice schema")
        target = row["gold"]["outcome"]["label"]
        if target not in ("success", "failure"):
            raise ValueError(f"typed row {index} has invalid target {target!r}")
    return rows


def _tokenize_rows(rows: list[dict], tokenizer, build_sequence, encode_text, head_max_len: int) -> list[dict]:
    items = []
    for row in rows:
        state_ids = encode_text(tokenizer, row["state"], add_special_tokens=False,
                                truncation=True, max_length=MAX_LENGTH)["input_ids"]
        ids, markers = build_sequence(
            tokenizer, row["state"], QUESTION, max_len=MAX_LENGTH,
            head_max_len=head_max_len, state_ids=state_ids,
        )
        if len(markers) != 2:
            raise ValueError(f"Laya sequence lost choice markers for {row['id']}")
        items.append({
            "ids": ids, "markers": markers,
            "label": int(row["gold"]["outcome"]["label"] == "failure"),
            "target": [float(row["gold"]["outcome"]["label"] == "success"),
                       float(row["gold"]["outcome"]["label"] == "failure")],
            "id": str(row["id"]), "repo": str(row["repo"]),
            "timestamp": str(row["timestamp"]),
        })
    return items


def _collate(items: list[dict], pad_id: int, device: torch.device) -> dict[str, torch.Tensor]:
    n, length = len(items), max(len(item["ids"]) for item in items)
    input_ids = torch.full((n, length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((n, length), dtype=torch.long)
    marker_pos = torch.zeros((n, 2), dtype=torch.long)
    marker_mask = torch.ones((n, 2), dtype=torch.bool)
    target = torch.zeros((n, 2), dtype=torch.float32)
    labels = torch.empty(n, dtype=torch.long)
    for index, item in enumerate(items):
        size = len(item["ids"])
        input_ids[index, :size] = torch.as_tensor(item["ids"], dtype=torch.long)
        attention_mask[index, :size] = 1
        marker_pos[index] = torch.as_tensor(item["markers"], dtype=torch.long)
        target[index] = torch.as_tensor(item["target"], dtype=torch.float32)
        labels[index] = item["label"]
    return {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device),
            "marker_pos": marker_pos.to(device), "marker_mask": marker_mask.to(device),
            "target": target.to(device), "labels": labels.to(device),
            "qtype": torch.zeros(n, dtype=torch.long, device=device)}


def _checked_gradients(model, diagnostics: dict) -> None:
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if any(not torch.isfinite(g).all().item() for g in grads):
        diagnostics["finite_gradients"] = False
        raise FloatingPointError("Non-finite Laya gradient")
    maximum = max((float(g.abs().max().item()) for g in grads), default=0.0)
    diagnostics["gradient_checks"] += 1
    diagnostics["max_gradient_abs"] = max(diagnostics["max_gradient_abs"], maximum)
    if maximum > MAX_GRADIENT_ABS:
        diagnostics["finite_gradients"] = False
        raise FloatingPointError(f"Gradient magnitude {maximum:g} exceeds limit")


def _checked_parameters(model, diagnostics: dict) -> None:
    params = [p.detach() for p in model.parameters()]
    if any(not torch.isfinite(p).all().item() for p in params):
        diagnostics["finite_parameters"] = False
        raise FloatingPointError("Non-finite Laya parameter")
    maximum = max((float(p.abs().max().item()) for p in params), default=0.0)
    diagnostics["parameter_checks"] += 1
    diagnostics["max_parameter_abs"] = max(diagnostics["max_parameter_abs"], maximum)
    if maximum > MAX_PARAMETER_ABS:
        diagnostics["finite_parameters"] = False
        raise FloatingPointError(f"Parameter magnitude {maximum:g} exceeds limit")


@torch.no_grad()
def _predict(model, items: list[dict], tokenizer, device, batch_size: int, qtype: int,
             diagnostics: dict | None = None) -> tuple[np.ndarray, float]:
    model.eval()
    outputs = []
    started = time.perf_counter()
    for offset in range(0, len(items), batch_size):
        batch_items = items[offset:offset + batch_size]
        batch = _collate(batch_items, tokenizer.pad_token_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                              batch["marker_mask"], torch.full_like(batch["labels"], qtype))
        logits = logits[:, :2].float()
        probabilities = checked_probabilities(logits)
        if diagnostics is not None:
            diagnostics["finite_validation_logits"] = True
            diagnostics["finite_validation_probabilities"] = True
        outputs.extend(logits.cpu().numpy())
    elapsed = time.perf_counter() - started
    model.train()
    return np.asarray(outputs, dtype=np.float64), elapsed


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    z = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    y = torch.as_tensor(labels, dtype=torch.long, device="cpu")
    log_temp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temp.exp().clamp(0.1, 10.0)
        loss = F.cross_entropy(z / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    value = float(log_temp.detach().exp().clamp(0.1, 10.0).item())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite Laya validation temperature")
    return value


def _write_predictions(rows: list[dict], probs: np.ndarray, path: Path, run_id: str) -> None:
    if len(rows) != len(probs) or not np.isfinite(probs).all():
        raise FloatingPointError("Invalid probabilities in Laya prediction export")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "repo", "timestamp", "label", "p_failure",
                                               "predicted_label", "model_id", "run_id"])
        writer.writeheader()
        for row, probability in zip(rows, probs):
            if not 0.0 <= float(probability) <= 1.0:
                raise FloatingPointError("Laya probability outside [0, 1]")
            writer.writerow({"id": row["id"], "repo": row["repo"], "timestamp": row["timestamp"],
                             "label": int(row["gold"]["outcome"]["label"] == "failure"),
                             "p_failure": float(probability), "predicted_label": int(probability >= 0.5),
                             "model_id": MODEL_ID, "run_id": run_id})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--encoder-learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=MICRO_BATCH)
    parser.add_argument("--include-test", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Laya experiment requires CUDA; refusing CPU fallback")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Laya L4 experiment requires bfloat16 support")
    if args.data_dir.joinpath("test.jsonl").exists() != args.include_test:
        raise ValueError("Test rows must be absent during tuning and present only for final runs")

    metadata = json.loads((args.data_dir / "benchmark.json").read_text())
    if metadata.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Expected frozen temporal-v1 typed-decision bridge")
    if metadata.get("split_hashes", {}).get("test") != EXPECTED_TEST_HASH:
        raise ValueError("Typed bridge test hash differs from benchmark lock")
    splits = {name: _load_rows(args.data_dir / f"{name}.jsonl")
              for name in (("train", "validation", "test") if args.include_test else ("train", "validation"))}

    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import QTYPES, build_model, build_sequence, encode_text, proper_reward

    model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    _fix_tokenizer_config(model_dir)
    config = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(Path(model_dir) / "tokenizer")
    tokenized = {name: _tokenize_rows(rows, tokenizer, build_sequence, encode_text,
                                     int(config.get("head_max_len", 256)))
                 for name, rows in splits.items()}

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda")
    config["gradient_checkpointing"] = True
    model = build_model(config, encoder_dir=str(Path(model_dir) / "encoder"))
    model.load_state_dict(load_file(str(Path(model_dir) / "model.safetensors")), strict=True)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    encoder_params = [p for name, p in model.named_parameters() if "encoder." in name]
    head_params = [p for name, p in model.named_parameters() if "encoder." not in name]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": args.encoder_learning_rate},
        {"params": head_params, "lr": HEAD_LR},
    ], weight_decay=0.01)
    n_micro = math.ceil(len(tokenized["train"]) / args.batch_size)
    total_updates = max(1, math.ceil(n_micro / GRAD_ACCUM) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    health = {
        "finite_loss": True, "finite_gradients": True, "finite_parameters": True,
        "finite_training_logits": True, "finite_validation_logits": True,
        "finite_validation_probabilities": True, "finite_test_logits": None,
        "finite_test_probabilities": None, "max_gradient_abs": 0.0, "max_parameter_abs": 0.0,
        "gradient_checks": 0, "parameter_checks": 0,
        "gradient_magnitude_limit": MAX_GRADIENT_ABS,
        "parameter_magnitude_limit": MAX_PARAMETER_ABS,
    }
    rng = torch.Generator().manual_seed(args.seed)
    training_started = time.perf_counter()
    losses = []
    for epoch in range(args.epochs):
        order = torch.randperm(len(tokenized["train"]), generator=rng).tolist()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum = 0
        for start in range(0, len(order), args.batch_size):
            selected = [tokenized["train"][index] for index in order[start:start + args.batch_size]]
            batch = _collate(selected, tokenizer.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, act = model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                                    batch["marker_mask"], batch["qtype"])
            logits = logits[:, :2].float()
            if not torch.isfinite(logits).all().item():
                health["finite_training_logits"] = False
                raise FloatingPointError("Non-finite Laya training logits")
            mask = batch["marker_mask"]
            target = batch["target"]
            count = mask.sum(-1, keepdim=True).float()
            sigma = 0.4 + (0.1 - 0.4) * (epoch / max(1, args.epochs - 1))
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / count) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask.unsqueeze(0), -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(q, target.unsqueeze(0), batch["qtype"], mask,
                                       w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask.unsqueeze(0)).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce + 0.0 * act.sum()) / GRAD_ACCUM
            check_finite_loss(loss)
            health["finite_loss"] = True
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().item() * GRAD_ACCUM))
            accum += 1
            last_batch = start + args.batch_size >= len(order)
            if accum == GRAD_ACCUM or last_batch:
                scaler.unscale_(optimizer)
                _checked_gradients(model, health)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                _checked_parameters(model, health)
                accum = 0
        print(f"epoch={epoch + 1}/{args.epochs} mean_loss={np.mean(losses[-n_micro:]):.5f}", flush=True)
    training_seconds = time.perf_counter() - training_started

    val_logits, val_seconds = _predict(model, tokenized["validation"], tokenizer, device,
                                       args.batch_size * 2, QTYPES["choice"], health)
    val_labels = np.asarray([item["label"] for item in tokenized["validation"]], dtype=int)
    temperature = _fit_temperature(val_logits, val_labels) if args.include_test else 1.0
    val_probabilities = checked_probabilities(torch.as_tensor(val_logits / temperature)).cpu().numpy()[:, 1]
    val_metrics = evaluation_report(val_labels, val_probabilities)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(splits["validation"], val_probabilities,
                       args.output_dir / "validation_predictions.csv", args.run_id)

    test_metrics = None
    inference_seconds = None
    if args.include_test:
        test_logits, inference_seconds = _predict(model, tokenized["test"], tokenizer, device,
                                                  args.batch_size * 2, QTYPES["choice"])
        health["finite_test_logits"] = bool(np.isfinite(test_logits).all())
        test_probabilities = checked_probabilities(torch.as_tensor(test_logits / temperature)).cpu().numpy()[:, 1]
        health["finite_test_probabilities"] = bool(np.isfinite(test_probabilities).all())
        test_labels = np.asarray([item["label"] for item in tokenized["test"]], dtype=int)
        test_metrics = evaluation_report(test_labels, test_probabilities)
        _write_predictions(splits["test"], test_probabilities,
                           args.output_dir / "test_predictions.csv", args.run_id)
        save_file({name: tensor.detach().half().contiguous().cpu()
                   for name, tensor in model.state_dict().items()},
                  str(args.output_dir / "model.safetensors"))

    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                                    capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    run_info = {
        "run_id": args.run_id, "model_family": "Laya", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "upstream_repository": "https://github.com/NandhaKishorM/laya",
        "upstream_commit": UPSTREAM_COMMIT, "git_commit": git_commit,
        "dataset_version": "temporal-v1", "benchmark_id": BENCHMARK_ID,
        "split_hashes": metadata["split_hashes"],
        "typed_bridge_sha256": {name: hashlib.sha256((args.data_dir / f"{name}.jsonl").read_bytes()).hexdigest()
                                for name in splits},
        "seed": args.seed,
        "training_config": {
            "encoder_learning_rate": args.encoder_learning_rate, "head_learning_rate": HEAD_LR,
            "epochs": args.epochs, "micro_batch": args.batch_size,
            "gradient_accumulation": GRAD_ACCUM, "effective_batch": args.batch_size * GRAD_ACCUM,
            "max_length": MAX_LENGTH, "head_max_len": int(config.get("head_max_len", 256)),
            "objective": "upstream RLCD proper-scoring-rule reward (spherical=0.75, RPS=1.0) + soft cross-entropy",
            "group_samples": GROUP_SIZE, "sigma_start": 0.4, "sigma_end": 0.1,
            "weight_decay": 0.01, "gradient_clip_norm": 1.0,
            "calibration": "scalar temperature fit by NLL on locked validation only" if args.include_test else None,
            "test_loaded": args.include_test,
        },
        "train_metrics": {"mean_loss": float(np.mean(losses))},
        "validation_metrics": val_metrics, "test_metrics": test_metrics,
        "calibration_temperature": temperature if args.include_test else None,
        "training_seconds": training_seconds, "validation_inference_seconds": val_seconds,
        "inference_seconds": inference_seconds,
        "inference_rows_per_second": (len(splits["test"]) / inference_seconds if inference_seconds else None),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "precision": "bf16", "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "numerical_health": health,
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "train": run_info["train_metrics"], "validation": val_metrics, "test": test_metrics,
    }, indent=2) + "\n")
    print(json.dumps(run_info, indent=2))


if __name__ == "__main__":
    main()

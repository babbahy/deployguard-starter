from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from deployguard.evaluation.metrics import binary_metrics


def read_test_split(data_dir: Path) -> tuple[list[dict], dict]:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Source test split did not pass its recorded audit")
    path = data_dir / "test.jsonl"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["splits"]["test"]["sha256"]:
        raise ValueError("Test split checksum differs from manifest")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows, manifest


def state_fields(state: str) -> dict[str, str]:
    return dict(
        line.split(": ", 1) for line in state.splitlines() if ": " in line
    )


def repository_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    results = []
    for repo, frame in predictions.groupby("repo", sort=True):
        labels = frame.label.to_numpy(dtype=int)
        probabilities = frame.p_failure.to_numpy(dtype=float)
        result = {
            "repo": repo,
            "runs": len(frame),
            "failures": int(labels.sum()),
            "actual_failure_rate": float(labels.mean()),
            "average_predicted_failure_probability": float(probabilities.mean()),
            "brier": float(np.mean(np.square(probabilities - labels))),
        }
        result.update(binary_metrics(labels, probabilities))
        results.append(result)
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export predictions from an existing local ModernBERT checkpoint; never trains."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed/cicd_full")
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path("runs/modernbert-l4-20260927T094219Z-56e25afb/best_model"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("analysis/modernbert_test_predictions.csv")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    status_path = args.output.with_name("modernbert_prediction_export_status.json")
    status = {
        "checkpoint": str(args.model_dir),
        "test_data": str(args.data_dir / "test.jsonl"),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "success": False,
    }

    try:
        torch.set_num_threads(args.threads)
        rows, manifest = read_test_split(args.data_dir)
        status["test_rows"] = len(rows)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir,
            local_files_only=True,
            attn_implementation="sdpa",
            reference_compile=False,
        ).eval()
        status["checkpoint_max_abs_parameter"] = max(
            float(parameter.detach().abs().max().cpu())
            for parameter in model.parameters()
        )
        device = torch.device(status["device"])
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
        records = []
        probabilities = []
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            fields = [state_fields(row["state"]) for row in batch]
            tokens = tokenizer(
                [row["state"] for row in batch], return_tensors="pt",
                padding=True, truncation=True, max_length=128,
            ).to(device)
            with torch.inference_mode():
                if device.type == "cuda" and torch.cuda.is_bf16_supported():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(**tokens).logits
                else:
                    logits = model(**tokens).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError(
                    f"Non-finite logits at test rows {start}:{start + len(batch)} on {device}"
                )
            batch_probabilities = logits.float().softmax(-1)[:, 1].cpu().numpy()
            if not np.isfinite(batch_probabilities).all():
                raise FloatingPointError("Non-finite failure probabilities")
            probabilities.extend(batch_probabilities.tolist())
            for row, feature, probability in zip(batch, fields, batch_probabilities):
                records.append({
                    "repo": row.get("group", row.get("repo")),
                    "id": row["id"],
                    "commit_sha": row.get("commit_sha"),
                    "label": int(row["label"]),
                    "p_failure": float(probability),
                    "event": feature.get("event"),
                    "branch": feature.get("head_branch"),
                    "total_churn": feature.get("total_churn"),
                    "files_modified": feature.get("files_modified"),
                    "msg_len": feature.get("msg_len"),
                    "is_merge": feature.get("is_merge"),
                    "created_at": row.get("created_at"),
                })

        predictions = pd.DataFrame(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        repo_metrics = repository_metrics(predictions)
        repo_metrics.to_csv(
            args.output.with_name("modernbert_test_metrics_by_repo.csv"), index=False
        )
        status.update({
            "success": True,
            "test_sha256": manifest["splits"]["test"]["sha256"],
            "predictions_csv": str(args.output),
            "repository_metrics_csv": str(
                args.output.with_name("modernbert_test_metrics_by_repo.csv")
            ),
            "finite_probabilities": bool(np.isfinite(probabilities).all()),
        })
    except Exception as error:
        status["error"] = f"{type(error).__name__}: {error}"
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2), file=sys.stderr)
        raise
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

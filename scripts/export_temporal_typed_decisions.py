from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from deployguard.data.temporal import rich_temporal_v1_state


QUESTION = {
    "outcome": {
        "type": "choice",
        "instructions": "Given only the pre-run CI/CD state, what will the workflow outcome be?",
        "criteria": {
            "success": "The workflow completes successfully.",
            "failure": "The workflow fails.",
        },
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Export frozen temporal-v1 examples as typed decisions")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/typed/temporal_v1"))
    args = parser.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    freeze = manifest.get("benchmark_freeze")
    if not freeze or freeze.get("version") != "temporal-v1":
        raise ValueError("Only frozen temporal-v1 may be exported")
    if json.loads((args.data_dir / "benchmark_lock.json").read_text()) != freeze:
        raise ValueError("Benchmark lock mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        source = args.data_dir / f"{split}.jsonl"
        if hashlib.sha256(source.read_bytes()).hexdigest() != freeze["split_sha256"][split]:
            raise ValueError(f"Frozen {split} split changed")
        target = args.output_dir / f"{split}.jsonl"
        with source.open(encoding="utf-8") as reader, target.open("w", encoding="utf-8") as writer:
            for line in reader:
                row = json.loads(line)
                outcome = "failure" if int(row["label"]) == 1 else "success"
                record = {
                    "id": row["id"], "repo": row["repo"],
                    "timestamp": row["created_at"], "workflow": "ci_outcome",
                    "state": rich_temporal_v1_state(row), "questions": QUESTION,
                    "gold": {"outcome": {"label": outcome,
                                             "probabilities": {"success": float(outcome == "success"),
                                                                "failure": float(outcome == "failure")}}},
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "benchmark.json").write_text(json.dumps({
        "benchmark_id": freeze["benchmark_id"],
        "split_hashes": freeze["split_sha256"],
        "note": "One typed-decision record per frozen temporal-v1 row; no split or state changes.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()

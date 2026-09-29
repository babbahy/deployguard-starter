from __future__ import annotations

import argparse
import json
from pathlib import Path


QUESTION = {
    "deployment_outcome": {
        "type": "choice",
        "instructions": (
            "Given only the pre-run CI/CD context, what is the most likely final pipeline outcome?"
        ),
        "criteria": {
            "success": "The pipeline completes successfully.",
            "failure": "The pipeline fails, times out, or requires intervention.",
        },
    }
}


def convert(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            label = row["label_name"]
            probs = {
                "success": 1.0 if label == "success" else 0.0,
                "failure": 1.0 if label == "failure" else 0.0,
            }
            out = {
                "id": row["id"],
                "workflow": "deployment_risk",
                "state": json.dumps(
                    {"pre_run_context": row["state"]},
                    ensure_ascii=False,
                ),
                "questions": json.dumps(QUESTION, ensure_ascii=False),
                "gold": json.dumps(
                    {
                        "deployment_outcome": {
                            "label": label,
                            "probabilities": probs,
                        }
                    },
                    ensure_ascii=False,
                ),
                "group": row.get("group"),
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation", "test"):
        convert(
            args.data_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
        )
        print(f"Wrote {args.output_dir / f'{split}.jsonl'}")


if __name__ == "__main__":
    main()

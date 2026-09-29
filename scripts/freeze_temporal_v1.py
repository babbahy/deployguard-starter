from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and verify temporal benchmark v1")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_temporal"))
    parser.add_argument("--source-raw", type=Path, default=Path("data/raw/final_research_dataset_MASTER.csv"))
    parser.add_argument("--source-manifest", type=Path, default=Path("data/processed/cicd_full/manifest.json"))
    args = parser.parse_args()

    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Cannot freeze a benchmark whose leakage audit failed")
    if set(manifest.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Expected train, validation and test split entries")

    split_hashes = {}
    split_summary = {}
    repositories = set()
    columns: list[str] | None = None
    for split in ("train", "validation", "test"):
        path = args.data_dir / f"{split}.jsonl"
        digest = sha256(path)
        declared = manifest["splits"][split]
        if digest != declared.get("sha256"):
            raise ValueError(f"{split} bytes differ from the current temporal manifest")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows or len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"{split} is empty or contains duplicate run IDs")
        if columns is None:
            columns = sorted(rows[0])
        if any(sorted(row) != columns for row in rows):
            raise ValueError(f"Inconsistent row schema in {split}")
        repo_set = sorted({row["repo"] for row in rows})
        repositories.update(repo_set)
        actual_failures = sum(int(row["label"]) for row in rows)
        if (len(rows), actual_failures, repo_set) != (
            declared["rows"], declared["failures"], declared["repositories"]
        ):
            raise ValueError(f"{split} summary differs from manifest")
        split_hashes[split] = digest
        split_summary[split] = {
            "rows": len(rows), "failures": actual_failures,
            "failure_rate": actual_failures / len(rows), "repositories": repo_set,
        }

    freeze = {
        "version": "temporal-v1",
        "immutable": True,
        "source_raw_sha256": sha256(args.source_raw),
        "source_manifest_sha256": sha256(args.source_manifest),
        "split_sha256": split_hashes,
        "repository_membership": sorted(repositories),
        "split_summary": split_summary,
        "feature_schema": {
            "serialized_columns": columns,
            "model_features": manifest["selected_features"],
            "prediction_time": manifest["prediction_time"],
            "outcome_column": "label (not part of state)",
            "identity_only_columns": ["id", "repo", "commit_sha", "created_at"],
        },
        "split_strategy": manifest["split_strategy"],
        "eligibility": manifest["eligibility"],
        "commit_purge": manifest["commit_purge"],
        "audit": manifest["audit"],
    }
    lock_payload = json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()
    freeze["benchmark_id"] = "temporal-v1-" + hashlib.sha256(lock_payload).hexdigest()[:16]

    old_freeze = manifest.get("benchmark_freeze")
    if old_freeze is not None and old_freeze != freeze:
        raise ValueError("A different benchmark freeze already exists; refusing to change v1")
    manifest["benchmark_freeze"] = freeze
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.data_dir / "benchmark_lock.json").write_text(json.dumps(freeze, indent=2) + "\n")
    print(json.dumps({"benchmark_id": freeze["benchmark_id"], "split_summary": split_summary}, indent=2))


if __name__ == "__main__":
    main()

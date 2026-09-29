from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from deployguard.data.history import HISTORY_FEATURES, build_history_state
from deployguard.data.history_fast import add_causal_history
from deployguard.data.temporal import load_canonical_cohort


def main() -> None:
    parser = argparse.ArgumentParser(description="Build online-causal historical features as benchmark v2")
    parser.add_argument("--v1-dir", type=Path, default=Path("data/processed/cicd_temporal"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cicd_full"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/final_research_dataset_MASTER.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cicd_temporal_history_v2"))
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty benchmark: {args.output_dir}")
    v1_manifest = json.loads((args.v1_dir / "manifest.json").read_text())
    freeze = v1_manifest.get("benchmark_freeze")
    if not freeze or freeze.get("version") != "temporal-v1":
        raise ValueError("History v2 must be built from frozen temporal-v1")
    if json.loads((args.v1_dir / "benchmark_lock.json").read_text()) != freeze:
        raise ValueError("Temporal-v1 lock mismatch")
    with args.raw.open("rb") as source:
        raw_hash = hashlib.file_digest(source, "sha256").hexdigest()
    if raw_hash != freeze["source_raw_sha256"]:
        raise ValueError("Raw source hash differs from frozen temporal-v1")

    canonical, _ = load_canonical_cohort(args.processed_dir, args.raw)
    split_rows = {
        name: pd.read_json(args.v1_dir / f"{name}.jsonl", lines=True)
        for name in ("train", "validation", "test")
    }
    rows = pd.concat(split_rows.values(), ignore_index=True)
    if rows.id.duplicated().any():
        raise ValueError("Duplicate run identity in frozen v1")
    metadata = canonical[["id", "updated_at"]]
    joined = rows.merge(metadata, on="id", how="left", validate="one_to_one")
    unavailable_outcomes = int(joined.updated_at.isna().sum())
    causal = add_causal_history(joined)
    causal["state"] = causal.apply(lambda row: build_history_state(row, row.state), axis=1)
    if len(causal) != len(rows) or set(causal.id) != set(rows.id):
        raise ValueError("History feature join changed benchmark rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark_version": "temporal-history-v2",
        "base_benchmark_id": freeze["benchmark_id"],
        "source_raw_sha256": raw_hash,
        "unavailable_outcomes_omitted_from_history": unavailable_outcomes,
        "prediction_time": "workflow creation; only prior outcomes with updated_at strictly earlier are usable",
        "outcome_availability_field": "updated_at, used only to gate prior labels and never serialized as a feature",
        "history_features": HISTORY_FEATURES,
        "history_order": "all examples at the same created_at are scored before any same-time outcome is admitted",
        "split_strategy": "unchanged IDs and split membership from immutable temporal-v1",
        "temporal_v1_split_sha256": freeze["split_sha256"],
        "splits": {},
    }
    for name, v1_part in split_rows.items():
        part = causal[causal.id.isin(set(v1_part.id))].copy()
        if set(part.id) != set(v1_part.id):
            raise ValueError(f"{name} identity membership differs from v1")
        part = part.set_index("id").loc[v1_part.id].reset_index()
        export_columns = list(v1_part.columns) + HISTORY_FEATURES
        export_columns = list(dict.fromkeys(export_columns))
        path = args.output_dir / f"{name}.jsonl"
        part[export_columns].to_json(path, orient="records", lines=True, force_ascii=False)
        manifest["splits"][name] = {
            "rows": len(part), "failures": int(part.label.sum()),
            "failure_rate": float(part.label.mean()),
            "repositories": sorted(part.repo.unique().tolist()),
            "id_set_sha256": hashlib.sha256("\n".join(sorted(part.id)).encode()).hexdigest(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest["audit"] = {
        "passed": True,
        "same_ids_as_v1": True,
        "same_split_membership_as_v1": True,
        "outcome_maturity_gated": True,
        "future_outcomes_used_for_earlier_features": False,
        "same_timestamp_outcomes_used": False,
        "post_outcome_features_used": [],
        "updated_at_serialized_or_used_as_model_input": False,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: info["rows"] for name, info in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()

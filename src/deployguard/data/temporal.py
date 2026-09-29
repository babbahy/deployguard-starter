from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from deployguard.data.prepare_cicd import serialize_state


CATEGORICAL_FEATURES = ["repo", "event", "head_branch", "workflow_id", "is_merge"]
NUMERIC_FEATURES = [
    "run_number", "additions", "deletions", "total_churn", "files_modified",
    "msg_len", "num_parents", "commit_age_minutes", "commit_hour",
    "commit_weekday", "created_hour", "created_weekday",
]
TEXT_FEATURE = "commit_message"
SPLIT_ORDER = {"train": 0, "validation": 1, "test": 2}


def load_canonical_cohort(processed_dir: Path, raw_path: Path) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Repository-disjoint source splits must pass the leakage audit")
    canonical_parts = []
    for split, info in manifest["splits"].items():
        path = processed_dir / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"{split} split checksum differs from its manifest")
        part = pd.read_json(path, lines=True)
        canonical_parts.append(part[["id", "group", "commit_sha", "created_at", "label", "state"]])
    canonical = pd.concat(canonical_parts, ignore_index=True)
    if canonical.id.duplicated().any():
        raise ValueError("Canonical run identity appears more than once")
    canonical["created_at"] = pd.to_datetime(canonical.created_at, utc=True, errors="coerce")
    if canonical.created_at.isna().any():
        raise ValueError("Canonical source includes invalid creation timestamps")

    raw = pd.read_csv(raw_path, low_memory=False)
    raw["repo"] = raw.repo.astype("string").str.strip().str.lower()
    raw["id"] = "gha:" + raw.repo.astype(str) + ":" + raw.run_id.astype(str)
    raw["label"] = raw.conclusion.map({"success": 0, "failure": 1})
    raw["created_at"] = pd.to_datetime(raw.created_at, utc=True, errors="coerce")
    raw["committed_date"] = pd.to_datetime(raw.committed_date, utc=True, errors="coerce")
    raw = raw[raw.id.isin(set(canonical.id))]
    joined = raw.merge(
        canonical,
        on=["id", "commit_sha", "created_at", "label"],
        how="inner",
        validate="many_to_one",
        suffixes=("_raw", "_canonical"),
    )
    joined["checked_state"] = joined.apply(serialize_state, axis=1)
    joined = joined[joined.checked_state.eq(joined.state)].copy()
    if joined.id.duplicated().any() or set(joined.id) != set(canonical.id):
        raise ValueError("Raw-to-canonical rich-feature join is ambiguous or incomplete")
    if not joined.group.eq(joined.repo).all():
        raise ValueError("Repository identity changed during the rich-feature join")
    if joined.committed_date.isna().any() or joined.committed_date.gt(joined.created_at).any():
        raise ValueError("Commit timestamp is missing or later than workflow creation")
    return joined, manifest


def _text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return " ".join(str(value).replace("\r", " ").replace("\n", " ").split())


def _category(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (float, np.floating)) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def add_rich_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["commit_message"] = rows.message.map(_text)
    rows["workflow_id"] = rows.workflow_id.map(_category)
    rows["head_branch"] = rows.head_branch.map(_text)
    for column in ("run_number", "additions", "deletions", "total_churn",
                   "files_modified", "msg_len", "num_parents"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    rows["created_hour"] = rows.created_at.dt.hour.astype(int)
    rows["created_weekday"] = rows.created_at.dt.dayofweek.astype(int)
    rows["commit_age_minutes"] = (
        (rows.created_at - rows.committed_date).dt.total_seconds() / 60.0
    )
    rows["commit_hour"] = rows.committed_date.dt.hour.astype(int)
    rows["commit_weekday"] = rows.committed_date.dt.dayofweek.astype(int)
    rows["is_merge"] = rows.is_merge.astype("boolean").fillna(False).astype(bool)
    return rows


def _rich_state(row: pd.Series) -> str:
    fields = [
        ("repository", row.repo),
        ("event", row.event),
        ("branch", row.head_branch),
        ("workflow_id", row.workflow_id),
        ("run_number", row.run_number),
        ("commit_message", row.commit_message),
        ("additions", row.additions),
        ("deletions", row.deletions),
        ("num_parents", row.num_parents),
        ("total_churn", row.total_churn),
        ("files_modified", row.files_modified),
        ("msg_len", row.msg_len),
        ("commit_age_minutes", row.commit_age_minutes),
        ("commit_hour_utc", row.commit_hour),
        ("commit_weekday_utc", row.commit_weekday),
        ("is_merge", str(bool(row.is_merge)).lower()),
        ("created_hour_utc", row.created_hour),
        ("created_weekday_utc", row.created_weekday),
    ]
    lines = []
    for name, value in fields:
        if pd.isna(value):
            value = "unknown"
        elif isinstance(value, (float, np.floating)):
            value = format(float(value), ".12g")
        else:
            value = str(value)
        lines.append(f"{name}: {value}")
    return "\n".join(lines)


TEMPORAL_V1_STATE_FIELDS = (
    "repo", "event", "head_branch", "workflow_id", "run_number", "commit_message",
    "additions", "deletions", "num_parents", "total_churn", "files_modified",
    "msg_len", "is_merge", "created_hour", "created_weekday",
)


def rich_temporal_v1_state(row: dict | pd.Series) -> str:
    values = [
        ("repository", _text(row["repo"])),
        ("event", _text(row["event"])),
        ("branch", _text(row["head_branch"])),
        ("workflow_id", _category(row["workflow_id"])),
        ("run_number", row["run_number"]),
        ("commit_message", _text(row["commit_message"])),
        ("additions", row["additions"]),
        ("deletions", row["deletions"]),
        ("num_parents", row["num_parents"]),
        ("total_churn", row["total_churn"]),
        ("files_modified", row["files_modified"]),
        ("msg_len", row["msg_len"]),
        ("is_merge", row["is_merge"]),
        ("created_hour_utc", row["created_hour"]),
        ("created_weekday_utc", row["created_weekday"]),
    ]
    values.sort(key=lambda item: item[0] == "commit_message")
    lines = []
    for name, value in values:
        if pd.isna(value):
            value = "unknown"
        elif isinstance(value, (bool, np.bool_)):
            value = str(value).lower()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            value = format(float(value), ".12g")
        else:
            value = _text(value)
        lines.append(f"{name}: {value}")
    return "\n".join(lines)


def _commit_purge(rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    rows = rows.copy()
    rows["split_rank"] = rows.split.map(SPLIT_ORDER)
    first_split = rows.groupby("commit_sha").split_rank.transform("min")
    keep = rows.split_rank.eq(first_split)
    dropped = int((~keep).sum())
    return rows.loc[keep].drop(columns="split_rank").copy(), dropped


def _audit_temporal_splits(splits: dict[str, pd.DataFrame]) -> dict:
    overlaps: dict[str, dict[str, int]] = {}
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = splits[left], splits[right]
            overlaps[f"{left}/{right}"] = {
                "id": len(set(a.id) & set(b.id)),
                "commit_sha": len(set(a.commit_sha) & set(b.commit_sha)),
            }
    if any(any(value for value in pair.values()) for pair in overlaps.values()):
        raise ValueError(f"Identity or commit leakage across temporal splits: {overlaps}")

    chronology = {}
    for repo, group in pd.concat(splits.values()).groupby("repo"):
        dates = {}
        for split, frame in splits.items():
            part = frame[frame.repo.eq(repo)]
            if not part.empty:
                dates[split] = {
                    "first": part.created_at.min().isoformat(),
                    "last": part.created_at.max().isoformat(),
                }
        present = [name for name in names if name in dates]
        ordered = all(
            pd.Timestamp(dates[left]["last"]) <= pd.Timestamp(dates[right]["first"])
            for left, right in zip(present, present[1:])
        )
        chronology[repo] = {"passed": ordered, "ranges": dates}
        if not ordered:
            raise ValueError(f"Future-to-past chronology violation for {repo}: {dates}")
    return {
        "passed": True,
        "overlaps": overlaps,
        "chronology_by_repo": chronology,
        "note": "Repeated commit rows assigned to an earlier partition are purged from later partitions; input-state duplicates are descriptive, not run identities.",
    }


def prepare_temporal(
    processed_dir: Path,
    raw_path: Path,
    output_dir: Path,
    min_rows: int = 300,
    min_train_failures: int = 20,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    rows, source_manifest = load_canonical_cohort(processed_dir, raw_path)
    rows = add_rich_features(rows)

    assignments = []
    repo_report = []
    for repo, group in rows.groupby("repo", sort=True):
        group = group.sort_values(["created_at", "id"], kind="stable").reset_index(drop=True)
        n = len(group)
        train_end = math.floor(n * 0.70)
        validation_end = math.floor(n * 0.85)
        train_failures = int(group.iloc[:train_end].label.sum())
        eligible = n >= min_rows and train_failures >= min_train_failures
        repo_report.append({
            "repo": repo,
            "eligible": eligible,
            "eligibility_reason": "included" if eligible else (
                "too_few_rows" if n < min_rows else "too_few_training_failures"
            ),
            "all_rows": n,
            "all_failures": int(group.label.sum()),
            "initial_train_rows": train_end,
            "initial_train_failures": train_failures,
            "initial_validation_rows": validation_end - train_end,
            "initial_test_rows": n - validation_end,
        })
        if not eligible:
            continue
        group["split"] = "train"
        group.loc[train_end:validation_end - 1, "split"] = "validation"
        group.loc[validation_end:, "split"] = "test"
        assignments.append(group)

    if not assignments:
        raise ValueError("No repositories meet the predeclared temporal support minimum")
    selected = pd.concat(assignments, ignore_index=True)
    selected, purged = _commit_purge(selected)
    if set(selected.split.unique()) != set(SPLIT_ORDER):
        raise ValueError("Temporal dataset must retain train, validation, and test rows")
    splits = {
        name: selected[selected.split.eq(name)].sort_values(
            ["repo", "created_at", "id"], kind="stable"
        ).reset_index(drop=True)
        for name in SPLIT_ORDER
    }
    if any(part.label.nunique() < 2 for part in splits.values()):
        raise ValueError("Temporal split is single-class; inspect repository/support criteria")
    audit = _audit_temporal_splits(splits)

    if output_dir.exists() and not any(output_dir.iterdir()):
        pass
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = (
        CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TEXT_FEATURE]
    )
    manifest = {
        "source_manifest": str(processed_dir / "manifest.json"),
        "source_split_strategy": source_manifest["split_strategy"],
        "cohort_rows_before_support_filter": len(rows),
        "eligible_repositories": [r["repo"] for r in repo_report if r["eligible"]],
        "excluded_repositories": [r for r in repo_report if not r["eligible"]],
        "eligibility": {
            "min_total_rows": min_rows,
            "min_failures_in_first_70pct_training_window": min_train_failures,
            "label-based eligibility uses training-window labels only": True,
        },
        "split_strategy": "within-repository chronological 70/15/15 by created_at, stable id tiebreaker",
        "commit_purge": {
            "later_partition_rows_removed": purged,
            "rule": "If a commit appears in multiple temporal partitions, retain rows in the earliest partition and remove its later-partition copies.",
        },
        "prediction_time": "workflow creation",
        "selected_features": feature_columns,
        "excluded_post_outcome_fields": [
            "conclusion", "status", "duration", "updated_at", "run_started_at",
            "time_since_last_commit (excluded pending source provenance)",
        ],
        "audit": audit,
        "repositories": repo_report,
        "splits": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_features = list(dict.fromkeys([
        "id", "repo", "commit_sha", "created_at", "committed_date", "label", "state",
        *feature_columns,
    ]))
    for name, part in splits.items():
        export = part[output_features].copy()
        export["created_at"] = export.created_at.map(lambda x: x.isoformat())
        export["committed_date"] = export.committed_date.map(lambda x: x.isoformat())
        path = output_dir / f"{name}.jsonl"
        export.to_json(path, orient="records", lines=True, force_ascii=False)
        manifest["splits"][name] = {
            "rows": len(export),
            "failures": int(export.label.sum()),
            "failure_rate": float(export.label.mean()),
            "repositories": sorted(export.repo.unique().tolist()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create a per-repository chronological CI benchmark")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/cicd_full"))
    parser.add_argument("--raw", type=Path, default=Path("data/raw/final_research_dataset_MASTER.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cicd_temporal_v2"))
    parser.add_argument("--min-rows", type=int, default=300)
    parser.add_argument("--min-train-failures", type=int, default=20)
    args = parser.parse_args()
    manifest = prepare_temporal(
        args.processed_dir, args.raw, args.output_dir, args.min_rows, args.min_train_failures
    )
    print(json.dumps({
        "eligible_repositories": len(manifest["eligible_repositories"]),
        "split_rows": {name: info["rows"] for name, info in manifest["splits"].items()},
        "commit_rows_purged": manifest["commit_purge"]["later_partition_rows_removed"],
        "audit": manifest["audit"]["passed"],
        "output_dir": str(args.output_dir),
    }, indent=2))


if __name__ == "__main__":
    main()

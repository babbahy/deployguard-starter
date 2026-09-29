from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SOURCE = "https://data.mendeley.com/datasets/mggwn7rj9f/1"
SOURCE_SHA256 = "b6a89418e39e6144f4ac71496256474c7bc27651e099557fe074d7798da1e73e"
# Prediction time is creation of a first-attempt push/pull_request workflow run.
FEATURES = ("event", "head_branch", "total_churn", "files_modified", "msg_len", "is_merge")
REQUIRED = set(FEATURES) | {
    "repo", "run_id", "commit_sha", "node_id", "run_attempt", "conclusion",
    "created_at", "committed_date", "head_sha",
}


def serialize_state(row: pd.Series) -> str:
    lines = []
    for name in FEATURES:
        value = row[name]
        if pd.isna(value):
            value = "unknown"
        elif name in {"total_churn", "files_modified", "msg_len"}:
            value = format(float(value), ".12g")
        else:
            value = str(value).strip().replace("\n", " ").replace("\r", " ")
        lines.append(f"{name}: {value}")
    return "\n".join(lines)


def grouped_split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.repo.isna().any() or df.repo.nunique() < 8:
        raise ValueError("Need at least 8 known repositories; random-row fallback is forbidden.")
    first = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=42)
    train, held = next(first.split(df, groups=df.repo))
    rest = df.iloc[held]
    second = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=43)
    val, test = next(second.split(rest, groups=rest.repo))
    return {"train": df.iloc[train].copy(), "validation": rest.iloc[val].copy(),
            "test": rest.iloc[test].copy()}


def audit_splits(splits: dict[str, pd.DataFrame]) -> dict:
    overlaps = {}
    for a, b in combinations(splits, 2):
        pair = {col: len(set(splits[a][col]) & set(splits[b][col]))
                for col in ("repo", "id", "commit_sha", "state")}
        if any(pair[col] for col in ("repo", "id", "commit_sha")):
            raise ValueError(f"Split leakage between {a} and {b}: {pair}")
        overlaps[f"{a}/{b}"] = pair
    for name, df in splits.items():
        if set(df.label) != {0, 1}:
            raise ValueError(f"{name} needs both classes; use a larger subset.")
    return {"passed": True, "overlaps": overlaps,
            "note": "Identical feature states are reported, not treated as unique run identities."}


def prepare(path: Path, output: Path, per_repo: int | None = None) -> dict:
    if per_repo is not None and per_repo < 1:
        raise ValueError("--per-repo must be positive")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("CSV checksum differs from verified Mendeley v1; inspect the new source first.")
    df = pd.read_csv(path, low_memory=False)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing verified source columns: {sorted(missing)}")
    report = {"source_url": SOURCE, "source_sha256": digest, "license": "CC BY 4.0",
              "rows_loaded": len(df), "columns": list(df.columns),
              "raw_outcomes": df.conclusion.fillna("<missing>").value_counts().to_dict(),
              "raw_duplicate_repo_run_ids": int(df.duplicated(["repo", "run_id"]).sum())}
    counts = {}

    def retain(mask, reason):
        nonlocal df
        counts[reason] = int((~mask).sum())
        df = df.loc[mask].copy()

    retain(df.node_id.notna(), "not_identifiable_github_actions")
    retain(df.event.isin(["push", "pull_request"]), "other_trigger")
    retain(df.run_attempt.eq(1), "not_first_attempt")
    retain(df.conclusion.isin(["success", "failure"]), "not_success_or_failure")
    retain(df[["repo", "run_id", "commit_sha"]].notna().all(axis=1), "missing_identity")
    df["repo"] = df.repo.str.strip().str.lower()
    retain(df.repo.str.fullmatch(r"[^/\s]+/[^/\s]+"), "invalid_repository")
    retain(df.commit_sha.eq(df.head_sha), "commit_join_mismatch")
    created = pd.to_datetime(df.created_at, utc=True, errors="coerce")
    committed = pd.to_datetime(df.committed_date, utc=True, errors="coerce")
    retain(created.notna() & committed.notna() & committed.le(created), "invalid_or_future_commit_time")
    identity = ["repo", "run_id"]
    consistency = list(FEATURES) + ["commit_sha", "conclusion", "created_at"]
    conflicts = df.groupby(identity)[consistency].transform("nunique").gt(1).any(axis=1)
    retain(~conflicts, "conflicting_run_snapshots")
    counts["duplicate_runs"] = int(df.duplicated(identity).sum())
    df = df.drop_duplicates(identity).copy()
    df["id"] = "gha:" + df.repo + ":" + df.run_id.astype(str)
    df["label"] = df.conclusion.map({"success": 0, "failure": 1})
    df["state"] = df.apply(serialize_state, axis=1)
    splits = grouped_split(df)
    full_audit = audit_splits(splits)
    # Cap after group assignment so larger samples retain the same held-out repos.
    if per_repo is not None:
        splits = {name: pd.concat([g.sample(n=min(per_repo, len(g)), random_state=42)
                                  for _, g in part.groupby("repo")])
                  for name, part in splits.items()}
    audit = audit_splits(splits)
    report.update({"cohort": "GitHub Actions, push/pull_request, first attempt, success/failure",
                   "prediction_time": "workflow creation", "excluded_counts_sequential": counts,
                   "eligible_rows": len(df), "per_repo_cap": per_repo,
                   "features": list(FEATURES), "label_mapping": {"success": 0, "failure": 1},
                   "split_strategy": "repository-disjoint, seeds 42/43, no row fallback",
                   "full_cohort_audit": full_audit, "audit": audit, "splits": {}})
    output.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        records = part[["id", "state", "label", "commit_sha", "created_at"]].copy()
        records["group"] = part.repo
        records["label_name"] = part.conclusion
        records.to_json(output / f"{name}.jsonl", orient="records", lines=True)
        report["splits"][name] = {"rows": len(part), "failure_rate": float(part.label.mean()),
                                  "labels": part.label.value_counts().to_dict(),
                                  "repositories": sorted(part.repo.unique().tolist()),
                                  "sha256": hashlib.sha256((output / f"{name}.jsonl").read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-repo", type=int, help="Seeded sample cap per repository, after splitting")
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output, args.per_repo), indent=2))


if __name__ == "__main__":
    main()

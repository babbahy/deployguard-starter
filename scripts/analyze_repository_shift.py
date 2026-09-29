from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_audited_rows(data_dir: Path) -> pd.DataFrame:
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if not manifest.get("audit", {}).get("passed"):
        raise ValueError("Full dataset must pass its recorded leakage audit")
    frames = []
    for split, info in manifest["splits"].items():
        path = data_dir / f"{split}.jsonl"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != info["sha256"]:
            raise ValueError(f"{split} split hash differs from manifest")
        frame = pd.read_json(path, lines=True)
        frame["split"] = split
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    if rows.id.duplicated().any():
        raise ValueError("Run identity repeats across canonical splits")
    parsed = rows.state.map(lambda state: dict(
        line.split(": ", 1) for line in state.splitlines() if ": " in line
    ))
    fields = pd.DataFrame(parsed.tolist(), index=rows.index)
    for column in ("total_churn", "files_modified", "msg_len"):
        fields[column] = pd.to_numeric(fields[column].replace("unknown", np.nan), errors="coerce")
    rows = pd.concat([rows.drop(columns="state"), fields], axis=1)
    rows["created_at"] = pd.to_datetime(rows.created_at, utc=True, errors="coerce")
    if rows.created_at.isna().any():
        raise ValueError("Canonical run has invalid creation timestamp")
    return rows


def repository_summary(rows: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for (split, repo), frame in rows.groupby(["split", "group"], sort=True):
        event_counts = frame.event.value_counts()
        branch_counts = frame.head_branch.fillna("unknown").value_counts()
        summary = {
            "split": split,
            "repo": repo,
            "runs": len(frame),
            "failures": int(frame.label.sum()),
            "failure_rate": float(frame.label.mean()),
            "push_runs": int(event_counts.get("push", 0)),
            "push_share": float(event_counts.get("push", 0) / len(frame)),
            "pull_request_runs": int(event_counts.get("pull_request", 0)),
            "pull_request_share": float(event_counts.get("pull_request", 0) / len(frame)),
            "churn_median": frame.total_churn.median(),
            "churn_p90": frame.total_churn.quantile(0.9),
            "files_modified_median": frame.files_modified.median(),
            "files_modified_p90": frame.files_modified.quantile(0.9),
            "message_length_median": frame.msg_len.median(),
            "message_length_p90": frame.msg_len.quantile(0.9),
            "merge_share": float(frame.is_merge.astype(str).str.lower().eq("true").mean()),
            "branch_count": int(frame.head_branch.nunique(dropna=True)),
            "top_branches": json.dumps(branch_counts.head(3).to_dict(), sort_keys=True),
            "first_created_at": frame.created_at.min().isoformat(),
            "last_created_at": frame.created_at.max().isoformat(),
        }
        summaries.append(summary)
    return pd.DataFrame(summaries).sort_values(["split", "repo"]).reset_index(drop=True)


def split_summary(rows: pd.DataFrame) -> pd.DataFrame:
    result = []
    for split, frame in rows.groupby("split", sort=False):
        events = frame.event.value_counts(normalize=True)
        result.append({
            "split": split,
            "runs": len(frame),
            "repositories": frame.group.nunique(),
            "failures": int(frame.label.sum()),
            "failure_rate": float(frame.label.mean()),
            "push_share": float(events.get("push", 0)),
            "pull_request_share": float(events.get("pull_request", 0)),
            "median_churn": frame.total_churn.median(),
            "median_files_modified": frame.files_modified.median(),
            "median_message_length": frame.msg_len.median(),
            "merge_share": float(frame.is_merge.astype(str).str.lower().eq("true").mean()),
        })
    return pd.DataFrame(result)


def write_report(rows: pd.DataFrame, per_repo: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_repo.to_csv(output_dir / "repository_shift.csv", index=False)
    by_split = split_summary(rows)
    lines = [
        "# Repository Distribution Shift",
        "",
        "Descriptive statistics for the unchanged repository-disjoint `cicd_full` splits.",
        "These split-level differences are confounded with repository identity because each repository belongs to exactly one split.",
        "",
        "## Split Summary",
        "",
        "```text",
        by_split.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "```",
        "",
        "## Repository Summary",
        "",
        "`push share`, median churn, and median message length show how different individual repositories are.",
        "The CSV includes p90s, branch counts/top branches, merge shares, and date ranges.",
        "",
        "```text",
        per_repo[["split", "repo", "runs", "failures", "failure_rate", "push_share", "churn_median", "files_modified_median", "message_length_median", "merge_share", "top_branches"]].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "```",
        "",
        "## Interpretation",
        "",
        "The repository-disjoint evaluation answers transfer to unseen repositories; it does not estimate performance after learning a customer's own history.",
        "Repository priors and event/workflow conventions can dominate pooled metrics. The separate temporal experiment evaluates later runs from repositories represented in its training period.",
        "",
    ]
    (output_dir / "repository_shift.md").write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repository shift in audited CI splits")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cicd_full"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    rows = load_audited_rows(args.data_dir)
    per_repo = repository_summary(rows)
    write_report(rows, per_repo, args.output_dir)
    print(f"Wrote {len(per_repo)} repository summaries for {len(rows)} runs to {args.output_dir}")
    print(split_summary(rows).to_string(index=False))


if __name__ == "__main__":
    main()

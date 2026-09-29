from __future__ import annotations

import heapq

import numpy as np
import pandas as pd


HISTORY_FEATURES = [
    "hist_workflow_previous_outcome", "hist_workflow_failure_rate",
    "hist_branch_failure_rate", "hist_failures_previous_5_runs",
    "hist_failures_previous_10_runs", "hist_failures_previous_7d",
    "hist_failures_previous_30d", "hist_runs_previous_7d",
    "hist_runs_previous_30d", "hist_hours_since_previous_failure",
    "hist_average_churn_previous_30d",
]


def add_causal_history(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "repo", "created_at", "updated_at", "label", "workflow_id",
                "head_branch", "total_churn"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"History features require: {sorted(missing)}")
    data = rows.copy()
    data["created_at"] = pd.to_datetime(data.created_at, utc=True, errors="coerce")
    data["updated_at"] = pd.to_datetime(data.updated_at, utc=True, errors="coerce")
    if data.created_at.isna().any() or data.updated_at.isna().any():
        raise ValueError("History construction requires valid created_at and updated_at")
    if data.updated_at.lt(data.created_at).any():
        raise ValueError("Outcome availability timestamp precedes run creation")
    if data.id.duplicated().any() or not set(data.label.unique()).issubset({0, 1}):
        raise ValueError("History input must have unique identities and binary labels")
    data["workflow_id"] = data.workflow_id.fillna("unknown").astype(str)
    data["head_branch"] = data.head_branch.fillna("unknown").astype(str)
    data["total_churn"] = pd.to_numeric(data.total_churn, errors="coerce")
    data = data.sort_values(["created_at", "id"], kind="stable").reset_index(drop=True)

    completion_events = []
    for row in data.itertuples(index=False):
        completion_events.append((row.updated_at, row.created_at, str(row.id), row))
    completion_events.sort(key=lambda item: (item[0], item[1], item[2]))
    by_repo: dict[str, list] = {}
    event_index = 0
    features = {name: [np.nan] * len(data) for name in HISTORY_FEATURES}

    for timestamp, batch in data.groupby("created_at", sort=True):
        # Only outcomes known strictly before this prediction timestamp enter history.
        while event_index < len(completion_events) and completion_events[event_index][0] < timestamp:
            _, created_at, _, row = completion_events[event_index]
            by_repo.setdefault(str(row.repo), []).append((
                created_at, int(row.label), str(row.workflow_id), str(row.head_branch),
                float(row.total_churn) if pd.notna(row.total_churn) else np.nan,
            ))
            event_index += 1

        for index, row in batch.iterrows():
            history = by_repo.get(str(row.repo), [])
            workflow_history = [item for item in history if item[2] == row.workflow_id]
            branch_history = [item for item in history if item[3] == row.head_branch]
            ordered_history = sorted(history, key=lambda item: item[0])
            previous_5 = ordered_history[-5:]
            previous_10 = ordered_history[-10:]
            since_7d = [item for item in history if timestamp - pd.Timedelta(days=7) <= item[0] < timestamp]
            since_30d = [item for item in history if timestamp - pd.Timedelta(days=30) <= item[0] < timestamp]
            failures = [item for item in history if item[1] == 1]
            features["hist_workflow_previous_outcome"][index] = (
                workflow_history[-1][1] if workflow_history else np.nan
            )
            features["hist_workflow_failure_rate"][index] = (
                float(np.mean([item[1] for item in workflow_history])) if workflow_history else np.nan
            )
            features["hist_branch_failure_rate"][index] = (
                float(np.mean([item[1] for item in branch_history])) if branch_history else np.nan
            )
            features["hist_failures_previous_5_runs"][index] = sum(item[1] for item in previous_5)
            features["hist_failures_previous_10_runs"][index] = sum(item[1] for item in previous_10)
            features["hist_failures_previous_7d"][index] = sum(item[1] for item in since_7d)
            features["hist_failures_previous_30d"][index] = sum(item[1] for item in since_30d)
            features["hist_runs_previous_7d"][index] = len(since_7d)
            features["hist_runs_previous_30d"][index] = len(since_30d)
            features["hist_hours_since_previous_failure"][index] = (
                (timestamp - failures[-1][0]).total_seconds() / 3600.0 if failures else np.nan
            )
            churns = [item[4] for item in since_30d if np.isfinite(item[4])]
            features["hist_average_churn_previous_30d"][index] = (
                float(np.mean(churns)) if churns else np.nan
            )

    for name, values in features.items():
        data[name] = values
    return data


def build_history_state(row: pd.Series, base_state: str) -> str:
    lines = [base_state]
    for name in HISTORY_FEATURES:
        value = row[name]
        if pd.isna(value):
            rendered = "unknown"
        elif isinstance(value, (float, np.floating)):
            rendered = format(float(value), ".12g")
        else:
            rendered = str(value)
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines)

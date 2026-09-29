from __future__ import annotations

from bisect import bisect_left

import numpy as np
import pandas as pd

from deployguard.data.history import HISTORY_FEATURES


def add_causal_history(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "repo", "created_at", "updated_at", "label", "workflow_id",
                "head_branch", "total_churn"}
    if required - set(rows.columns):
        raise ValueError(f"History features require: {sorted(required - set(rows.columns))}")
    data = rows.copy()
    data["created_at"] = pd.to_datetime(data.created_at, utc=True, errors="coerce")
    data["updated_at"] = pd.to_datetime(data.updated_at, utc=True, errors="coerce")
    if data.created_at.isna().any():
        raise ValueError("History construction requires valid creation timestamps")
    if (data.updated_at.notna() & data.updated_at.lt(data.created_at)).any():
        raise ValueError("Outcome availability timestamp precedes run creation")
    if data.id.duplicated().any() or not set(data.label.unique()).issubset({0, 1}):
        raise ValueError("History input must have unique IDs and binary labels")
    data["workflow_id"] = data.workflow_id.fillna("unknown").astype(str)
    data["head_branch"] = data.head_branch.fillna("unknown").astype(str)
    data["total_churn"] = pd.to_numeric(data.total_churn, errors="coerce")
    data = data.sort_values(["created_at", "id"], kind="stable").reset_index(drop=True)
    completed = sorted(
        ((r.updated_at, str(r.id), r) for r in data[data.updated_at.notna()].itertuples(index=False)),
        key=lambda event: (event[0], event[1]),
    )

    times, labels, fail_prefix, churn_prefix, churn_count_prefix = {}, {}, {}, {}, {}
    workflow_stats, workflow_latest, branch_stats, last_failure = {}, {}, {}, {}
    event_i = 0
    features = {name: [np.nan] * len(data) for name in HISTORY_FEATURES}
    for timestamp, batch in data.groupby("created_at", sort=True):
        # Complete events strictly earlier than this prediction are now observable.
        while event_i < len(completed) and completed[event_i][0] < timestamp:
            available_at, _, r = completed[event_i]
            repo, workflow, branch = str(r.repo), str(r.workflow_id), str(r.head_branch)
            label = int(r.label)
            times.setdefault(repo, []).append(available_at)
            labels.setdefault(repo, []).append(label)
            fail_prefix.setdefault(repo, []).append(label + (fail_prefix[repo][-1] if fail_prefix[repo] else 0))
            churn = float(r.total_churn) if pd.notna(r.total_churn) else 0.0
            churn_prefix.setdefault(repo, []).append(churn + (churn_prefix[repo][-1] if churn_prefix[repo] else 0.0))
            churn_count_prefix.setdefault(repo, []).append(
                int(pd.notna(r.total_churn)) + (churn_count_prefix[repo][-1] if churn_count_prefix[repo] else 0)
            )
            w = workflow_stats.setdefault((repo, workflow), [0, 0])
            w[0] += 1
            w[1] += label
            workflow_latest[(repo, workflow)] = label
            b = branch_stats.setdefault((repo, branch), [0, 0])
            b[0] += 1
            b[1] += label
            if label:
                last_failure[repo] = available_at
            event_i += 1

        for idx, row in batch.iterrows():
            repo, workflow, branch = str(row.repo), str(row.workflow_id), str(row.head_branch)
            hist_times, hist_labels = times.get(repo, []), labels.get(repo, [])
            n = len(hist_times)
            i7 = bisect_left(hist_times, timestamp - pd.Timedelta(days=7))
            i30 = bisect_left(hist_times, timestamp - pd.Timedelta(days=30))
            fp = fail_prefix.get(repo, [])
            cp = churn_prefix.get(repo, [])
            ccp = churn_count_prefix.get(repo, [])
            w, b = workflow_stats.get((repo, workflow)), branch_stats.get((repo, branch))
            features["hist_workflow_previous_outcome"][idx] = workflow_latest.get((repo, workflow), np.nan)
            features["hist_workflow_failure_rate"][idx] = w[1] / w[0] if w else np.nan
            features["hist_branch_failure_rate"][idx] = b[1] / b[0] if b else np.nan
            features["hist_failures_previous_5_runs"][idx] = sum(hist_labels[max(0, n - 5):n])
            features["hist_failures_previous_10_runs"][idx] = sum(hist_labels[max(0, n - 10):n])
            features["hist_failures_previous_7d"][idx] = fp[-1] - (fp[i7 - 1] if i7 else 0) if n else 0
            features["hist_failures_previous_30d"][idx] = fp[-1] - (fp[i30 - 1] if i30 else 0) if n else 0
            features["hist_runs_previous_7d"][idx] = n - i7
            features["hist_runs_previous_30d"][idx] = n - i30
            features["hist_hours_since_previous_failure"][idx] = (
                (timestamp - last_failure[repo]).total_seconds() / 3600 if repo in last_failure else np.nan
            )
            churn_count = ccp[-1] - (ccp[i30 - 1] if i30 else 0) if n else 0
            churn_sum = cp[-1] - (cp[i30 - 1] if i30 else 0.0) if n else 0.0
            features["hist_average_churn_previous_30d"][idx] = churn_sum / churn_count if churn_count else np.nan
    for name, values in features.items():
        data[name] = values
    return data

import pandas as pd
import pytest

from deployguard.data.history import HISTORY_FEATURES
from deployguard.data.history_fast import add_causal_history


def rows():
    return pd.DataFrame([
        {"id": "a", "repo": "org/repo", "created_at": "2026-01-01T00:00:00Z",
         "updated_at": "2026-01-01T02:00:00Z", "label": 1, "workflow_id": "ci",
         "head_branch": "main", "total_churn": 5},
        {"id": "b", "repo": "org/repo", "created_at": "2026-01-01T01:00:00Z",
         "updated_at": "2026-01-01T01:30:00Z", "label": 0, "workflow_id": "ci",
         "head_branch": "main", "total_churn": 7},
        {"id": "c", "repo": "org/repo", "created_at": "2026-01-01T01:00:00Z",
         "updated_at": "2026-01-01T01:40:00Z", "label": 1, "workflow_id": "lint",
         "head_branch": "main", "total_churn": 9},
        {"id": "d", "repo": "org/repo", "created_at": "2026-01-01T03:00:00Z",
         "updated_at": "2026-01-01T03:30:00Z", "label": 0, "workflow_id": "ci",
         "head_branch": "main", "total_churn": 3},
    ])


def test_outcomes_are_only_admitted_after_they_become_available():
    result = add_causal_history(rows()).set_index("id")
    assert result.loc["b", "hist_runs_previous_30d"] == 0
    assert result.loc["c", "hist_runs_previous_30d"] == 0
    assert result.loc["d", "hist_runs_previous_30d"] == 3
    assert result.loc["d", "hist_failures_previous_5_runs"] == 2
    assert result.loc["d", "hist_workflow_previous_outcome"] == 1
    assert result.loc["d", "hist_branch_failure_rate"] == pytest.approx(2 / 3)


def test_changing_future_labels_cannot_change_earlier_features():
    source = rows()
    baseline = add_causal_history(source).set_index("id")
    changed = source.copy()
    changed.loc[changed.id.isin(["c", "d"]), "label"] = 1 - changed.loc[
        changed.id.isin(["c", "d"]), "label"
    ]
    altered = add_causal_history(changed).set_index("id")
    pd.testing.assert_frame_equal(
        baseline.loc[["a", "b", "c"], HISTORY_FEATURES],
        altered.loc[["a", "b", "c"], HISTORY_FEATURES],
    )


def test_same_timestamp_outcomes_do_not_leak_between_sibling_runs():
    source = rows()
    baseline = add_causal_history(source).set_index("id")
    changed = source.copy()
    changed.loc[changed.id.eq("b"), "label"] = 1
    altered = add_causal_history(changed).set_index("id")
    pd.testing.assert_series_equal(
        baseline.loc["c", HISTORY_FEATURES], altered.loc["c", HISTORY_FEATURES]
    )


def test_invalid_outcome_availability_fails_closed():
    source = rows()
    source.loc[0, "updated_at"] = "2025-12-31T23:59:59Z"
    with pytest.raises(ValueError, match="precedes run creation"):
        add_causal_history(source)

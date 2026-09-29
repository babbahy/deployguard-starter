import pandas as pd
import pytest

from deployguard.data.temporal import (
    _audit_temporal_splits, _commit_purge, _rich_state, rich_temporal_v1_state,
)


def temporal_row(identity, commit, timestamp):
    return {
        "id": identity,
        "commit_sha": commit,
        "repo": "owner/repo",
        "created_at": pd.Timestamp(timestamp, tz="UTC"),
        "label": 0,
    }


def test_commit_repeated_in_future_partition_is_purged():
    rows = pd.DataFrame([
        temporal_row("a", "same", "2024-01-01"),
        {**temporal_row("b", "same", "2024-02-01"), "split": "validation"},
    ])
    rows.loc[0, "split"] = "train"
    kept, removed = _commit_purge(rows)
    assert removed == 1
    assert kept.id.tolist() == ["a"]


def test_temporal_audit_accepts_ordered_nonoverlapping_splits():
    splits = {
        "train": pd.DataFrame([temporal_row("a", "sha-a", "2024-01-01")]),
        "validation": pd.DataFrame([temporal_row("b", "sha-b", "2024-02-01")]),
        "test": pd.DataFrame([temporal_row("c", "sha-c", "2024-03-01")]),
    }
    assert _audit_temporal_splits(splits)["passed"]


def test_temporal_audit_rejects_future_to_past_order():
    splits = {
        "train": pd.DataFrame([temporal_row("a", "sha-a", "2024-03-01")]),
        "validation": pd.DataFrame([temporal_row("b", "sha-b", "2024-02-01")]),
        "test": pd.DataFrame([temporal_row("c", "sha-c", "2024-04-01")]),
    }
    with pytest.raises(ValueError, match="chronology"):
        _audit_temporal_splits(splits)


def test_rich_text_contains_only_preoutcome_fields():
    row = pd.Series({
        "repo": "owner/repo", "event": "push", "head_branch": "main",
        "workflow_id": "42", "run_number": 10, "commit_message": "fix build",
        "additions": 2, "deletions": 1, "num_parents": 1, "total_churn": 3,
        "files_modified": 1, "msg_len": 9, "is_merge": False,
        "commit_age_minutes": 1.5, "commit_hour": 11, "commit_weekday": 1,
        "created_hour": 12, "created_weekday": 1,
        "conclusion": "success", "status": "completed", "duration": 10,
        "updated_at": "2024-01-01", "time_since_last_commit": 15,
    })
    state = _rich_state(row)
    for forbidden in ("conclusion", "status", "duration", "updated_at",
                      "time_since_last_commit"):
        assert forbidden not in state
    assert "commit_message: fix build" in state


def test_rich_temporal_v1_state_uses_only_audited_features():
    row = {
        "repo": "owner/repo", "event": "push", "head_branch": "main",
        "workflow_id": "42", "run_number": 10, "commit_message": "fix build",
        "additions": 2, "deletions": 1, "num_parents": 1, "total_churn": 3,
        "files_modified": 1, "msg_len": 9, "is_merge": False,
        "created_hour": 12, "created_weekday": 1,
        "label": 1, "conclusion": "failure", "status": "completed",
        "duration": 10, "updated_at": "2024-01-01", "commit_sha": "secret",
    }
    state = rich_temporal_v1_state(row)
    assert len(state.splitlines()) == 15
    assert state.splitlines()[-1].startswith("commit_message:")
    for field in ("repository: owner/repo", "commit_message: fix build",
                  "created_hour_utc: 12", "created_weekday_utc: 1"):
        assert field in state
    for forbidden in ("label", "conclusion", "status", "duration", "updated_at", "commit_sha"):
        assert forbidden not in state

import pandas as pd
import pytest

from deployguard.data.prepare_cicd import FEATURES, audit_splits, grouped_split, serialize_state
from deployguard.evaluation.metrics import binary_metrics


def test_outcome_mutation_cannot_change_input():
    row = pd.Series(dict.fromkeys(FEATURES, 1))
    row['conclusion'] = 'success'
    before = serialize_state(row)
    for key in ('conclusion', 'status', 'duration', 'updated_at', 'time_since_last_commit', 'run_attempt'):
        row[key] = 'POST_OUTCOME_SENTINEL'
    assert serialize_state(row) == before
    assert 'POST_OUTCOME_SENTINEL' not in before


def test_no_random_row_fallback():
    with pytest.raises(ValueError, match='fallback'):
        grouped_split(pd.DataFrame({'repo': ['a/b'] * 20}))


def test_group_split_and_shared_commit_detection():
    df = pd.DataFrame([{'repo': f'owner/repo{i}', 'id': f'{i}:{j}',
                        'commit_sha': f'sha{i}:{j}', 'state': str(j), 'label': j % 2}
                       for i in range(30) for j in range(4)])
    splits = grouped_split(df)
    assert audit_splits(splits)['passed']
    splits['test'].iloc[0, splits['test'].columns.get_loc('commit_sha')] = splits['train'].iloc[0].commit_sha
    with pytest.raises(ValueError, match='leakage'):
        audit_splits(splits)


def test_single_class_metrics_are_explicitly_undefined():
    result = binary_metrics([0, 0], [0.1, 0.2])
    assert result['roc_auc'] is None
    assert result['recall_at_fpr_1pct'] is None

# Prediction-Time Feature Audit

**Scope:** richer state used by the separate cicd_temporal benchmark. The prediction point is workflow creation. The original repository-disjoint dataset and its six-field state are unchanged.

The public CSV is a merged research dataset, not an event-time feature store. The availability decisions below are therefore explicit hypotheses based on the field meaning and the run/commit timestamps. Before product use, each feature must be captured from the real webhook/API payload as it existed when the workflow was created.

| Source field | Temporal input | Decision and prediction-time rationale |
| --- | --- | --- |
| repo | repository | Included for the customer-specific experiment. The repository is known from the incoming run event. This intentionally does not test unseen-repository transfer. |
| event | event | Included. push or pull_request is the workflow trigger type, known at creation. |
| head_branch | branch | Included. The head/ref name is part of the trigger/run metadata. |
| workflow_id | workflow_id | Included as a category. It identifies the workflow definition that was selected for the run. Missing values become unknown. |
| run_number | run_number | Included as numeric workflow history. It is assigned when the run is created. It is also a time proxy, so performance may depend on temporal trends and drift. |
| message | commit_message | Included as text. It describes the already-created head commit that triggered the run. It is not a future workflow result. Newlines are flattened. |
| additions, deletions | same names | Included. These describe the pushed/PR change. This assumes the source's commit/diff join reflects data available at workflow creation; the raw dataset does not document its extraction timestamp. Verify against event-time production logs. |
| num_parents | same name | Included. Commit ancestry is part of the already-existing head commit. Source extraction timing still needs production-parity validation. |
| total_churn, files_modified | same names | Included. Change-size metadata from the commit/diff. Same source-timing caveat as additions/deletions. |
| msg_len | same name | Included for comparability with the prior model; derived from the commit message and partly redundant with its text. |
| is_merge | same name | Included. It describes the already-existing commit/ref state, not the workflow outcome. Validate the source's exact merge-definition semantics before product use. |
| created_at | created_hour, created_weekday | The full timestamp is used for chronological partitioning and metadata only. UTC hour and weekday are derived features; the workflow creation time is known at the prediction point. |
| committed_date | commit_age_minutes, commit_hour, commit_weekday | Added in temporal v2. Canonical filtering verifies the commit time is no later than workflow creation; these are derived in UTC. The raw CSV does not document its extraction timestamp, so confirm event-time parity before product use. |
| commit_sha | metadata only | Used to audit repeated commits across splits; never included in model text/features. |
| time_since_last_commit | excluded | The research CSV provides a precomputed number but does not document its lookback, aggregation or exact calculation time. Excluded until provenance is verified. |
| authored_date, timestamp, timestamp_raw, commit_date | excluded | The merged CSV has inconsistent/partly missing representations for these fields; they are not used in the temporal state. |
| conclusion, status, duration, updated_at, run_started_at | excluded | Outcome, terminal/intermediate state, elapsed runtime, or timestamps updated after workflow creation. These would leak future execution information. |
| author/committer names and emails; actor logins | excluded | Not required for this test, can encode identity/privilege proxies, and may be sensitive. |
| run_attempt | cohort filter only | The cohort keeps first attempts. It is not an input; retries occur after an initial failure and would change the prediction population. |

## Join and State Construction

The richer rows are joined back to the raw CSV using canonical run ID, commit SHA, label and normalized creation timestamp, then checked against the exact original six-feature state. The join yields exactly one matching raw row for each of the 68,354 canonical identities; taking the first duplicate raw run ID without these checks would have selected wrong/incomplete metadata for many rows.

Temporal v1 includes repository, trigger, branch, workflow ID, run number, commit message, additions, deletions, parent count, churn, files, message length, merge flag, and UTC creation hour/weekday. Temporal v2 preserves v1 and adds only commit age, commit hour, and commit weekday derived from committed_date. No conclusion/status/duration/update-time field is serialized. The undocumented time_since_last_commit field is intentionally absent from both versions.

Within the selected temporal cohort, the richer fields have one unknown branch value; 8,439 distinct branch strings and 137 workflow IDs occur across 19 repositories. Rare/unseen categories are handled by train-fitted preprocessing rather than learned from validation/test.

These are dataset-level availability judgments, not proof of production feature parity. A real deployment-risk model should log the exact input snapshot at workflow creation and reject fields that only become available later.

# Dataset contract and audit

Verified on 2026-09-27 against Mendeley version 1:
https://data.mendeley.com/datasets/mggwn7rj9f/1

Attribution: Joshua Faruna and Funmilayo Sanusi, *A comprehensive GitHub
Actions dataset for CI/CD pipeline failure forecasting*, 2026,
DOI 10.17632/mggwn7rj9f.1, CC BY 4.0. This project filters and serializes
their data; the resulting cohort is not the full published benchmark.

The file `final_research_dataset_MASTER.csv` is 194,638,363 bytes,
303,079 rows and 38 columns. SHA-256:
`b6a89418e39e6144f4ac71496256474c7bc27651e099557fe074d7798da1e73e`.
The manifest records all columns and exclusion counts.

## Source problems and chosen scope

- 50,529 repeated `(repo, run_id)` identities in the full CSV.
- 91,047 missing `conclusion` values. Some records instead put outcomes in
  `status`; silently guessing a common label schema is unsafe.
- `repo` is populated throughout; `repo_name` is mostly empty.
- The source mixes archival TravisTorrent and GitHub Actions, with very
  different missing-field patterns. We require a GitHub Actions `node_id`.
- Triggers such as `workflow_run` and `status` can occur after another
  workflow result. We restrict this experiment to `push` and `pull_request`.
- We require `run_attempt == 1`, then exclude attempt number from features.
  The exported final retry count would be unsafe for predicting an initial run.
- We use only literal `success=0` and `failure=1`. Cancellations, skipped runs,
  startup failures and intervention outcomes are outside this first task.
- 66 otherwise eligible records have invalid/future commit timing and are
  excluded. Matching `commit_sha` and `head_sha` is required.

This leaves 68,354 runs from 30 repositories. Restricting to observed first
attempts and terminal outcomes creates selection bias: rerun and cancelled
workflows are not represented. This is a conditional retrospective benchmark,
not yet an unbiased production failure predictor.

## Prediction-time feature review

Prediction time: creation of the workflow run, after the triggering commit.

| Field | Why included / excluded |
| --- | --- |
| event, head_branch | Trigger context available when the run is created |
| total_churn, files_modified | Properties of the triggering commit diff |
| msg_len | Length of the existing commit message |
| is_merge | Commit parentage, not later PR merge status |
| conclusion | Label only |
| status, duration, updated_at | Post-outcome; never serialized |
| run_attempt | Cohort filter only; final retry count can leak |
| time_since_last_commit | Excluded: computation direction/order not verified |
| timestamps | Metadata/timing checks only; no historical aggregates |
| repo, run_id, commit_sha | Audit and split identities only |
| authors, emails, messages | Excluded from model input |

The authors describe GitPython commit extraction in their paper:
https://pmc.ncbi.nlm.nih.gov/articles/PMC13503099/
The supplementary script could not be retrieved as a usable ZIP in this
session. Commit-feature semantics rely on that description and consistency
checks, not an independent reconstruction of every diff.

## Split audit

GroupShuffleSplit assigns repositories with fixed seeds 42/43. There is no
random-row fallback. Split membership is decided before sampling and remains
stable as the per-repository sample cap increases. Both classes are required
in every split. Full-cohort and subset audits reject shared repositories,
run IDs or commit SHAs.

The 32-per-repository smoke subset has 672/128/160 train/validation/test rows,
on 21/4/5 distinct repositories. Failure counts are 78/19/9. There are zero
shared repositories, runs, commits or input strings between these subsets.
The full cohort has 85/82/86 shared input strings between split pairs. These
are collisions in coarse features of different commits, not duplicated run
identities; they are reported rather than deleted based on held-out data.

This tests unseen-repository generalization, not future-time generalization.
Branch naming can encode project conventions. Repeated commits within a
split make rows correlated. A later benchmark needs multiple group splits,
repository-level uncertainty, a temporal holdout and tabular baselines.
Do not infer meaningful calibration or 1% FPR performance from nine test
failures. The current PR-AUC field uses average precision, not trapezoidal area.

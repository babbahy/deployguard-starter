# Experiment History

This project started with the universal question: can pre-run CI metadata predict success/failure across repositories? The first full result did not support that formulation. The project direction now tests whether customer-specific history plus richer event-time context is more useful. Negative results are retained rather than rewritten.

## 1. Data and Leakage Audit

The public Mendeley v1 source was checksum-verified and narrowed from 303,079 mixed-schema records to 68,354 first-attempt GitHub Actions push/pull_request runs with explicit success/failure outcomes. Only six pre-outcome fields entered the initial state. Splits were repository-disjoint; run identity, commit and repository overlap checks passed. See docs/DATA_AUDIT.md.

## 2. Machinery Smoke Runs

A two-update CPU ModernBERT run established that the local Hugging Face training path worked. A 100-update Modal L4 pilot completed, but achieved zero failure recall at threshold 0.5 and PR-AUC 0.0545 against 0.0563 test prevalence. It was a smoke experiment, not a useful detector. The checkpoint produced finite L4 probabilities but non-finite CPU logits.

## 3. Full Repository-Disjoint Experiment

On the 68,354-run cohort, logistic regression using the six original features scored test PR-AUC 0.0361 versus test prevalence 0.0400, and ROC-AUC 0.4526. ModernBERT trained for 8,850 updates on one Modal L4 scored PR-AUC 0.0324 and ROC-AUC 0.4241; it caught 1 of 595 test failures at threshold 0.5. Both were effectively non-predictive under this unseen-repository split.

Repository-level analysis exposed major shift: split failure rates were 8.31% / 20.85% / 4.00%; validation was 99.4% push and 0.2% merge, unlike train/test. See analysis/repository_shift.md.

The full ModernBERT checkpoint also has severe numerical portability problems: maximum absolute parameter 8.46e37, with CPU float32 and bfloat16 producing NaN logits. The independent L4 rescore was finite and reproduced aggregate metrics, but no per-example GPU predictions were saved. The requested CSV export therefore remains blocked; the fail-closed status is in analysis/modernbert_prediction_export_status.json. No additional Modal compute was used for this analysis.

## 4. Temporal Customer-Specific Experiment

A separate dataset, data/processed/cicd_temporal, keeps repositories represented in both history and future test periods. The rule is at least 300 total runs and at least 20 failures in each repository's first 70% training window. Nineteen of 30 repositories qualify. Each is split chronologically 70/15/15; 107 later copies of commits already present in earlier partitions were removed. The audit found no cross-split run/commit overlap and no future-to-past ordering.
A preserved v2, data/processed/cicd_temporal_v2, adds commit age/hour/weekday derived from committed_date after verifying it precedes workflow creation. It uses the same rows and split IDs as v1; v1 remains available for reproducibility.

CPU-only v1 baselines on 8,788 future-period test runs (8.64% failures):

- Prevalence reference: PR-AUC 0.0864, ROC-AUC 0.500.
- Repo-only logistic: PR-AUC 0.2003, ROC-AUC 0.698.
- Rich TF-IDF + structured logistic: PR-AUC 0.3717, ROC-AUC 0.851.
- Rich logistic without repository identity: PR-AUC 0.3499, ROC-AUC 0.836.
- Structured-only histogram boosting: PR-AUC 0.3610, ROC-AUC 0.826.
On v2, rich logistic scored PR-AUC 0.3719 / ROC-AUC 0.851 and structured boosting 0.3574 / 0.826. Commit-time features changed logistic PR-AUC by only +0.0002 and reduced boosting PR-AUC by 0.0036; they added no material lift.

Repository identity contributes, but does not explain all of the signal. Rich models beat the within-repository prevalence PR-AUC in 16/19 or 17/19 repositories; individual repositories still vary substantially. At a validation-selected 90%-precision threshold, test recall was only 2.2-2.8%, so this is not yet a good alert policy. Detailed v1/v2 metrics are in analysis/temporal_experiment.md, analysis/temporal_baselines/, and analysis/temporal_baselines_v2/.

## Current Direction

Temporal v1 is frozen as `temporal-v1-3e9cc5c049399d0b`. The original six-field `state` string is a compact baseline input, not the rich v1 feature state. Feature timing assumptions for the rich state are documented in `docs/FEATURE_AUDIT.md`; future architecture comparisons must render all audited v1 model features from their separate columns. Historical availability features remain isolated in temporal-history-v2.

The eventual product requires deployments, incidents/rollbacks, service context and production telemetry, not just CI outcome prediction. No Jev result is included; it remains a future hosted external reference.

Hypothesis: workflow/history structure, not commit prose or change magnitude alone, carries the signal. Dataset: immutable `data/processed/cicd_temporal`, benchmark `temporal-v1-3e9cc5c049399d0b`; no split bytes were rewritten. Models: fixed C=1 logistic ablations and train-prevalence reference. The test set was used once for reporting; 95% intervals use 1,000 repository-cluster bootstrap replicates.

| Inputs | AP | ROC-AUC | Repo-bootstrap AP interval |
| --- | ---: | ---: | ---: |
| Train prevalence | 0.0864 | 0.500 | 0.052-0.129 |
| Commit text only | 0.1692 | 0.642 | 0.070-0.275 |
| Structured metadata, no repo | 0.3546 | 0.850 | 0.243-0.503 |
| Change size only | 0.0922 | 0.501 | 0.052-0.144 |
| Workflow/history proxies only | 0.3491 | 0.844 | 0.241-0.497 |


## Frozen Temporal-v1 ModernBERT: Compact-State Result

Hypothesis: a pretrained encoder can predict failures from pre-run context. Dataset: frozen benchmark `temporal-v1-3e9cc5c049399d0b`; all nine sweep points used the locked validation split, and the selected setup was evaluated on the locked test split only after selection. Important input audit: this first run tokenized the compact six-field `state` string, not all audited rich feature columns. Preserve it as a compact-state negative result; it does not satisfy the requested rich-state Transformer comparison.

Selection used validation AP only. The best of LR `{1e-5, 2e-5, 5e-5}` x epochs `{2, 3, 5}` was LR `5e-5`, 3 epochs, validation AP 0.2077 / ROC-AUC 0.6881. Test prevalence is 0.0864 (8,788 rows).

| Seed | Test AP | ROC-AUC |
| ---: | ---: | ---: |
| 17 | 0.1521 | 0.6376 |
| 42 | 0.1551 | 0.6365 |
| 73 | 0.1619 | 0.6532 |
| Mean +/- SD | 0.1564 +/- 0.0050 | 0.6424 +/- 0.0093 |

This trails the rich TF-IDF + structured logistic v1 baseline (AP 0.3716 / ROC-AUC 0.851) and HGB (AP 0.357). All three seeds passed finite loss, gradient, parameter and probability checks; maximum parameter magnitude was about 4.42, peak GPU memory about 3.11 GB, training 19.2-19.4 minutes, and inference throughput 1,341-1,419 rows/s. The 8,788-row prediction exports and run reports are in `runs/temporal_v1_modernbert/final/`; sweep records are in `runs/temporal_v1_modernbert/sweep/`. Repository-cluster bootstrap intervals were broad and overlapped prevalence. Conclusion: the model is numerically healthy but this compact input underperforms. Next decision: run a corrected rich-state comparison without changing the frozen dataset.

## Rich-State Tokenization and Sweep Contract

Tokenizer audit used only the 41,191 temporal-v1 training rows with the pinned ModernBERT tokenizer revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`. Rich-state lengths including special tokens: p50 137, p90 331, p95 462, p99 1,039, maximum 12,064. Fractions exceeding max length: 128 = 64.09%; 256 = 15.52%; 512 = 4.12% (1,697 / 41,191). Selected context is 512: it contains at least 95% of normal states while keeping fine-tuning practical. The formatter places the 14 concise structured fields first and `commit_message` last. Decoded p95, p99 and maximum examples retained all 15 field names at 512; very long commit prose is truncated only after the structured context. Audit: `analysis/temporal_v1_rich_token_audit.json`.

The first rich-state launcher attempt was interrupted at 81 / 5,150 optimizer steps after the requested token audit began; it produced no validation or test result. Two preceding attempts failed before training due to an output-directory collision. Preserve these as infrastructure failures, not model results. The corrected launcher submits each learning-rate/epoch pair as a separate Modal function invocation with its own 90-minute timeout and unique run ID. Sweep mode uploads and reads only train and validation rows, skips test prediction export, and selects by validation AP. Final mode alone uploads test rows and evaluates seeds 17, 42 and 73 after selection. No benchmark file or split hash was modified.

Upstream pins checked 2026-09-27: Laya repository `4066d5d5fbf08b66c6757ddeedbd797bd7655bc0`, Laya model `convaiinnovations/laya` revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; CLM repository `bb42c6c5bf914fd449bed2f6ca65be80602cb1f7`. Record these revisions in typed-model run metadata rather than referring to `main`.


## Temporal-History-v2 CPU Baselines

Hypothesis: strictly earlier outcomes add customer-specific signal. Dataset: separate `data/processed/cicd_temporal_history_v2`, retaining the exact temporal-v1 IDs and split membership. Historical outcomes are admitted only when `updated_at < created_at`; equal-time outcomes are excluded. Test prevalence is 0.0864 on 8,788 rows. These are v2-only results because v2 has additional event-time information.

| Model | Test AP | ROC-AUC | Brier | Recall at 1% FPR | Recall at 5% FPR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rich TF-IDF logistic + causal history | 0.4288 | 0.8694 | 0.0621 | 0.1581 | 0.4045 |
| HistGradientBoosting + causal history | 0.3964 | 0.8629 | 0.0652 | 0.1291 | 0.3347 |

At confidence threshold 0.90, v2 logistic had 78.4% coverage, 2.7% selective error, and 5.9% failure recall. Predictions, per-repository metrics, reliability bins and selective curves are in `analysis/temporal_history_v2_baselines/`.

## Typed-Decision Integration Status

`data/typed/temporal_v1/` contains one derived typed-choice record per frozen v1 row. Each state renders the 15 audited pre-run fields; the source JSONL rows and hashes are unchanged. Model-specific adapters must retain the locked split IDs and the exact test hash in `benchmark_lock.json`; no v2 features and no test-based selection.

Before typed-model training, the derived bridge was regenerated with the canonical `rich_temporal_v1_state` renderer. The prior bridge had the same 15 fields and IDs but placed commit prose before structured metadata; the canonical v1 contract places the 14 concise structured fields first and `commit_message` last. The source v1 split hashes remained unchanged. `scripts/train_laya_temporal_v1_modal.py::submit_sweep` verifies each uploaded train/validation typed row against its original frozen source row; it does not read or upload test examples for tuning.

## Rich-State ModernBERT: Completed Negative Architecture Result

Hypothesis: a pretrained encoder can use the full audited rich temporal-v1 state to improve on the frozen tabular/text baselines. Dataset and split: immutable `temporal-v1-3e9cc5c049399d0b`; tokenizer context 512; full 15-field state; test hash `1410c9dcdc2c87ab503178f3f9749a674bd34959b402ca04ba16d3a0e22af003`. The nine-config sweep selected LR `5e-5`, 3 epochs solely by final validation AP `0.4598572579`. No test metrics were used for tuning, and no further ModernBERT tuning will use this test result.

| Final seed | Test AP | ROC-AUC | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.3344 | 0.8220 | 0.0748 | 0.0526 |
| 42 | 0.3494 | 0.8320 | 0.0697 | 0.0416 |
| 73 | 0.3881 | 0.8480 | 0.0660 | 0.0343 |
| Mean +/- SD | 0.357278 +/- 0.027684 | 0.833982 +/- 0.013143 | 0.070152 +/- 0.004401 | 0.042798 +/- 0.009193 |

The 3-seed mean AP is below frozen rich TF-IDF + structured logistic (0.371658) and structured HistGradientBoosting (0.360971); mean ROC-AUC is 0.833982. All three runs passed recorded finite loss, gradient and parameter checks; max parameter magnitude was about 4.42. Final reports, complete per-example test predictions, per-seed metrics and baseline comparison are in `runs/temporal_v1_rich_modernbert/final/`. Conclusion: the rich input fixes the compact-state underperformance substantially, but ModernBERT does not beat the frozen logistic or HGB baselines on the locked test set. Preserve as a completed negative architecture result; do not retune on this test.

Next experiments remain the already-specified typed-decision comparisons on the identical rich temporal-v1 rows and prediction-time contract. ModernBERT's test result does not change their inputs, splits, tuning protocol or selection criteria.

## Laya Temporal-v1 Protocol

Upstream code was checked out at `9d955671415fc19f069b9cc998928075c1f255ec`; public Laya weights are pinned at Hub revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The project adapter uses the official typed-choice sequence builder and RLCD recipe (proper-scoring-rule reward plus soft cross-entropy), with state=`rich temporal-v1 state` and options=`success` / `failure`. Token budget is 512 including Laya's question and choice markers. The tuning matrix is encoder LR `{1e-5, 2.5e-5, 5e-5}` x epochs `{2, 4}`, fixed head LR `1e-4`, microbatch 8, effective batch 64, seed 2026. The middle-LR/four-epoch point matches the upstream notebook recipe. Select by final validation AP only; test JSONL is not uploaded to sweep jobs. After selection, seeds 17/42/73 are trained once; one scalar temperature per seed is fitted on locked validation NLL, then test is evaluated once. Details and artifacts will be appended after completion.

The initial sweep attempt (`runs/laya_temporal_v1/sweep/failed_attempt.json`) produced no validation scores: fp16 gradients became non-finite on the L4 and the fail-closed health check stopped training; the two 2.5e-5 calls also exposed a run-ID slug validation bug. Remaining calls were cancelled, and the test split was never uploaded. This is an execution failure, not an architecture result. The corrected attempt keeps the RLCD loss, matrix, batch sizes, seed and data fixed, uses L4-supported bf16 mixed precision (no gradient scaler), truncates state tokenization at the already-registered 512-token budget, and normalizes decimal points in run IDs. It runs in parallel as six separate Modal calls. Active app: `ap-4Qx1wdTtgkfKfWaBTau2kM`; artifacts: `runs/laya_temporal_v1/sweep_retry_01/`. Train/validation split hashes and the test hash are unchanged; test rows are physically absent from the tuning volume.

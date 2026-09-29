# Temporal, Customer-Specific Experiment

**Status:** CPU-only exploration complete. No new Modal call, GPU training, or model-family change was run. The source repository-disjoint dataset and prior experiments remain intact.

## Why This Split

The repository-disjoint benchmark tested transfer to unseen repositories. Its train/validation/test failure rates were 8.31% / 20.85% / 4.00%; validation was 99.4% push and 0.2% merge, versus 54.0% push / 23.1% merge in train and 65.1% push / 19.6% merge in test. These differences are confounded with repository identity because each repository belongs to one split.

The temporal benchmark instead sorts runs by creation time separately within each repository: first 70% train, next 15% validation, last 15% test. Eligibility is at least 300 runs total and at least 20 failures in the first 70% training window. This includes 19 of 30 repositories and does not select repositories based on validation/test failure labels. Repository identity is retained as an input because the intended hypothesis is customer-specific adaptation.

## Temporal Dataset and Leakage Checks

Temporal v1 is preserved at data/processed/cicd_temporal. Current v2 is data/processed/cicd_temporal_v2 and adds only commit age/hour/weekday derived from the verified pre-creation committed_date. Both contain 41,191 train runs (10.06% failures), 8,770 validation runs (8.22%), and 8,788 test runs (8.64%).
The closer rates do not prove stationarity, but they are a much better-behaved temporal test than the repository-disjoint split.

The rich raw-data join matches canonical run ID, commit SHA, label, creation time, and the original six-field serialized state. It resolves the merged source CSV's duplicate run IDs and yields one source row per canonical identity. When a commit SHA appeared across temporal partitions, 107 later-partition rows were removed; no identity or commit overlaps remain. Each repository's maximum timestamp in an earlier split is no later than the next split's minimum timestamp. The audit is in the temporal manifest.

Repository inclusion thresholds are intentionally conservative and reported in the manifest. Eleven repositories are excluded for insufficient total volume or insufficient training-window failures. This experiment says nothing about those repositories.

## CPU Baselines

All imputers, encoders, text vocabularies and scalers fit on train only. The table reports current v2; v1 results are retained in analysis/temporal_baselines/. Model settings were fixed; validation was used only to select an optional 90%-precision operating threshold. Test metrics below use threshold 0.5 unless noted.

| Model | Inputs | PR-AUC | ROC-AUC | Precision | Recall | F1 | Recall @ 1% FPR | Recall @ 5% FPR | Brier | ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Majority class | constant negative | 0.0864 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0864 | 0.0864 |
| Train prevalence | constant p=0.1006 | 0.0864 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.0791 | 0.0142 |
| Logistic, repo only | repository identity | 0.2003 | 0.698 | 0.386 | 0.163 | 0.230 | 0.000 | 0.244 | 0.0752 | 0.0254 |
| Logistic, rich without repo | commit text + structured, no repo | 0.3500 | 0.836 | 0.490 | 0.133 | 0.209 | 0.113 | 0.281 | 0.0678 | 0.0235 |
| Logistic, rich with repo | commit text + structured + repo | 0.3719 | 0.851 | 0.479 | 0.136 | 0.211 | 0.113 | 0.311 | 0.0655 | 0.0187 |
| HistGradientBoosting | structured fields + repo, no text | 0.3574 | 0.826 | 0.399 | 0.229 | 0.291 | 0.125 | 0.277 | 0.0711 | 0.0417 |

The random-ranking PR-AUC reference is test prevalence, 0.0864. Repository identity alone scores 0.200, but rich logistic without it scores 0.350; identity is useful but not the whole signal. Adding commit-time features changes logistic PR-AUC only from 0.3717 (v1) to 0.3719 (v2), while boosting moves from 0.3610 to 0.3574. In v2, rich models beat per-repository prevalence PR-AUC in 16/19 (logistic) and 17/19 (boosting); macro median per-repository PR-AUC is 0.255 and 0.237. Per-repository values remain noisy; flutter/flutter's perfect score is based on 90 test runs and 17 failures. See analysis/temporal_baselines_v2/per_repository_metrics.csv.

The strong ranking metrics do not imply a ready decision policy. A validation threshold selected for at least 90% precision yielded 92.6% validation precision / 3.47% recall for rich logistic; on test it achieved 95.5% precision but only 2.77% recall. Boosting achieved 95.8% validation precision / 3.19% recall and 100% test precision / 2.24% recall. High-precision alerting catches only a small fraction of failures. Calibration and false-negative costs need customer-specific study.

The comparison to the previous repository-disjoint experiment is descriptive, not a controlled ablation: cohort, repositories, feature set, and prediction target representation differ. The result supports further testing of within-customer historical signal, not a claim that temporal deployment-risk prediction is solved.

## ModernBERT Export Status

The requested per-example ModernBERT CSV was not produced. Local CPU float32 and bfloat16 inference both returned NaN logits in the first batch. The maximum absolute checkpoint parameter is 8.46e37 (head.dense.weight); float64 produced finite but saturated logits that do not reproduce L4 inference. An earlier L4 rescore showed finite predictions, but its row-level probabilities were not saved. No new Modal inference was started because this phase is intended to avoid further GPU credit until the data formulation is reviewed.

The export script is fail-closed: scripts/export_modernbert_predictions.py records the failure in analysis/modernbert_prediction_export_status.json and does not create a misleading partial CSV. Per-repository ModernBERT metrics therefore remain unavailable. The already-reported aggregate full-run metrics are still in runs/modernbert-l4-20260927T094219Z-56e25afb/test_metrics.json.

## Decision

The temporal + richer-feature CPU results show meaningful signal worth validating. Do not treat them as proof of a production predictor: this is one public dataset, one temporal cutoff, a small set of repositories, and an outcome limited to CI success/failure. The eventual continue/canary/rollback/human-review system also needs deployment, incident, rollback, service context and production telemetry.

No Laya, CLM, Jev, or new ModernBERT training has been run. Before considering a new GPU experiment, review the feature-timing caveats in docs/FEATURE_AUDIT.md and the checkpoint instability. If a later ModernBERT run is approved, it should use the richer temporal state, enforce finite/logit/parameter-magnitude guards, and retain the CPU baselines as the benchmark.

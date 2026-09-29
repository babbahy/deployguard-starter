# DeployGuard

Predict CI workflow success/failure from information available when a run
is created. The first experiment uses real public data and ModernBERT.
Deployment actions and other model families remain later work.

## Local setup (WSL)

The repo is already on the Linux filesystem. WSL has system Python 3.10;
the project uses an isolated Python 3.11 environment created with `uv`.
`uv` installs Python/packages; editable installation makes changes in `src/`
immediately available to Python without reinstalling the project.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

For local CPU training, install CPU PyTorch first to avoid CUDA downloads:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e '.[train,dev]'
```

`requirements-wsl-cpu.txt` records the exact installed versions. To reproduce
them in a fresh environment, install its pinned CPU torch version from the
PyTorch CPU index, then run `uv pip install -r requirements-wsl-cpu.txt`
and `uv pip install -e . --no-deps`.

## Canonical dataset

Source: [Mendeley v1](https://data.mendeley.com/datasets/mggwn7rj9f/1),
Faruna and Sanusi (2026), CC BY 4.0. The downloaded CSV has 303,079 records
and mixes providers and schemas. See [the audit](docs/DATA_AUDIT.md) for
the actual columns, exclusions, leakage checks and limitations.

We use first-attempt GitHub Actions `push`/`pull_request` runs with explicit
success/failure outcomes. The verified eligible cohort has 68,354 runs from
30 repositories. This is narrower than the original dataset.

```bash
python scripts/download_cicd.py
python -m deployguard.data.prepare_cicd \
  --input data/raw/final_research_dataset_MASTER.csv \
  --output data/processed/cicd_smoke --per-repo 32
python scripts/inspect_dataset.py data/processed/cicd_smoke/train.jsonl
python -m pytest -q
```

The manifest includes hashes, filters, class counts, repository membership
and leakage audits. Sampling occurs after repository-disjoint splitting.
Omit `--per-repo` and use a different output directory for the full cohort.
There is no random-row fallback. Raw data, processed data and models are
ignored by Git; this downloaded folder currently has no Git history.

The full canonical cohort is processed to `data/processed/cicd_full`. The
logistic-regression baseline and full-cohort ModernBERT run have both been
completed on the same repository-disjoint splits. Reproduce the baseline:

```bash
python scripts/train_tabular_baseline.py \
  --data-dir data/processed/cicd_full --output-dir runs/tabular_full
```

This fits imputers, log transforms, scaling and categorical encoding on the
training split only. It selects a 90% precision threshold on validation,
then evaluates the test split once. Test results at threshold 0.5 are also
reported. The baseline test average precision was 0.036 (prevalence 0.040;
ROC-AUC 0.453). ModernBERT trained for three passes on one Modal L4 scored
test average precision 0.032 and ROC-AUC 0.424; at threshold 0.5 it found 1
of 595 failures. Neither model beats the random ranking reference here. The
held-out repositories have sharply different failure rates, so these are
exploratory, repository-limited results, not a deployable predictor. See
[the Modal experiment record](docs/MODAL.md), [baseline metrics](runs/tabular_full/metrics.json),
and [ModernBERT metrics](runs/modernbert-l4-20260927T094219Z-56e25afb/test_metrics.json).


## Temporal Customer-Specific Experiment

The repository-disjoint result was negative, so the next benchmark asks whether
richer pre-run context predicts later failures within repositories represented
in training. The current dataset is data/processed/cicd_temporal_v2; v1 is
preserved at data/processed/cicd_temporal for comparison. Neither overwrites cicd_full.
Results and design limits are recorded in
[the temporal report](analysis/temporal_experiment.md), [feature audit](docs/FEATURE_AUDIT.md),
and [experiment history](docs/EXPERIMENTS.md).

Run the CPU-only analysis with scripts/analyze_repository_shift.py,
python -m deployguard.data.temporal, and scripts/evaluate_temporal_baselines.py.
The full run is not a universal cross-repository predictor: it tests later
runs from 19 repositories also represented in the historical training period.

On its 8,788-run test period, rich TF-IDF logistic reached PR-AUC 0.372 /
ROC-AUC 0.851 and structured histogram boosting reached 0.357 / 0.826,
against failure prevalence 0.086. A repo-only model scored PR-AUC 0.200;
rich logistic without repository identity still scored 0.350. This is
encouraging ranking signal, not a reliable high-recall alert policy.

The prior ModernBERT checkpoint cannot currently be scored locally: CPU
float32 and bfloat16 produce NaN logits. The inference exporter fails closed
and records this in analysis/modernbert_prediction_export_status.json.
No new Modal inference/training has been run during this analysis.

## First real fine-tune

```bash
python scripts/train_modernbert.py \
  --data-dir data/processed/cicd_smoke --output-dir runs/modernbert_smoke \
  --cpu --threads 2 --batch-size 1 --max-length 128 --max-steps 2 \
  --revision 8949b909ec900327062f0ebf497f51aef5e6f0c8
```

This downloads the pretrained `answerdotai/ModernBERT-base` checkpoint
(roughly 600 MB), initializes a two-class head and updates all parameters.
No Hugging Face account is required for this public checkpoint.

In PyTorch terms, `transformers` provides the encoder module and tokenizer;
`datasets` loads/maps JSONL; the padding collator builds batches; `Trainer`
runs forward, cross-entropy, backward and AdamW updates. `accelerate` handles
device placement. Only `state` becomes tokens; label and audit metadata do
not become model inputs. A newly initialized classifier warning is expected.

The smoke run saves `best_model/`, `run_info.json` and `test_metrics.json`.
The first run completed successfully; see [the run report](docs/SMOKE_RUN.md).
Two updates verify the machinery, not predictive skill. The small test set
cannot support claims about low false-positive rates or calibration.
Use the same saved tokenizer with a trained checkpoint for inference.

The temporal customer-specific CPU experiment and its baselines are now complete.
The ranking signal is promising, but high-precision thresholds catch few
failures and the data lacks incidents, rollback outcomes and production
telemetry. No further GPU work should start before reviewing docs/EXPERIMENTS.md.

## Modal status

Two historical L4 runs are documented in docs/MODAL.md. Do not rerun the
100-step command or start another GPU job yet. The full-cohort ModernBERT
checkpoint underperformed the CPU baseline and has non-finite CPU logits.
The temporal CPU results now justify reviewing a richer ModernBERT proposal,
but GPU inference/training remains paused pending that review and stability
guards.

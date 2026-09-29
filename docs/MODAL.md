# ModernBERT on Modal

The local CPU run established that the canonical data and training loop work.
Training can now move to a remote GPU. Your PC uploads files and streams logs;
it does not need a GPU or enough RAM to hold the model.

## Accounts

Hugging Face hosts the public pretrained checkpoint and provides the Python
libraries. `answerdotai/ModernBERT-base` is public and ungated, so downloading
it needs no account or token. Gated/private models and uploads need authentication.

Modal provides the remote machine and requires its own account and token.
The client is installed in this project's `.venv`. Authenticate in WSL:

```bash
source .venv/bin/activate
modal setup
modal token info
```

Complete the browser flow. Keep credentials out of source files and chat.
Modal's GPU documentation currently requires a payment method on file,
even though the Starter plan includes $30/month of free compute. Check your
workspace's Billing page for the actual remaining credit.

## Submit a bounded pilot

```bash
modal run scripts/train_modal.py --max-steps 100 --batch-size 8
```

This uses the existing audited 960-example subset. No preprocessing or
training runs on the laptop. To select another already-processed dataset,
pass `--data-dir data/processed/YOUR_DATASET`.

The launcher:

1. Builds a Python 3.11 container with pinned CUDA PyTorch and Hugging Face
   libraries. This is a separate Python environment from your local `.venv`.
2. Verifies split hashes against the audit manifest and uploads just the
   manifest and three JSONL files, plus project source needed for training.
3. Requests one NVIDIA L4 with two CPU cores and 16 GiB host RAM.
4. Calls the same `train_modernbert.py` with CUDA required, bfloat16 when
   supported, max length 128, and the pinned pretrained model revision.
5. Writes checkpoints and the Hugging Face cache to the persistent
   `deployguard-training` Volume. A Volume is cloud storage that survives
   the temporary training container.
6. Returns metrics to a unique local `runs/modernbert-l4-.../` directory.

Limits: 1-10,000 updates, batch size 1-16, 90-minute function timeout,
about 87 minutes for the trainer subprocess, one container, no configured retries,
no scheduled jobs. Modal infrastructure preemption/recovery may still occur.
The invocation uses `modal run`; there is no persistent deployed service.
The first image build and model download can take several minutes.

L4 GPU time is currently $0.000222/second (about $0.80/hour), with CPU and
memory billed separately. The function timeout is a ceiling, not a target
runtime or a billing cap. Check the Modal dashboard before and after long
runs; this estimate is not a claim about your remaining credit.

## Results and checkpoint retrieval

Local `submission.json` records the run ID even if the job fails. Successful
runs return `run_info.json` and `test_metrics.json`; the run report includes
the GPU name, peak GPU memory, precision, package versions and dataset hashes.
The checkpoint stays on Modal until explicitly downloaded:

```bash
modal volume get deployguard-training /runs/RUN_ID/best_model runs/RUN_ID/best_model
```

The launcher prints this command with the actual run ID. Keep the terminal
open while the pilot runs. A failed run raises an error; check the Modal logs
and the Volume for any completed checkpoints before resubmitting.

Check a stored checkpoint on an L4 against the same audited test set:

```bash
modal run scripts/train_modal.py --verify-run RUN_ID
```

This reports whether all probabilities are finite, the probability range,
and rescored metrics. Full precision CPU inference on the first 100-step
checkpoint produced non-finite logits, while L4 bfloat16 inference produced
finite probabilities. Keep this pilot on the GPU path until CPU inference is
validated; its zero failure recall at threshold 0.5 makes it unsuitable as
a detector either way.

The 100-update job is a GPU pilot, not the full research benchmark. It uses
the small audited subset; the test results are descriptive, not tuning input.
The first pilot achieved no failure recall at threshold 0.5 and its test
average precision (0.055) was slightly below the 0.056 test prevalence.
Its full-precision CPU reload produced non-finite logits; the same checkpoint
produced finite probabilities on the L4 in bfloat16. That pilot is not a
usable detector.

## Full-cohort run

The canonical cohort and repository-disjoint splits are in
`data/processed/cicd_full`. A logistic-regression baseline on the six
pre-outcome features scored test average precision 0.036, below the 0.040
test prevalence (random ranking reference); test ROC-AUC was 0.453. A
validation threshold selected for 90% precision caught no test failures.
The held-out repository failure rates differ sharply (train 8.3%, validation
20.9%, test 4.0%), so report ranking metrics and repository limits rather
than treating accuracy or a transferred threshold as evidence of utility.

Launch approximately three passes over the full training split (47,263
examples, batch size 16; 8,850 optimizer updates):

```bash
modal run scripts/train_modal.py \
  --data-dir data/processed/cicd_full \
  --max-steps 8850 --batch-size 16 --learning-rate 2e-5
```

This uploads the full audited split files and runs the same pinned model and
trainer on one L4. It may take tens of minutes. The function timeout is a
hard upper bound, not a billing estimate. The full test split is evaluated
once after training; do not tune against it. A run that exceeds the timeout
may leave intermediate artifacts on the persistent Volume, but the launcher
does not automatically resume from them.


### Completed Run

Run `modernbert-l4-20260927T094219Z-56e25afb` completed 8,850 updates
(2.996 epochs) with batch size 16 and learning rate 2e-5. Training took
1,186 seconds on an NVIDIA L4 in bfloat16. On the untouched test set, average
precision was 0.0324 versus 0.0400 prevalence, and ROC-AUC was 0.4242. At
threshold 0.5 it predicted 87 failures and caught 1 of 595 (recall 0.00168);
recall at 5% false-positive rate was 0.0101. These results are worse than
the same-split logistic baseline (average precision 0.0361, ROC-AUC 0.4526),
so this checkpoint is not a useful failure detector.

An independent L4 reload produced finite probabilities and reproduced test
metrics to rounding precision. The checkpoint, tokenizer and reports are
## References

- https://huggingface.co/docs/hub/models-adding-libraries
- https://huggingface.co/docs/hub/models-gated
- https://modal.com/docs/guide/getting-started
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/volumes
- https://modal.com/pricing

Checked 2026-09-27. Remote execution still requires a successful account sign-in.

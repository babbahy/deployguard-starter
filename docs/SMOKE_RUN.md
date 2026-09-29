# First ModernBERT smoke run

Completed 2026-09-27 on WSL2 CPU, Python 3.11.16. Exact dependency versions
are in `requirements-wsl-cpu.txt`; commands are in the README.

- Data: `data/processed/cicd_smoke`, checksum-verified Mendeley v1 cohort.
- Split sizes: 672 train, 128 validation, 160 test; repository-disjoint.
- Model: `answerdotai/ModernBERT-base`.
- Revision: `8949b909ec900327062f0ebf497f51aef5e6f0c8`.
- Two AdamW updates, batch size 1, max length 128, two CPU threads.
- 149,606,402 trainable parameters; no encoder freezing.
- Finite training losses: 0.1592 and 0.0107.
- Training/validation/checkpoint phase: 22.08 seconds.
- Training through test evaluation and final save: 38.02 seconds,
  excluding initial model download/loading and tokenization.
- Classifier maximum absolute weight change: 0.0000300258.
- Independently compared saved encoder tensor `model.layers.0.attn.Wo.weight`
  against the original checkpoint: maximum absolute change 0.0000301003.
- Reloaded saved model and tokenizer offline; output probabilities were finite
  and summed to one.
- Four regression tests passed; dependency compatibility check passed.

Artifacts: `runs/modernbert_smoke/best_model/`, `run_info.json`,
`test_metrics.json`, and `checkpoint-2/` (includes optimizer state).

Test average precision was 0.05698, ROC-AUC 0.49558, accuracy 0.94375,
failure recall 0.0 and Brier score 0.05599. The model predicted success for
all 160 test examples at threshold 0.5. Only two training examples were
used for optimizer updates. These results validate execution and serialization,
not useful discrimination, calibration or generalization. Do not tune on this
test result. Next work is a tabular baseline and a properly trained experiment
with validation-driven choices and stronger held-out evaluation.

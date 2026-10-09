# Zehra Özcan

- Student ID: 2309011051
- GitHub: @zehraoz36

## Summary

I added the "new" method of the project: a fine-tuned DeBERTa-v3 model that is
compared with the baselines on the same data, splits and metrics. I also set up
the repository.

## What I did

- Set up the GitHub repository and added the LICENSE and requirements files.
- Added `train_transformer.py`, which fine-tunes `microsoft/deberta-v3-small`
  for bug / non-bug classification. It reads the cleaned data and the two fixed
  splits produced by `preprocess.py` and reuses the metric function of
  `benchmark.py`, so its scores are directly comparable with the baselines.
- Ran the training on Google Colab (Tesla T4 GPU): 2 epochs, maximum 256 tokens,
  AdamW with learning rate 2e-5, batch size 16, mixed precision, seed 42.
- Tested the script first on a small subset (`--limit 400`) and fixed an error
  found there: the published checkpoint stores half-precision weights, so the
  model has to be cast to float32 before training with a gradient scaler.
- Added the result files and reported the scores to the team for the README table.

The script was developed with the help of an AI assistant (Claude). I ran,
tested and debugged it and produced the results myself.

## Results

| Experiment | Model | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Random split | DeBERTa-v3-small | 90.55 | 87.58 |
| Random split | Logistic Regression (best baseline) | 88.08 | 83.52 |
| Cross-project (SciPy) | DeBERTa-v3-small | 90.77 | 89.37 |
| Cross-project (SciPy) | Logistic Regression (best baseline) | 86.89 | 84.07 |

Limitations: one run with a single seed, and each report is truncated to its
first 256 tokens.

## Files

- `train_transformer.py`
- `results/transformer_benchmark.csv`
- `results/transformer_metrics.json`
- `LICENSE`, `requirements.txt`

# Zehra Özcan

- Student ID: 2309011051
- GitHub: [@zehraoz36](https://github.com/zehraoz36)

## Summary

My contribution has three parts: I set up the shared repository, I added the
project's new method (a fine-tuned DeBERTa-v3 model, trained and evaluated on
exactly the same data, splits and metric function as the baselines), and I
tested and debugged the training script before the full run.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Set up the public repository with the license and requirements files | Initial commit, commit `28de307` |
| 2 | Added the DeBERTa-v3 fine-tuning script | Commit `28de307`, `train_transformer.py` |
| 3 | Found and fixed a half-precision loading error with a small test run | Section 2 below, `train_transformer.py` lines 73-75 |
| 4 | Ran the full training on Google Colab and added the results | `results/transformer_benchmark.csv`, `results/transformer_metrics.json` |

## 1. The new method

The course rules treat DeBERTa as a new model, while the classical models and
vanilla BERT are baselines. `train_transformer.py` fine-tunes
`microsoft/deberta-v3-small` with a two-class classification head.

To keep the comparison controlled, the script reads `data/cleaned.jsonl` and the
two fixed splits produced by `preprocess.py`, and imports `SEED` and the
`metrics` function from `benchmark.py`. Only the model differs from the baselines.

| Setting | Value |
|---|---|
| Model | `microsoft/deberta-v3-small` (6 transformer layers) |
| Input | First 256 tokens of each report, padding per batch |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | Linear warm-up over 6% of the steps, then linear decay |
| Batch size | 16 for training, 64 for evaluation |
| Epochs | 2, checkpoint selected by validation Macro-F1 |
| Other | Mixed precision, gradient clipping at 1.0, seed 42 |

## 2. Testing and the error I fixed

Before the full run I tested the script on 400 records per split
(`--limit 400`), which takes about one minute and saves nothing.

The first test stopped with `ValueError: Attempting to unscale FP16 gradients`.
The cause: the published checkpoint stores its weights in half precision and the
library loaded them in that type, but a gradient scaler needs float32 parameters.
The fix is to cast the model to float32 after loading it (`model.float()`).
After the fix the test ran through both scenarios.

During the full run I checked the first training-loss values (0.54, 0.42, 0.38
after 200, 400 and 600 steps) to confirm that the model was learning before
letting it finish.

## 3. Training run

Google Colab, Tesla T4, Python 3.13.15, torch 2.11.0+cu130, transformers 5.18.0.

| Scenario | Train / val / test | Epoch 1 loss / val Macro-F1 | Epoch 2 loss / val Macro-F1 | Selected epoch | Time |
|---|---|---|---|---:|---:|
| Random | 24,229 / 3,462 / 6,923 | 0.3178 / 0.8724 | 0.2279 / 0.8853 | 2 | 780 s |
| Cross-project | 27,461 / 2,583 / 4,570 | 0.3040 / 0.8110 | 0.2171 / 0.8272 | 2 | 861 s |

## 4. Test results

The test set was evaluated once, with the selected checkpoint.

| Scenario | Accuracy (%) | Macro-F1 (%) | Bug F1 (%) |
|---|---:|---:|---:|
| Random | 90.55 | 87.58 | 93.66 |
| Cross-project (SciPy) | 90.77 | 89.37 | 93.22 |

Per-class results (from `results/transformer_metrics.json`):

| Scenario | Class | Precision | Recall | F1 | Confusion counts |
|---|---|---:|---:|---:|---|
| Random | non-bug | 0.853 | 0.781 | 0.815 | TN 1,441, FP 405 |
| Random | bug | 0.923 | 0.951 | 0.937 | FN 249, TP 4,828 |
| Cross-project | non-bug | 0.877 | 0.834 | 0.855 | TN 1,246, FP 248 |
| Cross-project | bug | 0.921 | 0.943 | 0.932 | FN 174, TP 2,902 |

In both scenarios the model is weaker on the minority class: non-bug reports are
labelled as bugs more often than the reverse.

## 5. Comparison with the other methods

All numbers are from the repository.

| Scenario | Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| Random | Logistic Regression (best classical baseline) | 88.08 | 83.52 |
| Random | BERT `bert-base-uncased` (Tuana's run) | 90.52 | 87.81 |
| Random | DeBERTa-v3-small (mine) | 90.55 | 87.58 |
| Cross-project | Logistic Regression (best classical baseline) | 86.89 | 84.07 |
| Cross-project | BERT `bert-base-uncased` (Tuana's run) | 90.02 | 88.26 |
| Cross-project | DeBERTa-v3-small (mine) | 90.77 | 89.37 |

DeBERTa beats the best classical baseline by 4.1 Macro-F1 points on the random
split and 5.3 on the cross-project split. Against BERT it is about equal on the
random split and 1.1 points better on the cross-project split, using a model
with half as many transformer layers (6 instead of 12). These are single runs,
so differences of a few tenths of a point should not be over-interpreted.

## Limitations

One training seed, inputs truncated to 256 tokens, and the cross-project test
covers SciPy only.

## Use of AI assistance

The script was developed with the help of an AI assistant (Claude). I ran,
tested and debugged it and produced the results myself.

## Files

- `train_transformer.py`
- `results/transformer_benchmark.csv`
- `results/transformer_metrics.json`
- `LICENSE`, `requirements.txt` (initial versions)

## Commit evidence

- [Add the fine-tuning script, results, license and requirements](https://github.com/zehraoz36/tokens-project-1/commit/28de307)
- [Add contribution details](https://github.com/zehraoz36/tokens-project-1/commit/0ddfb52)

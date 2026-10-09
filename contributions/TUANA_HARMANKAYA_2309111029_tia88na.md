# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)

## Summary

My contribution has five parts: I integrated the team's dataset and classical benchmark package into the shared repository, added a missing dependency, corrected the recorded hashes of the raw files after a line-ending change, ran the pipeline on fresh clones to check that it reproduces, and added the vanilla BERT baseline (`bert-base-uncased`), trained and evaluated on exactly the same data, splits and metric function as every other method.

## Overview of my contributions

| # | Contribution | Evidence |
|---|---|---|
| 1 | Integrated the dataset and classical benchmark package, with a compatibility audit against the existing transformer script | Commit `d890186` (12 files), audit table below |
| 2 | Added the missing dependency (`transformers`) | Commit `4662eff`, `requirements.txt` line 7 |
| 3 | Corrected the recorded hashes of the raw CSV files after Git converted their line endings (CRLF to LF) | Commit `b52d376`, per-file table below |
| 4 | Verified reproducibility from a fresh clone (data pipeline, classical benchmark, transformer path) | Verification tables below |
| 5 | Implemented, trained and evaluated the vanilla BERT baseline, with a verifiable prediction export | Commit `6489cef`, `train_bert.py`, `results/bert_*` |

## 1. Integration and compatibility audit

`train_transformer.py` imports `benchmark.py` and reads `data/cleaned.jsonl`, so it could not run in the repository without the team package. I added it (commit `d890186`): `data/raw/` (five CSV files), `data/cleaned.jsonl`, `data/LICENSE`, `preprocess.py`, `benchmark.py`, `results/benchmark.csv`, `results/metrics.json` and `results/data_audit.json`.

Before merging I checked that the package matches what the existing transformer script and its recorded results expect:

| Check | Result |
|---|---|
| `cleaned.jsonl` contains the fields the transformer script reads (`text`, `label`, `random_split`, `cross_project_split`) | Yes (it also has `record_id`, `project`, `text_sha256`) |
| `benchmark.py` exposes `SEED` and `metrics`, which the transformer script imports | Yes |
| Random split sizes (train / val / test) | 24,229 / 3,462 / 6,923, as in `data_audit.json` |
| Cross-project split sizes (train / val / test) | 27,461 / 2,583 / 4,570, as in `data_audit.json` |
| Best classical baseline (Logistic Regression) agrees with the numbers already reported by the team | Yes: 88.08 / 83.52 (random) and 86.89 / 84.07 (cross-project), accuracy / Macro-F1 |

## 2. Dependency fix

The transformer script needs the `transformers` library, but `requirements.txt` did not list it, so a fresh install could not run the script. I added `transformers==5.18.0` (commit `4662eff`), the version recorded in `results/transformer_metrics.json`. A fresh virtual environment on Windows (Python 3.12.9) resolved and installed every pinned package, and the same file installs on Colab (Linux, Python 3.13).

## 3. Raw-file hash mismatch: cause and fix

Regenerating `results/data_audit.json` from the repository gave SHA-256 hashes for all five raw CSV files that differed from the ones the team package recorded. Comparing the files in the repository with the originals gives:

| Raw file | Original size (bytes) | Size in repository | Difference | CRLF line breaks in original |
|---|---:|---:|---:|---:|
| numpy | 6,089,986 | 6,037,734 | 52,252 | 52,252 |
| pandas | 11,810,897 | 11,598,680 | 212,217 | 212,217 |
| salt | 27,060,844 | 26,780,708 | 280,136 | 280,136 |
| scipy | 6,932,196 | 6,874,336 | 57,860 | 57,860 |
| weblate | 3,874,066 | 3,814,669 | 59,397 | 59,397 |

In every file the size difference equals the number of CRLF line breaks, and after converting CRLF to LF the originals are byte-identical to the repository copies. The cause is Git converting line endings of the raw CSV files when they were committed on Windows. The content is unchanged and `cleaned.jsonl` is identical in both versions, so only the recorded hashes were wrong. I updated the five hashes in `results/data_audit.json` and documented the conversion in `data/LICENSE` (commit `b52d376`).

## 4. Reproducibility verification

The integrated repository was checked in fresh clones:

| Check | Environment | Result |
|---|---|---|
| `preprocess.py` regenerates `data/cleaned.jsonl` | Fresh clone of the GitHub repository | Same SHA-256 as the committed file |
| `preprocess.py` regenerates `results/data_audit.json` | Same | Same SHA-256 as the committed file |
| `preprocess.py` | Google Colab, fresh clone | 34,614 clean records and identical split counts |
| `benchmark.py` | Google Colab (Linux, Python 3.13, scikit-learn 1.9.1) | All 10 results identical to `results/benchmark.csv` at 4 decimals (table below) |
| `train_transformer.py --limit 200` | Google Colab, CPU | Loading, tokenization, training and evaluation ran through for the first scenario; I stopped the run during the second scenario, because on CPU it is slow |

Reproduced classical benchmark (committed value equals Colab value in all rows):

| Scenario | Method | Accuracy | Macro-F1 |
|---|---|---:|---:|
| Random | Majority | 0.7334 | 0.4231 |
| Random | Count_NB | 0.6383 | 0.6284 |
| Random | TFIDF_LinearSVM | 0.8746 | 0.8309 |
| Random | TFIDF_TorchLR | 0.8376 | 0.7526 |
| Random | TFIDF_SklearnLR | 0.8808 | 0.8352 |
| Cross-project | Majority | 0.6731 | 0.4023 |
| Cross-project | Count_NB | 0.5438 | 0.5399 |
| Cross-project | TFIDF_LinearSVM | 0.8606 | 0.8350 |
| Cross-project | TFIDF_TorchLR | 0.8004 | 0.7269 |
| Cross-project | TFIDF_SklearnLR | 0.8689 | 0.8407 |

Environment findings:

- On Colab, the preinstalled `torchvision` is incompatible with the `torch` version pinned in `requirements.txt` and crashes the `transformers` import. Uninstalling `torchvision` (the project does not use it) fixes it. This needs a note in the README.
- On my Windows machine, an operating-system application-control policy blocked a `scipy` DLL, so the scientific stack could not run locally. This is a property of that machine, not of the repository, which is why I verified on Colab.

## 5. Vanilla BERT baseline

**Why.** The course rules treat methods presented in their vanilla form as baselines (BERT is the example) and DeBERTa as a new model. The repository had only classical baselines, so a vanilla BERT baseline was missing.

**Implementation.** `train_bert.py` (195 lines) is adapted from `train_transformer.py`. It differs from that script in 6 removed and 26 added lines, and the differences are:

| Change | Purpose |
|---|---|
| Default model is `bert-base-uncased` | The baseline model |
| Output files are `results/bert_benchmark.csv` and `results/bert_metrics.json` | Do not overwrite the DeBERTa results |
| New export of test predictions (`results/bert_predictions_<scenario>.csv`) with columns `text_sha256`, `true_label`, `predicted_label` | Error analysis without republishing issue texts; skipped when `--limit` is used, so smoke tests cannot overwrite published results |
| Updated docstring | Describes the script as a baseline |

Everything else is shared with the DeBERTa script, so the comparison is controlled: the same `cleaned.jsonl` and fixed splits, the same `metrics` function from `benchmark.py`, seed 42 from `benchmark.SEED`, truncation to 256 tokens with padding per batch, AdamW (weight decay 0.01) with a linear warm-up schedule, batch size 16 for training and 64 for evaluation, and checkpoint selection by validation Macro-F1.

**Run.** I ran a smoke test first (`--limit 200 --epochs 1`), then the full experiments on Google Colab (Tesla T4, mixed precision, torch 2.11.0+cu130, transformers 5.18.0).

| Scenario | Train / val / test | Epoch 1 loss / val Macro-F1 | Epoch 2 loss / val Macro-F1 | Selected epoch | Time |
|---|---|---|---|---:|---:|
| Random | 24,229 / 3,462 / 6,923 | 0.3046 / 0.8882 | 0.2045 / 0.8876 | 1 | 718 s |
| Cross-project | 27,461 / 2,583 / 4,570 | 0.2858 / 0.8257 | 0.1932 / 0.8222 | 1 | 791 s |

**Test results** (test set evaluated once, with the selected checkpoint):

| Scenario | Accuracy (%) | Macro-F1 (%) | Bug F1 (%) |
|---|---:|---:|---:|
| Random | 90.52 | 87.81 | 93.56 |
| Cross-project (SciPy) | 90.02 | 88.26 | 92.81 |

**Integrity of the prediction export.** The exported files agree with the data and the recorded results:

| Check | Random | Cross-project |
|---|---|---|
| Rows equal the number of test records in `cleaned.jsonl` | 6,923 | 4,570 |
| SHA-256 of every test text, in order, equals `text_sha256` in the export | Yes | Yes |
| True labels equal the labels in `cleaned.jsonl` | Yes | Yes |
| Accuracy and Macro-F1 recomputed from the export equal the recorded values | Yes | Yes |

**Per-class results** (computed from the exported prediction files):

| Scenario | Class | Precision | Recall | F1 | Confusion counts |
|---|---|---:|---:|---:|---|
| Random | non-bug | 0.829 | 0.812 | 0.820 | TN 1,499, FP 347 |
| Random | bug | 0.932 | 0.939 | 0.936 | FN 309, TP 4,768 |
| Cross-project | non-bug | 0.898 | 0.784 | 0.837 | TN 1,171, FP 323 |
| Cross-project | bug | 0.901 | 0.957 | 0.928 | FN 133, TP 2,943 |

Per-project accuracy on the random test split (computed from the same files): numpy 92.1% (n = 1,231), pandas 91.4% (1,759), salt 90.5% (2,484), scipy 90.8% (934), weblate 83.3% (515). Weblate is the weakest project.

**Comparison with the other methods** (all numbers from the repository):

| Scenario | Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| Random | Logistic Regression (best classical baseline) | 88.08 | 83.52 |
| Random | BERT `bert-base-uncased` (mine) | 90.52 | 87.81 |
| Random | DeBERTa-v3-small (Zehra's run) | 90.55 | 87.58 |
| Cross-project | Logistic Regression (best classical baseline) | 86.89 | 84.07 |
| Cross-project | BERT `bert-base-uncased` (mine) | 90.02 | 88.26 |
| Cross-project | DeBERTa-v3-small (Zehra's run) | 90.77 | 89.37 |

BERT beats the best classical baseline by about 4.3 Macro-F1 points on the random split and 4.2 on the cross-project split. DeBERTa is roughly equal on the random split and about 1.1 points better on the cross-project split. These are single runs, so differences of a few tenths of a point should not be over-interpreted.

**Limitations.** One training seed, inputs truncated to 256 tokens, and the cross-project test covers SciPy only.

## Files I integrated or modified

- Team-provided package (integrated, not authored by me): `data/raw/`, `data/cleaned.jsonl`, `preprocess.py`, `benchmark.py`, `results/benchmark.csv`, `results/metrics.json` and `results/data_audit.json`.
- Dependency fix: `requirements.txt`.
- Raw-file hash correction: `results/data_audit.json` and the line-ending note in `data/LICENSE`.
- BERT baseline: `train_bert.py`.
- BERT results: `results/bert_benchmark.csv` and `results/bert_metrics.json`.
- BERT test predictions: `results/bert_predictions_random.csv` and `results/bert_predictions_cross_project.csv`.

## Commit evidence

- [Integrate the datasets, preprocessing and benchmark package](https://github.com/zehraoz36/tokens-project-1/commit/d890186b5f922f1264b52cbccec337807720f729)
- [Add the Transformers dependency](https://github.com/zehraoz36/tokens-project-1/commit/4662effce6e55397a581075dd61c5b1aa65933ea)
- [Update raw-file audit hashes and the line-ending note](https://github.com/zehraoz36/tokens-project-1/commit/b52d3769e6bc7bbab2e0cf3ce3f2ddcb35a8e776)
- [Add the BERT training script and evaluation results](https://github.com/zehraoz36/tokens-project-1/commit/6489cef191e7950de072d1986a06d5fe77fcacde)

# Ece Mina Örenler

- Student ID: 220901539
- GitHub: [@eceorenler](https://github.com/eceorenler)

## Summary

I prepared the initial BugHub dataset, preprocessing and classical-baseline pilot
and shared it with the team as `BugHub_Pilot.zip`. This package
became the data and baseline foundation for the later neural and transformer
experiments. Tuana subsequently integrated the package into the shared repository.

## Overview of my contributions

| Contribution | Files / evidence |
|---|---|
| Researched the dataset and its license, and prepared a five-project subset for bug / non-bug classification | `data/raw/`, `data/LICENSE`; source: [Zenodo 7377402](https://doi.org/10.5281/zenodo.7377402) |
| Prepared the reproducible preprocessing pipeline and documented the cleaning decisions | `preprocess.py`, `data/cleaned.jsonl`, `results/data_audit.json` |
| Established the fixed random and cross-project evaluation splits and the shared metrics function | `preprocess.py`, `benchmark.py` |
| Prepared and ran the initial classical baselines, including logistic regression implemented in PyTorch | `benchmark.py`, `results/benchmark.csv`, `results/metrics.json` |
| Verified the consistency of the cleaned data, evaluation splits and baseline results before sharing the pilot with the team | `results/data_audit.json`, `results/benchmark.csv`, `results/metrics.json` |

## Data preparation and evaluation

The pilot uses NumPy, Pandas, Salt, SciPy and Weblate: 34,949 records in total.
The original authors and issue-tracker participants supplied the labels; I did
not collect or manually label these reports.

The preprocessing removes explicit title tags such as `BUG:` and `[ENH]`, cleans
HTML/Markdown formatting, normalizes URLs, Unicode, case and whitespace, and
limits unusually long text to 20,000 characters. Code-block contents and negation
words are retained. It removes 2 empty texts, 33 duplicate records and 300 rows
whose identical cleaned text has conflicting labels, leaving **34,614 records**.

I prepared two evaluation scenarios with seed 42:

- **Random:** stratified 70% training, 10% validation and 20% test.
- **Cross-project:** train on NumPy/Pandas/Salt, validate on Weblate, test on SciPy.

The vectorizers are fitted only on training data. Exact text duplicates are
removed before splitting. Macro-F1 is the primary metric because the classes
are imbalanced; accuracy, bug-class F1 and per-class metrics are also reported.

## Initial baseline results

These are the original pilot results, before the team's later feature-engineering
and neural-model additions. Values are percentages.

| Model | Random accuracy | Random Macro-F1 | SciPy accuracy | SciPy Macro-F1 |
|---|---:|---:|---:|---:|
| Majority reference | 73.34 | 42.31 | 67.31 | 40.23 |
| Multinomial Naive Bayes | 63.83 | 62.84 | 54.38 | 53.99 |
| Linear SVM | 87.46 | 83.09 | 86.06 | 83.50 |
| Logistic regression, PyTorch | 83.76 | 75.26 | 80.04 | 72.69 |
| Logistic regression, scikit-learn | 88.08 | 83.52 | 86.89 | 84.07 |

The PyTorch logistic regression uses one linear layer, binary cross-entropy loss
and minibatch SGD. The SVM baseline uses LinearSVC for efficient training on sparse
TF-IDF features.

## ZIP handoff and repository attribution

My initial contribution was shared as a ZIP before it was committed to this
repository. The import commit was made by **Tuana**, not by my GitHub account:

- [d890186 — Add datasets, preprocessing and benchmark scripts](https://github.com/zehraoz36/tokens-project-1/commit/d890186b5f922f1264b52cbccec337807720f729)
- [Tuana's contribution statement](TUANA_HARMANKAYA_2309111029_tia88na.md), section "Integration and compatibility audit".

Comparison with the retained pilot package confirms that `preprocess.py` and
`benchmark.py` have the same source content, and `data/cleaned.jsonl` is
byte-identical. The raw CSVs differ only in line endings, as documented by Tuana.
This distinguishes preparation of the initial package from its repository
integration. I do not claim authorship of Tuana's import commit or of teammates'
BERT, DeBERTa, GloVe neural baselines, extra experiments or paper-protocol scripts.

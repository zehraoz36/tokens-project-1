# Begüm Ziyade

- Student ID: 220901561
- GitHub: <!-- TODO: [@username](https://github.com/username) -->

## Summary

I completed the parts of the project that the course requirements and rubric ask for and that were still missing: the benchmark comparison with the published results, the error analysis, feature-engineering and class-imbalance experiments, the Week 3 neural baselines, figures for the presentation, a live demo, and the README.

## Overview of my contributions

| # | Contribution | Course requirement it covers | Files |
|---|---|---|---|
| 1 | Re-implemented the source paper's protocol and compared our results with the published numbers | "Compare your results with benchmarks" | `paper_protocol.py`, `results/paper_protocol.*` |
| 2 | Error analysis: error rates by class, project, length and misleading wording; manual review of 50 errors | Rubric: "error analysis and insights" | `error_analysis.py`, `results/error_analysis.json`, `results/error_examples.csv` |
| 3 | Naive Bayes variants, feature engineering (word + character n-grams, hand-crafted features), class-imbalance handling, ensemble, all tuned on validation | Rubric: "feature engineering creativity", "number of methods compared"; Week 3 tips on class imbalance and ensembles | `experiments.py`, `results/experiments.*`, `results/classical_predictions_*.csv` |
| 4 | TextCNN and BiLSTM with GloVe embeddings | Week 3 project strategy: "Try CNN or LSTM" | `train_neural.py` |
| 5 | Prediction export and model saving for the DeBERTa script | Needed for error analysis and the demo | `train_transformer.py` |
| 6 | Figures | Presentation: "show results with visualizations" | `make_figures.py`, `results/figures/` |
| 7 | Demo that classifies a new issue and shows the words behind the decision | Presentation: "demo if possible" | `predict.py` |
| 8 | Colab notebook for the GPU runs | Reproducibility | `colab_runs.ipynb` |
| 9 | README with results, method, reproduction steps; `.gitignore` entries for downloads and saved models | GitHub repository due before the presentation | `README.md`, `.gitignore` |

## Key results

The detailed tables are in the README. In short:

- **Benchmark.** Running the paper's protocol on our data reproduces its classifier ranking, and the published F-measures match our non-bug-class F1 within 0.015 for all five classifiers. Our tuned models clearly exceed the paper's best configuration on the same projects.
- **Error analysis.** Non-bug reports that use failure vocabulary are misclassified 45.5% of the time (13.8% without it); non-bug reports filed with the bug template 55.5%. Weblate is the weakest project because it is the only one with more non-bugs than bugs.
- **Feature engineering.** TF-IDF weighting lifts Naive Bayes from 63.1 to 80.5 Macro-F1 (random split). Larger vocabularies, character n-grams, hand-crafted features and a tuned decision threshold raise the best classical model from 83.5 to 85.6 (random) and from 84.1 to 87.1 (unseen project), closing about half of the gap to the transformers.
- **Neural baselines.** TextCNN and BiLSTM scripts are tested on small samples; the full runs are done with `colab_runs.ipynb`. <!-- TODO: add their scores after the Colab run -->

## Use of AI assistance

The scripts, the experiment runs on CPU and the README were produced with the help of an AI assistant (Claude). I reviewed the code and the results; the GPU runs were done on Google Colab.

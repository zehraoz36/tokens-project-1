# Tuana Harmankaya

- Student ID: 2309111029
- GitHub: [@tia88na](https://github.com/tia88na)

## Summary

I integrated the team's dataset and classical benchmark package into the shared repository, corrected dependency and documentation inconsistencies, and carried out AI-assisted verification of the existing pipeline. I also added, trained and evaluated the vanilla BERT baseline.

## What I did

- Integrated the team-provided raw and cleaned datasets, `preprocess.py`, `benchmark.py` and classical benchmark results into the repository. This supplied the files needed by the existing transformer training script. The original dataset selection, preprocessing implementation and classical pilot were another team member's work.
- Checked that `cleaned.jsonl` contained the required `text`, `label`, `random_split` and `cross_project_split` fields, and that `benchmark.py` exposed `SEED` and `metrics`. Checked that split sizes and baseline scores agreed with the existing experiment records.
- Added `transformers==5.18.0` to `requirements.txt`, matching the version recorded in the existing transformer experiment.
- Updated the existing license documents: removed outdated pilot wording, corrected the team copyright description, fixed the reference from the missing `source.json` to `results/data_audit.json`, and corrected a typographical error. These were maintenance changes to existing files.
- Updated the raw-file SHA-256 hashes in `results/data_audit.json` to match the CSV bytes stored in the repository. Documented Git's conversion of raw CSV line endings from CRLF to LF in `data/LICENSE`.
- Corrected the student ID placeholder in an existing team member's contribution file to match its filename.
- Adapted the existing transformer training flow, with AI assistance, to add `train_bert.py` for `bert-base-uncased`. Kept the existing data splits and shared metric function, and used separate BERT output filenames.
- Ran the BERT smoke test and full experiments in Colab. Added test prediction exports with text hashes, true labels and predicted labels, and committed the BERT script and results.

## Verification

With AI assistance, I checked the integrated repository through the following tests:

- A clean-clone preprocessing check reproduced `cleaned.jsonl` and `data_audit.json` byte for byte.
- The Colab classical benchmark reproduced all ten recorded results: five methods in each of the two scenarios.
- A small transformer check with `--limit 200` verified the integrated loading, training and evaluation path.

For my BERT baseline, I first ran `--limit 200 --epochs 1` successfully, then ran the full experiments on the existing random and cross-project splits.

## BERT experiment

- Architecture: BERT.
- Specific model: `bert-base-uncased`, fine-tuned for bug / non-bug classification.
- Environment: Google Colab, Tesla T4 GPU.
- Settings: 2 epochs, maximum input length of 256 tokens, batch size 16, AdamW learning rate 2e-5, mixed precision and seed 42.
- Checkpoint selection: highest validation Macro-F1; the test set was evaluated using the selected checkpoint. Epoch 1 was selected in both scenarios.

| Experiment | Model | Accuracy (%) | Macro-F1 (%) |
|---|---|---:|---:|
| Random split | bert-base-uncased | 90.52 | 87.81 |
| Cross-project (SciPy) | bert-base-uncased | 90.02 | 88.26 |

Limitations: one training seed, inputs truncated to 256 tokens, and a cross-project test covering SciPy only.

## AI assistance

AI assistants supported repository review, verification and BERT script adaptation. I ran the BERT smoke test, full training and evaluation in Colab, checked the outputs and committed the resulting files.

## Files I integrated or modified

- Team-provided package: `data/raw/`, `data/cleaned.jsonl`, `preprocess.py`, `benchmark.py`, `results/benchmark.csv`, `results/metrics.json` and `results/data_audit.json`.
- Repository corrections: `requirements.txt`, `LICENSE`, `data/LICENSE` and the existing contribution file's student ID.
- BERT baseline: `train_bert.py`.
- BERT results: `results/bert_benchmark.csv` and `results/bert_metrics.json`.
- BERT test predictions: `results/bert_predictions_random.csv` and `results/bert_predictions_cross_project.csv`.

## Commit evidence

- [Integrate the datasets, preprocessing and benchmark package](https://github.com/zehraoz36/tokens-project-1/commit/d890186b5f922f1264b52cbccec337807720f729).
- [Add the Transformers dependency](https://github.com/zehraoz36/tokens-project-1/commit/4662effce6e55397a581075dd61c5b1aa65933ea).
- [Update raw-file audit hashes and the line-ending note](https://github.com/zehraoz36/tokens-project-1/commit/b52d3769e6bc7bbab2e0cf3ce3f2ddcb35a8e776).
- [Add the BERT training script and evaluation results](https://github.com/zehraoz36/tokens-project-1/commit/6489cef191e7950de072d1986a06d5fe77fcacde).



# Yunus Emre Demirel

- Student ID: 210911053
- GitHub: [@iondris](https://github.com/iondris)

## Summary

Yunus's contribution is a BERT-specific verification and qualitative audit of the existing saved BERT predictions and results.

Attribution boundaries:

- **Tuana Harmankaya:** BERT implementation, training, metrics and prediction exports (`train_bert.py`, `results/bert_metrics.json`, `results/bert_predictions_random.csv`, `results/bert_predictions_cross_project.csv`).
- **Begüm Ziyade:** the repository's broad multi-model error analysis (`error_analysis.py`, `results/error_analysis.json`, `results/error_examples.csv`).
- **Yunus Emre Demirel:** BERT-specific verification and qualitative audit of the existing saved BERT predictions and results.

Yunus manually reviewed all 20 sampled BERT misclassifications, validated the existing six-category qualitative taxonomy on this sample, and refined two category assignments. Yunus did not redesign or create the taxonomy.

## Overview of the contribution

| # | Contribution | Files |
|---|---|---|
| 1 | Independent verification of the saved BERT predictions and results | `bert_error_review.py`, `results/bert_error_summary.csv` |
| 2 | Reproducible, deterministic seed-42 sample of 20 BERT errors (5 false positives + 5 false negatives per scenario) | `bert_error_review.py`, `results/bert_error_examples.csv` |
| 3 | Manual review of all 20 sampled cases: existing qualitative annotations validated, two category assignments refined | `results/bert_error_examples.csv` (`review_category`, `review_notes`) |
| 4 | Short report | `results/bert_error_analysis.md` |

## How this differs from Tuana's and Begüm's work

| Contributor | Related work | Relation to this contribution |
|---|---|---|
| Tuana Harmankaya | Implemented and trained BERT and exported its metrics and test predictions | This contribution only reads those files; it trains nothing and changes none of them |
| Begüm Ziyade | Broad error analysis across all models: error rates by class, project, length and cue words, model agreement, and a manual reading of sampled errors | This contribution adds none of these analyses |

The primary distinctions of this contribution are:

- **BERT-only analysis**, based only on the saved BERT predictions and metrics.
- **A deterministic 20-case audit:** 5 false positives and 5 false negatives per scenario, including SciPy errors from the cross-project test.
- **A case-level qualitative review of the 20 sampled BERT errors.**
- **One review category and one review note per record**, stored in `results/bert_error_examples.csv`.
- **Independent verification of the saved BERT predictions and results** (`bert_error_review.py`).

Minor factual note: none of the 20 records appear in `results/error_examples.csv`.

## 1. Quantitative verification

`bert_error_review.py` recomputes BERT's errors from the saved predictions:

| Scenario | Test records | False positives | False negatives | Total errors | Error rate | Non-bug error rate | Bug error rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 6,923 | 347 | 309 | 656 | 0.094757 | 0.187974 | 0.060863 |
| cross_project | 4,570 | 323 | 133 | 456 | 0.099781 | 0.216198 | 0.043238 |

The non-bug error rate is false positives divided by non-bug test records. The bug error rate is false negatives divided by bug test records. The non-bug class has the higher error rate in both scenarios.

All 29 checks of the script pass:

- **Test sizes.** The test splits have 6,923 records (random) and 4,570 (cross_project), and each prediction file has exactly that many rows.
- **Join.** The prediction export has no `record_id`, so rows are joined to the fixed test records by `text_sha256`. The hashes follow the test records of `data/cleaned.jsonl` in order, and every test text hashes to its `text_sha256`.
- **Labels.** Every saved true label equals the label in `data/cleaned.jsonl`.
- **Confusion matrices.** The recomputed matrices equal the saved ones in `results/bert_metrics.json`:
  - random: `[[1499, 347], [309, 4768]]`
  - cross_project: `[[1171, 323], [133, 2943]]`
- **Summary.** Every count and rate in `results/bert_error_summary.csv` equals the recomputed value. The file is verified, not rewritten.
- **Report.** The quantitative and category tables in `results/bert_error_analysis.md` match the recomputed values and `results/bert_error_examples.csv`.
- **Sampled cases.** The checks on the 20 cases are listed in section 2.

Run `python bert_error_review.py`. It needs only the standard library and takes a few seconds. It reads files but never writes them, and it exits with status 1 if any check fails.

## 2. Sample and review method

The original Colab analysis produced the three result artifacts. `bert_error_review.py` independently re-verifies those artifacts from the repository data and the saved BERT predictions.

**Sampling procedure** (from the original Colab analysis):

1. For each scenario, collect BERT's misclassified test records and create `random.Random(42)`.
2. For false positives, then false negatives, sort that type's errors by `record_id` and draw 5 with `rng.sample`.

Redrawing with this procedure gives the same 20 records, in the same order, as `results/bert_error_examples.csv`. This was checked with Python 3.12; the original Colab run used Python 3.13.

**Checks on the 20 cases:**

- every `record_id` is in the correct fixed test split;
- `project`, `true_label` and `text` equal `data/cleaned.jsonl`;
- `predicted_label` equals the BERT prediction export;
- every case is genuinely misclassified and has the correct `error_type`;
- every case has one of the six review categories and a non-empty review note.

**Qualitative review.** The six categories and the initial annotations came from the original Colab artifacts: the Colab script held them in a `REVIEWS` dictionary and wrote them into `bert_error_examples.csv`. Yunus manually reviewed all 20 sampled cases and their existing qualitative annotations, and refined two category assignments:

| Record | Previous category | Refined category | Reason |
|---|---|---|---|
| numpy:1346165 | change_or_fix_non_bug | limited_context | Short exception-chaining change description with an issue reference. It does not show whether the change repairs a defect or improves error reporting |
| scipy:1374992 | change_or_fix_non_bug | limited_context | Short test-coverage change description with an issue reference. It does not show whether the referenced issue was a defect |

The review notes of these two records were rewritten to explain the limited-context judgment. The other 18 assignments were kept. `bert_error_review.py` reads and validates the annotations; it does not generate or write them.

## 3. Review categories

These are post-hoc qualitative review categories. They are not ground-truth labels and not causal explanations of BERT's behaviour. Each record has exactly one category. Each category occurs with only one error type, because the categories are partly defined by the source label.

| Category | Description | Error type | Count | Records |
|---|---|---|---:|---|
| ambiguous_bug_enhancement_boundary | Bug-labelled text where the line between bug and enhancement is itself unclear (support proposals, design discussion, the author asking which it is) | FN | 3 | pandas:1332223, pandas:1334857, scipy:1373537 |
| bug_like_non_bug | Non-bug text that reads like a defect report (unexpected behaviour, reproduction code, the bug-report template, "is this by design?") | FP | 3 | weblate:1448934, pandas:1338676, salt:1256951 |
| change_or_fix_non_bug | Non-bug text that describes a change, fix or improvement with repair-related vocabulary | FP | 3 | scipy:1374342, scipy:1374625, scipy:1375067 |
| feature_or_question_wording_bug | Bug-labelled text phrased as a feature request, documentation request, missing support or question | FN | 4 | weblate:1447447, scipy:1371585, scipy:1371600, scipy:1371712 |
| implementation_oriented_bug | Bug-labelled text that describes implementation details or a proposed change rather than a conventional failure report | FN | 3 | salt:1248107, pandas:1333060, scipy:1371336 |
| limited_context | Very short text with little supporting context for classification | FP | 4 | pandas:1340305, numpy:1346165, scipy:1374926, scipy:1374992 |

## 4. Observed patterns

These are hypotheses from text inspection. They are not causal explanations of model decisions or proof that source labels are wrong.

- **Bug-like non-bugs.** Bug-like non-bug records describe unexpected behaviour, reproduction steps or fixes (pandas:1338676, salt:1256951).
- **Unclear label boundaries.** Source-label boundaries can be unclear from text alone. pandas:1334857 explicitly discusses whether its change is a bug fix or an enhancement.
- **Wording versus label.** weblate:1447447 has a bug label but uses a feature-request template. This is a possible mismatch between surface wording and the source label.
- **Shared repair vocabulary.** Repair and implementation descriptions occur in both classes (scipy:1374625, non-bug; scipy:1371336, bug).
- **Short records.** Short records such as pandas:1340305, numpy:1346165, scipy:1374992 and scipy:1374926 provide limited context.
- **Question wording in bugs.** Some bug-labelled records use question or documentation-request wording (scipy:1371600, scipy:1371585).

## Limitations

- **Sample size.** Only 20 errors were manually reviewed, out of 656 (random) and 456 (cross-project) BERT errors.
- **No prevalence estimate.** The sample is balanced by error type, so the category counts cannot estimate how common each category is among all BERT errors.
- **Single reviewer.** Yunus independently reviewed all 20 cases, but no second independent reviewer was used and no inter-annotator agreement was measured.
- **One category per record.** Each record has one category, even where more than one could apply.
- **Truncation.** The qualitative review considered the sampled issue content and annotations, while BERT inputs were limited to 256 tokens. The effect of truncation was not measured.
- **One training run.** Results come from one BERT training run.
- **Cross-project coverage.** The cross-project test covers SciPy only.

## Files

Added:

- `bert_error_review.py`
- `contributions/YUNUS_EMRE_DEMIREL_210911053_iondris.md`

BERT error-review artifacts included in Yunus's contribution (originally produced by the Colab analysis):

- `results/bert_error_summary.csv` (unchanged)
- `results/bert_error_examples.csv` (two category assignments and their review notes refined: numpy:1346165, scipy:1374992)
- `results/bert_error_analysis.md` (category counts and two pattern bullets updated for the refinement, and the truncation limitation reworded for accuracy)

Read, not modified:

- `data/cleaned.jsonl`
- `results/bert_predictions_random.csv`
- `results/bert_predictions_cross_project.csv`
- `results/bert_metrics.json`

## Use of AI assistance

The verification script and this contribution text were drafted with the help of an AI assistant (Claude). Yunus manually reviewed all 20 cases and reviewed the resulting outputs and final contribution materials.

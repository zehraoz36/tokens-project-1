# BERT Error Analysis

Model: bert-base-uncased. Labels: 0 = non-bug, 1 = bug.
Predictions were checked against the cleaned test splits and saved confusion matrices.

## Quantitative results

| Scenario | Test records | False positives | False negatives | Error rate | Non-bug error rate | Bug error rate |
|---|---:|---:|---:|---:|---:|---:|
| random | 6923 | 347 | 309 | 9.48% | 18.80% | 6.09% |
| cross_project | 4570 | 323 | 133 | 9.98% | 21.62% | 4.32% |

Non-bug records have a higher class-conditional error rate in both scenarios. This supports reporting Macro-F1 and per-class metrics alongside accuracy.

## Qualitative review

Twenty errors were sampled: five false positives and five false negatives per scenario, using seed 42. Review notes are stored in bert_error_examples.csv.

| Review category | Sampled records |
|---|---:|
| ambiguous_bug_enhancement_boundary | 3 |
| bug_like_non_bug | 3 |
| change_or_fix_non_bug | 3 |
| feature_or_question_wording_bug | 4 |
| implementation_oriented_bug | 3 |
| limited_context | 4 |

### Observed patterns

- Bug-like non-bug records describe unexpected behavior, reproduction steps or fixes. Examples include pandas:1338676 and salt:1256951.
- Source-label boundaries can be unclear from text alone. pandas:1334857 explicitly discusses the bug-versus-enhancement distinction.
- weblate:1447447 has a bug label but uses a feature-request template. This illustrates a possible mismatch between surface wording and the source label.
- Repair and implementation descriptions occur in both classes, as illustrated by scipy:1374625 and scipy:1371336.
- Short records such as pandas:1340305, numpy:1346165, scipy:1374992 and scipy:1374926 provide limited context.
- Some bug-labelled records use question or documentation-request wording, including scipy:1371600 and scipy:1371585.

## Limitations

These observations are hypotheses from text inspection, not causal explanations of model decisions or proof of incorrect source labels.
The sample is balanced by error type and does not estimate pattern frequencies across all test errors.
The qualitative review considered the sampled issue content and annotations, while BERT inputs were limited to 256 tokens. The effect of truncation was not measured.
Results come from one training seed; the cross-project test covers SciPy only.

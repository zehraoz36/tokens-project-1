"""BERT-specific error review: re-verification of the saved BERT error-review artifacts.
CPU only, standard library, a few seconds. Writes nothing.

The three bert_error_* artifacts were produced by the original Colab analysis. This script does
not generate or rewrite them; it re-checks them against the repository data and the saved BERT
predictions, and exits with status 1 if any check fails.

Reads
  data/cleaned.jsonl
  results/bert_predictions_<scenario>.csv   saved BERT test predictions (train_bert.py)
  results/bert_metrics.json                 saved BERT test confusion matrices (train_bert.py)
  results/bert_error_summary.csv            BERT error counts and rates per scenario
  results/bert_error_examples.csv           20 sampled BERT errors with qualitative annotations
  results/bert_error_analysis.md            short report

Checks
  - prediction rows match the fixed test split record by record (text_sha256, in order)
    and the saved true labels match cleaned.jsonl
  - false positives, false negatives and error rates, recomputed from the predictions, match
    bert_error_summary.csv and the confusion matrices in bert_metrics.json
  - the seed-42 sample (5 false positives + 5 false negatives per scenario) is redrawn and
    compared with bert_error_examples.csv
  - every sampled row is a BERT error on the right test record, with the cleaned text
  - review_category is one of the six review categories and review_notes is filled in.
    These two columns are qualitative annotations (initially from the Colab analysis, reviewed
    and refined by Yunus); they are read and validated, never generated or written.
"""
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
PER_TYPE = 5
SCENARIOS = {"random": 6923, "cross_project": 4570}  # expected test split sizes
ERROR_TYPES = ("false_positive", "false_negative")
CATEGORIES = ("ambiguous_bug_enhancement_boundary", "bug_like_non_bug", "change_or_fix_non_bug",
              "feature_or_question_wording_bug", "implementation_oriented_bug", "limited_context")
SUMMARY_COLUMNS = ["scenario", "n_test", "false_positives", "false_negatives", "total_errors",
                   "error_rate", "non_bug_error_rate", "bug_error_rate"]
EXAMPLE_COLUMNS = ["scenario", "record_id", "project", "true_label", "predicted_label",
                   "error_type", "text", "review_category", "review_notes"]

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def verify_predictions(scenario, test):
    """Join the saved predictions to the fixed test records and recompute the BERT errors."""
    _, preds = read_csv(RESULTS / f"bert_predictions_{scenario}.csv")
    check(f"{scenario}: test split has {SCENARIOS[scenario]} records", len(test) == SCENARIOS[scenario],
          str(len(test)))
    check(f"{scenario}: prediction rows equal test records", len(preds) == len(test),
          f"{len(preds)} vs {len(test)}")
    check(f"{scenario}: prediction rows follow the test records in order (text_sha256)",
          [p["text_sha256"] for p in preds] == [r["text_sha256"] for r in test])
    check(f"{scenario}: every test text hashes to its text_sha256",
          all(hashlib.sha256(r["text"].encode("utf-8")).hexdigest() == r["text_sha256"] for r in test))
    by_hash = {r["text_sha256"]: r for r in test}
    matrix, predicted, errors, bad_labels = [[0, 0], [0, 0]], {}, [], 0
    for p in preds:
        r = by_hash.get(p["text_sha256"])
        if (r is None or {p["true_label"], p["predicted_label"]} - {"0", "1"}
                or int(p["true_label"]) != r["label"]):
            bad_labels += 1
            continue
        actual, pred = int(p["true_label"]), int(p["predicted_label"])
        matrix[actual][pred] += 1
        predicted[r["record_id"]] = pred
        if actual != pred:
            errors.append(dict(record_id=r["record_id"], error_type=ERROR_TYPES[actual]))
    check(f"{scenario}: saved true labels match cleaned.jsonl", bad_labels == 0,
          f"{bad_labels} mismatched rows" if bad_labels else "")
    (tn, fp), (fn, tp) = matrix
    n = len(preds)
    summary = dict(scenario=scenario, n_test=n, false_positives=fp, false_negatives=fn,
                   total_errors=fp + fn, error_rate=round((fp + fn) / n, 6),
                   non_bug_error_rate=round(fp / (tn + fp), 6), bug_error_rate=round(fn / (fn + tp), 6))
    return summary, matrix, predicted, errors


def draw_sample(scenario, errors):
    """The original selection: one Random(42) per scenario; 5 false positives, then 5 false
    negatives, drawn from that scenario's errors sorted by record_id."""
    rng = random.Random(SEED)
    sample = []
    for error_type in ERROR_TYPES:
        candidates = sorted((e for e in errors if e["error_type"] == error_type), key=lambda e: e["record_id"])
        sample += [(scenario, e["record_id"]) for e in rng.sample(candidates, min(PER_TYPE, len(candidates)))]
    return sample


def main():
    with (ROOT / "data/cleaned.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    check("cleaned.jsonl: record_id and text_sha256 are unique",
          len({r["record_id"] for r in rows}) == len({r["text_sha256"] for r in rows}) == len(rows),
          f"{len(rows)} records")
    tests = {s: [r for r in rows if r[s + "_split"] == "test"] for s in SCENARIOS}
    saved_metrics = {e["scenario"]: e for e in
                     json.loads((RESULTS / "bert_metrics.json").read_text(encoding="utf-8"))["experiments"]}
    summary_columns, saved_summary = read_csv(RESULTS / "bert_error_summary.csv")
    saved_summary = {s["scenario"]: s for s in saved_summary}
    check("bert_error_summary.csv: expected columns and one row per scenario",
          summary_columns == SUMMARY_COLUMNS and sorted(saved_summary) == sorted(SCENARIOS))

    summaries, predicted, expected_sample = {}, {}, []
    for scenario in SCENARIOS:
        summary, matrix, predicted[scenario], errors = verify_predictions(scenario, tests[scenario])
        summaries[scenario] = summary
        recorded = saved_metrics[scenario]
        check(f"{scenario}: confusion matrix equals bert_metrics.json",
              matrix == recorded["test"]["confusion_matrix"] and summary["n_test"] == recorded["n_test"],
              str(matrix))
        saved = saved_summary.get(scenario, {})
        differ = [key for key in SUMMARY_COLUMNS[1:] if key not in saved or (
            str(summary[key]) != saved[key] if isinstance(summary[key], int)
            else summary[key] != round(float(saved[key]), 6))]
        check(f"{scenario}: bert_error_summary.csv matches recomputed values", not differ,
              "differs: " + ", ".join(differ) if differ else "")
        expected_sample += draw_sample(scenario, errors)

    example_columns, examples = read_csv(RESULTS / "bert_error_examples.csv")
    saved_sample = [(e["scenario"], e["record_id"]) for e in examples]
    check("bert_error_examples.csv: expected columns", example_columns == EXAMPLE_COLUMNS)
    check("bert_error_examples.csv: 20 rows, no duplicate records",
          len(examples) == 2 * len(SCENARIOS) * PER_TYPE and len(set(saved_sample)) == len(saved_sample),
          str(len(examples)))
    counts = Counter((e["scenario"], e["error_type"]) for e in examples)
    check("bert_error_examples.csv: 5 false positives + 5 false negatives per scenario",
          all(counts[(s, t)] == PER_TYPE for s in SCENARIOS for t in ERROR_TYPES)
          and sum(counts.values()) == len(examples))
    if saved_sample == expected_sample:
        outcome = "same 20 records in the same order"
    elif sorted(saved_sample) == sorted(expected_sample):
        outcome = "same records, different order"
    else:
        outcome = f"{len(set(saved_sample) & set(expected_sample))} of {len(expected_sample)} records match"
    check("seed-42 sample redrawn from the predictions reproduces bert_error_examples.csv",
          saved_sample == expected_sample, outcome)

    test_by_id = {s: {r["record_id"]: r for r in test} for s, test in tests.items()}
    row_checks = [
        ("record_id is in the scenario's fixed test split", lambda e, r: True),
        ("project and true_label match cleaned.jsonl",
         lambda e, r: e["project"] == r["project"] and e["true_label"] == str(r["label"])),
        ("predicted_label matches the BERT prediction export",
         lambda e, r: e["predicted_label"] == str(predicted[e["scenario"]].get(e["record_id"]))),
        ("row is misclassified and error_type matches the labels",
         lambda e, r: e["true_label"] != e["predicted_label"]
         and e["error_type"] == {"0": ERROR_TYPES[0], "1": ERROR_TYPES[1]}.get(e["true_label"])),
        ("text equals the cleaned text", lambda e, r: e["text"] == r["text"]),
        ("review_category is one of the six review categories", lambda e, r: e["review_category"] in CATEGORIES),
        ("review_notes is filled in", lambda e, r: bool(e["review_notes"].strip())),
    ]
    for name, row_ok in row_checks:
        bad = [e["record_id"] for e in examples
               if e["record_id"] not in test_by_id.get(e["scenario"], {})
               or not row_ok(e, test_by_id[e["scenario"]][e["record_id"]])]
        check(f"examples: {name}", not bad, ", ".join(bad))

    categories = Counter(e["review_category"] for e in examples)
    report = (RESULTS / "bert_error_analysis.md").read_text(encoding="utf-8")
    missing = [s["scenario"] for s in summaries.values() if (
        f"| {s['scenario']} | {s['n_test']} | {s['false_positives']} | {s['false_negatives']} | "
        f"{s['error_rate']:.2%} | {s['non_bug_error_rate']:.2%} | {s['bug_error_rate']:.2%} |") not in report]
    check("bert_error_analysis.md: quantitative table matches recomputed values", not missing,
          "differs: " + ", ".join(missing) if missing else "")
    missing = [c for c in CATEGORIES if f"| {c} | {categories[c]} |" not in report]
    check("bert_error_analysis.md: category table matches bert_error_examples.csv", not missing,
          "differs: " + ", ".join(missing) if missing else "")

    print(f"\n{'scenario':<14} {'n_test':>6} {'FP':>4} {'FN':>4} {'errors':>6} "
          f"{'error_rate':>10} {'non_bug_rate':>12} {'bug_rate':>9}")
    for s in summaries.values():
        print(f"{s['scenario']:<14} {s['n_test']:>6} {s['false_positives']:>4} {s['false_negatives']:>4} "
              f"{s['total_errors']:>6} {s['error_rate']:>10.6f} {s['non_bug_error_rate']:>12.6f} "
              f"{s['bug_error_rate']:>9.6f}")
    print("\nReview categories (sampled records):")
    for category in CATEGORIES:
        print(f"  {category:<36} {categories[category]}")

    failed = results.count(False)
    if failed:
        print(f"\n{failed} of {len(results)} checks failed.")
        sys.exit(1)
    print(f"\nAll {len(results)} checks passed.")


if __name__ == "__main__":
    main()

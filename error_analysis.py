"""Error analysis of saved test predictions. CPU only, a few seconds.

Reads every prediction file that exists in results/:
  bert_predictions_<scenario>.csv          (train_bert.py)
  transformer_predictions_<scenario>.csv   (train_transformer.py)
  classical_predictions_<scenario>.csv     (experiments.py)
  neural_<cnn|bilstm>_predictions_<scenario>.csv (train_neural.py)
and joins it with data/cleaned.jsonl.

Outputs
  results/error_analysis.json   error rates by project, text length and cue words;
                                false positive / false negative counts; model agreement
  results/error_examples.csv    a fixed random sample (seed 42) of misclassified test
                                reports of BERT (or the first available model), with the first
                                300 characters of text, for manual review
"""
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCES = [("deberta", "transformer_predictions_{}.csv"), ("bert", "bert_predictions_{}.csv"),
           ("classical", "classical_predictions_{}.csv"),
           ("cnn", "neural_cnn_predictions_{}.csv"), ("bilstm", "neural_bilstm_predictions_{}.csv")]
LENGTH_BINS = [(0, 50, "<50 words"), (50, 150, "50-149"), (150, 400, "150-399"),
               (400, 10 ** 9, "400+ (beyond the 256-token limit)")]
CUES = {
    "error / exception words": r"\b(?:error|exception|traceback|fail(?:s|ed|ure)?|crash(?:es|ed)?|segfault)\b",
    "request / docs words": r"\b(?:add|support|allow|feature|enhancement|proposal|suggest|deprecat\w*|documentation|docs?|typo)\b",
    "question mark": r"\?",
    "pull-request text": r"\b(?:closes|fixes|xref) #\d+|tests added|what does this (?:pr|implement)|whatsnew entry",
    "bug-report template": r"confirmed this bug|describe the bug|steps to reproduce|expected (?:behaviou?r|output)",
}


def load_predictions(scenario, rows_by_hash, rows_by_id):
    preds = {}
    for name, pattern in SOURCES:
        path = ROOT / "results" / pattern.format(scenario)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            p = {}
            for r in reader:
                record = (rows_by_id[r["record_id"]] if "record_id" in r
                          else rows_by_hash[(r["text_sha256"], scenario)])
                if int(r["true_label"]) != record["label"]:
                    raise ValueError(f"{path.name}: label mismatch for {record['record_id']}")
                p[record["record_id"]] = int(r["predicted_label"])
        preds[name] = p
    return preds


def rate(errors, total):
    return dict(n=int(total), errors=int(errors), error_rate=round(errors / total, 4) if total else None)


def main():
    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    rows_by_id = {r["record_id"]: r for r in rows}
    rows_by_hash = {}
    for r in rows:
        for scenario in ["random", "cross_project"]:
            if r[scenario + "_split"] == "test":
                rows_by_hash[(r["text_sha256"], scenario)] = r
    report, examples = {}, []
    for scenario in ["random", "cross_project"]:
        preds = load_predictions(scenario, rows_by_hash, rows_by_id)
        if not preds:
            continue
        test = [r for r in rows if r[scenario + "_split"] == "test"]
        section = {}
        for name, p in preds.items():
            if len(p) != len(test):
                raise ValueError(f"{name}/{scenario}: {len(p)} predictions for {len(test)} test rows")
            wrong = {r["record_id"] for r in test if p[r["record_id"]] != r["label"]}
            fp = sum(1 for r in test if r["record_id"] in wrong and r["label"] == 0)
            fn = len(wrong) - fp
            s = dict(errors=len(wrong), error_rate=round(len(wrong) / len(test), 4),
                     false_positives_nonbug_called_bug=fp, false_negatives_bug_called_nonbug=fn)
            s["by_project"] = {proj: rate(sum(1 for r in test if r["project"] == proj and r["record_id"] in wrong),
                                          sum(1 for r in test if r["project"] == proj))
                               for proj in sorted({r["project"] for r in test})}
            s["by_true_class"] = {cls: rate(sum(1 for r in test if r["label"] == lab and r["record_id"] in wrong),
                                            sum(1 for r in test if r["label"] == lab))
                                  for cls, lab in [("non-bug", 0), ("bug", 1)]}
            s["by_length"] = {}
            for lo, hi, label in LENGTH_BINS:
                sub = [r for r in test if lo <= len(r["text"].split()) < hi]
                s["by_length"][label] = rate(sum(1 for r in sub if r["record_id"] in wrong), len(sub))
            # cue words: how often a misleading cue co-occurs with an error
            s["cues"] = {}
            for cue, pattern in CUES.items():
                for cls, lab in [("non-bug", 0), ("bug", 1)]:
                    sub = [r for r in test if r["label"] == lab and re.search(pattern, r["text"])]
                    s["cues"][f"{cls} WITH {cue}"] = rate(
                        sum(1 for r in sub if r["record_id"] in wrong), len(sub))
                    sub = [r for r in test if r["label"] == lab and not re.search(pattern, r["text"])]
                    s["cues"][f"{cls} WITHOUT {cue}"] = rate(
                        sum(1 for r in sub if r["record_id"] in wrong), len(sub))
            section[name] = s
        # agreement between models
        names = list(preds)
        if len(names) > 1:
            agreement = {}
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    wa = {k for k, v in preds[a].items() if v != rows_by_id[k]["label"]}
                    wb = {k for k, v in preds[b].items() if v != rows_by_id[k]["label"]}
                    agreement[f"{a} vs {b}"] = dict(
                        both_wrong=len(wa & wb), only_first_wrong=len(wa - wb),
                        only_second_wrong=len(wb - wa),
                        oracle_accuracy_if_either_right=round(1 - len(wa & wb) / len(test), 4))
            section["agreement"] = agreement
        report[scenario] = section
        # examples for manual review: fixed random sample
        # the manual review in the README was done on BERT's errors; keep that sample stable
        best = "bert" if "bert" in names else names[0]
        wrong = [r for r in test if preds[best][r["record_id"]] != r["label"]]
        rng = np.random.default_rng(SEED)
        for kind, lab in [("false positive (non-bug called bug)", 0), ("false negative (bug called non-bug)", 1)]:
            pool = [r for r in wrong if r["label"] == lab]
            for i in rng.choice(len(pool), min(25, len(pool)), replace=False):
                r = pool[i]
                examples.append(dict(scenario=scenario, model=best, error=kind, record_id=r["record_id"],
                                     project=r["project"], words=len(r["text"].split()),
                                     others_wrong=",".join(n for n in names[1:]
                                                           if preds[n][r["record_id"]] != r["label"]),
                                     text_start=r["text"][:300]))
    (ROOT / "results/error_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (ROOT / "results/error_examples.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)
    print(json.dumps({s: {n: {k: v for k, v in d.items() if k in ("errors", "error_rate",
          "false_positives_nonbug_called_bug", "false_negatives_bug_called_nonbug")}
          for n, d in sec.items() if n != "agreement"} for s, sec in report.items()}, indent=1))
    print("Saved results/error_analysis.json and results/error_examples.csv")


if __name__ == "__main__":
    main()

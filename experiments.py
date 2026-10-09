"""Extra classical experiments: Naive Bayes variants, feature engineering,
class imbalance handling and an ensemble. CPU only; the two scenarios run in
parallel, about 40-60 minutes on a 2-core machine.

Run preprocess.py first. Uses the same cleaned data, fixed splits, seed and
metrics() function as benchmark.py, so every number is comparable with
results/benchmark.csv. Every configuration (including hyperparameters such as
alpha, C and the decision threshold) is chosen on the validation set; the test
set is scored once per experiment, with the selected configuration.

Outputs
  results/experiments.csv                     one row per experiment and scenario
  results/experiments_metrics.json            full metrics + validation grids
  results/classical_predictions_<scenario>.csv test predictions of the best
                                              classical model (for error analysis)
"""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from benchmark import SEED, metrics

ROOT = Path(__file__).resolve().parent
CHAR_LIMIT = 3000  # character n-grams are built from the first 3,000 characters


# ---------------------------------------------------------------- features
def word_tfidf(max_features=5000, sublinear=False):
    return TfidfVectorizer(max_features=max_features, min_df=2, ngram_range=(1, 2),
                           sublinear_tf=sublinear, dtype=np.float32)


def char_tfidf(max_features=100_000):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           max_features=max_features, sublinear_tf=True, dtype=np.float32,
                           preprocessor=lambda t: t[:CHAR_LIMIT])


META_PATTERNS = {
    "traceback": r"traceback|stack trace|stacktrace",
    "error_words": r"\b(?:error|exception|fail(?:s|ed|ure)?|crash(?:es|ed)?|segfault|wrong|incorrect|broken)\b",
    "request_words": r"\b(?:add|support|allow|feature|enhancement|would be nice|proposal|suggest|deprecat\w*|doc(?:s|umentation)?|typo)\b",
    "expected_actual": r"\bexpected\b|\bactual\b",
    "version": r"\b\d+\.\d+(?:\.\d+)?\b",
    "url": r"\burl\b",
    "question": r"\?",
}


def meta_features(texts):
    """Hand-crafted signals that bag-of-words does not capture directly."""
    rows = []
    for t in texts:
        words = t.split()
        n = max(len(words), 1)
        row = [np.log1p(len(t)), np.log1p(len(words)),
               sum(c.isdigit() for c in t) / max(len(t), 1),
               sum(not c.isalnum() and not c.isspace() for c in t) / max(len(t), 1)]
        row += [np.log1p(len(re.findall(p, t))) for p in META_PATTERNS.values()]
        row += [len(re.findall(META_PATTERNS["error_words"], t)) / n,
                len(re.findall(META_PATTERNS["request_words"], t)) / n]
        rows.append(row)
    return np.array(rows, dtype=np.float32)


META_NAMES = (["log_chars", "log_words", "digit_ratio", "symbol_ratio"]
              + [f"log_count_{k}" for k in META_PATTERNS] + ["error_word_rate", "request_word_rate"])


class Features:
    """Fit feature extractors on train, transform val and test."""

    def __init__(self, kind, texts):
        self.parts = []
        if kind in ("word5k", "word50k", "word+char", "word+char+meta", "word+meta"):
            self.parts.append(word_tfidf(5000) if kind == "word5k" else word_tfidf(50_000, True))
        if kind in ("char", "word+char", "word+char+meta"):
            self.parts.append(char_tfidf())
        self.meta = kind.endswith("meta")
        self.x = {}
        for split in ["train", "val", "test"]:
            blocks = []
            for vec in self.parts:
                blocks.append(vec.fit_transform(texts[split]) if split == "train"
                              else vec.transform(texts[split]))
            if self.meta:
                m = meta_features(texts[split])
                if split == "train":
                    self.scaler = StandardScaler().fit(m)
                blocks.append(csr_matrix(self.scaler.transform(m).astype(np.float32)))
            self.x[split] = hstack(blocks).tocsr() if len(blocks) > 1 else blocks[0]


# ---------------------------------------------------------------- helpers
def macro(y, pred):
    return float(f1_score(y, pred, average="macro"))


def select(candidates, y_val):
    """candidates: list of (params, fitted_model, val_scores_or_pred). Highest val Macro-F1 wins."""
    grid = [dict(params=p, val_macro_f1=round(s, 5)) for p, _, s in candidates]
    best = max(candidates, key=lambda c: c[2])
    return best, grid


def fit_lr(x, y, x_val, y_val, class_weight=None, tune_threshold=False):
    candidates = []
    for c in [0.5, 1, 2, 4, 8, 16]:
        model = LogisticRegression(C=c, max_iter=3000, class_weight=class_weight,
                                   random_state=SEED).fit(x, y)
        prob = model.predict_proba(x_val)[:, 1]
        thresholds = np.arange(0.30, 0.71, 0.02) if tune_threshold else [0.5]
        for th in thresholds:
            candidates.append((dict(C=c, threshold=round(float(th), 2)), model,
                               macro(y_val, (prob >= th).astype(int))))
    (params, model, _), grid = select(candidates, y_val)
    th = params["threshold"]
    return params, grid, lambda m: (model.predict_proba(m)[:, 1] >= th).astype(int), \
        lambda m: model.predict_proba(m)[:, 1]


def fit_svm(x, y, x_val, y_val, class_weight=None):
    candidates = []
    for c in [0.05, 0.1, 0.25, 0.5]:  # larger C converges very slowly on these features
        model = LinearSVC(C=c, class_weight=class_weight, max_iter=3000,
                          random_state=SEED).fit(x, y)
        candidates.append((dict(C=c), model, macro(y_val, model.predict(x_val))))
    (params, model, _), grid = select(candidates, y_val)
    return params, grid, model.predict, model.decision_function


def fit_nb(kind, x, y, x_val, y_val):
    candidates = []
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0]:
        model = (ComplementNB if kind == "complement" else MultinomialNB)(alpha=alpha).fit(x, y)
        candidates.append((dict(alpha=alpha), model, macro(y_val, model.predict(x_val))))
    (params, model, _), grid = select(candidates, y_val)
    return params, grid, model.predict, lambda m: model.predict_proba(m)[:, 1]


# ---------------------------------------------------------------- main
EXPERIMENTS = [
    # group, name, features, model
    ("A. Naive Bayes", "NB_counts_tuned", "counts5k", "multinomial"),
    ("A. Naive Bayes", "NB_tfidf", "word50k", "multinomial"),
    ("A. Naive Bayes", "ComplementNB_tfidf", "word50k", "complement"),
    ("B. Features (LR)", "LR_word5k", "word5k", "lr"),
    ("B. Features (LR)", "LR_word50k_sublinear", "word50k", "lr"),
    ("B. Features (LR)", "LR_char2-5", "char", "lr"),
    ("B. Features (LR)", "LR_word+char", "word+char", "lr"),
    ("B. Features (LR)", "LR_word+char+meta", "word+char+meta", "lr"),
    ("C. Imbalance", "LR_word+char+meta_balanced", "word+char+meta", "lr_balanced"),
    ("C. Imbalance", "LR_word+char+meta_threshold", "word+char+meta", "lr_threshold"),
    ("C. Imbalance", "SVM_word+char+meta", "word+char+meta", "svm"),
    ("C. Imbalance", "SVM_word+char+meta_balanced", "word+char+meta", "svm_balanced"),
    ("D. Ensemble", "Ensemble_LR+SVM+CNB", "word+char+meta", "ensemble"),
]


def percentile_average(scores, references):
    """Average of each member's score expressed as a percentile of its VALIDATION
    scores, so probabilities and SVM margins mix fairly. Only validation data is
    used as the reference, never the test distribution."""
    return np.mean([np.searchsorted(np.sort(ref), s, side="right") / len(ref)
                    for s, ref in zip(scores, references)], axis=0)


def run_scenario(scenario):
    """All experiments for one scenario (run in its own process)."""
    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    sets = {s: [r for r in rows if r[scenario + "_split"] == s] for s in ["train", "val", "test"]}
    y = {s: np.array([r["label"] for r in v]) for s, v in sets.items()}
    texts = {s: [r["text"] for r in v] for s, v in sets.items()}
    print(f"== {scenario}: " + str({s: len(v) for s, v in sets.items()}), flush=True)
    cache, trained, summary, details, best = {}, {}, [], [], (-1.0, None, None)
    for group, name, feats, kind in EXPERIMENTS:
        start = time.perf_counter()
        if feats not in cache:
            if feats == "counts5k":
                vec = CountVectorizer(max_features=5000, min_df=2)
                cache[feats] = {"train": vec.fit_transform(texts["train"]),
                                "val": vec.transform(texts["val"]),
                                "test": vec.transform(texts["test"])}
            else:
                cache[feats] = Features(feats, texts).x
        x = cache[feats]
        with threadpool_limits(limits=1):
            if kind in ("multinomial", "complement"):
                params, grid, predict, score = fit_nb(kind, x["train"], y["train"], x["val"], y["val"])
            elif kind.startswith("lr"):
                params, grid, predict, score = fit_lr(
                    x["train"], y["train"], x["val"], y["val"],
                    class_weight="balanced" if kind == "lr_balanced" else None,
                    tune_threshold=kind == "lr_threshold")
            elif kind.startswith("svm"):
                params, grid, predict, score = fit_svm(
                    x["train"], y["train"], x["val"], y["val"],
                    class_weight="balanced" if kind == "svm_balanced" else None)
            else:  # ensemble of already-selected models, threshold chosen on validation
                members = ["LR_word+char+meta_balanced", "SVM_word+char+meta_balanced",
                           "ComplementNB_tfidf"]
                refs = [trained[m]["score"](trained[m]["x"]["val"]) for m in members]

                def avg(split):
                    return percentile_average([trained[m]["score"](trained[m]["x"][split])
                                               for m in members], refs)
                val_scores, test_scores = avg("val"), avg("test")
                cands = [(dict(members=members, threshold=round(float(t), 2)), None,
                          macro(y["val"], (val_scores >= t).astype(int)))
                         for t in np.arange(0.10, 0.61, 0.01)]
                (params, _, _), grid = select(cands, y["val"])
                t = params["threshold"]
                val_pred = (val_scores >= t).astype(int)
                test_pred = (test_scores >= t).astype(int)
            if kind != "ensemble":
                trained[name] = dict(score=score, x=x)
                val_pred, test_pred = predict(x["val"]), predict(x["test"])
        val, test = metrics(y["val"], val_pred), metrics(y["test"], test_pred)
        seconds = round(time.perf_counter() - start, 2)
        row = dict(scenario=scenario, group=group, model=name, features=feats,
                   selected=json.dumps(params), val_macro_f1=val["macro_f1"],
                   accuracy=test["accuracy"], macro_f1=test["macro_f1"], bug_f1=test["bug_f1"],
                   nonbug_f1=test["per_class"]["non-bug"]["f1-score"], seconds=seconds)
        summary.append(row)
        details.append(dict(**row, n_features=int(x["train"].shape[1]),
                            validation_grid=grid, validation=val, test=test))
        print(f"{scenario[:6]} {name:30s} val={val['macro_f1']:.4f}  test acc={test['accuracy']:.4f} "
              f"macro-F1={test['macro_f1']:.4f}  {seconds:.0f}s  {params}", flush=True)
        # best classical model by VALIDATION score -> export its test predictions
        if val["macro_f1"] > best[0]:
            best = (val["macro_f1"], name, test_pred)
    _, name, pred = best
    with (ROOT / "results" / f"classical_predictions_{scenario}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["record_id", "true_label", "predicted_label", "model"])
        for r, p in zip(sets["test"], pred):
            writer.writerow([r["record_id"], r["label"], int(p), name])
    print(f"{scenario}: best on validation = {name}", flush=True)
    return summary, details


def main():
    from concurrent.futures import ProcessPoolExecutor
    (ROOT / "results").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:  # the two scenarios in parallel
        outputs = list(pool.map(run_scenario, ["random", "cross_project"]))
    summary = [row for s, _ in outputs for row in s]
    details = dict(seed=SEED, primary_metric="macro_f1", positive_class="bug",
                   selection="validation Macro-F1", meta_features=META_NAMES,
                   char_ngrams_first_chars=CHAR_LIMIT, experiments=[d for _, ds in outputs for d in ds])
    out = ROOT / "results"
    with (out / "experiments.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (out / "experiments_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("\nSaved results/experiments.csv, results/experiments_metrics.json, "
          "results/classical_predictions_*.csv")


if __name__ == "__main__":
    main()

"""Benchmark comparison with the dataset's source paper.

Andrade et al. (arXiv 2503.00660, Table 4) report, for exactly our five Python
projects (numpy, pandas, salt, scipy, weblate), an average F-measure of
NB 0.71, LR 0.74, RF 0.73, SVM 0.72, KNN 0.69.

Their protocol differs from ours, so their numbers cannot be compared with
results/benchmark.csv directly. This script re-implements their protocol on the
raw data, so that (1) we can check that we reproduce the published numbers and
(2) we can measure how much our own choices (full text instead of titles,
no undersampling, richer features, transformers) add on top.

Protocol as described in the paper (Sections 3.2-3.5, Table 3):
  * input: issue title only
  * lower-casing, punctuation and single-character removal, English stop words
    removed except "not", WordNet lemmatisation
  * TF-IDF bag of words, chi-squared feature selection to 250 features
  * random 70 / 30 split; training set balanced by random undersampling,
    test set keeps the original class proportions
  * classifiers with the paper's tuned parameters (Table 3)
  * 30 repetitions with different random splits; mean over runs
The paper does not say whether "F-measure" is the bug-class F1, the macro or the
weighted average, and whether the Python row pools the five projects or averages
per-project runs. The text says Table 4 averages runs over "the full dataset"
(30 runs x 5 classifiers x 52 projects = 7,800 values, i.e. one experiment per
project), while RQ3 pools 1,600 random reports from each of 5 projects (8,000).
We therefore report all three F1 variants for three settings: per-project runs
averaged (Table 4 reading), the RQ3 8,000-report sample, and all reports pooled.

Also runs the same protocol with title + description (our cleaned text) to show
the effect of the input field alone.

Needs nltk with WordNet:  pip install nltk && python -m nltk.downloader wordnet omw-1.4
CPU only, about 15-25 minutes.  Output: results/paper_protocol.csv, results/paper_protocol.json
"""
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RUNS = 30
K_FEATURES = 250
RQ3_PER_PROJECT = 1600
PAPER_PYTHON = {"NB": 0.71, "LR": 0.74, "RF": 0.73, "SVM": 0.72, "KNN": 0.69}
PROJECTS = {
    "numpy": "github_numpy_numpy_labeled.csv",
    "pandas": "github_pandasdev_pandas_labeled.csv",
    "salt": "github_saltstack_salt_labeled.csv",
    "scipy": "github_scipy_scipy_labeled.csv",
    "weblate": "github_weblateorg_weblate_labeled.csv",
}
STOP = ENGLISH_STOP_WORDS - {"not"}


def classifiers(seed):
    """Parameters from the paper's Table 3 (NB has none listed: defaults)."""
    return {
        "NB": MultinomialNB(),
        "LR": LogisticRegression(C=1.5, solver="newton-cg", max_iter=1000),
        "RF": RandomForestClassifier(n_estimators=200, criterion="entropy", min_samples_leaf=5,
                                     random_state=seed, n_jobs=2),
        "SVM": SVC(kernel="linear", C=100, gamma=1e-3, random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=9, weights="uniform", leaf_size=10, p=2),
    }


_lemmatizer = WordNetLemmatizer()
_lemma_cache = {}


def paper_clean(text):
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = []
    for tok in text.split():
        if len(tok) < 2 or tok in STOP:
            continue
        if tok not in _lemma_cache:
            _lemma_cache[tok] = _lemmatizer.lemmatize(tok)
        tokens.append(_lemma_cache[tok])
    return " ".join(tokens)


def load():
    csv.field_size_limit(10_000_000)
    data = []
    cleaned = {}
    for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        cleaned[r["record_id"]] = r["text"]
    for project, filename in PROJECTS.items():
        with (ROOT / "data/raw" / filename).open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                record_id = project + ":" + row["pk"]
                data.append(dict(project=project, label=int(row["type"].strip() == "bug"),
                                 title=paper_clean(row["summary"]),
                                 # full text: our cleaned title + description (None if removed)
                                 full=paper_clean(cleaned[record_id]) if record_id in cleaned else None))
    return data


def one_run(texts, labels, seed):
    idx = np.arange(len(labels))
    train, test = train_test_split(idx, test_size=0.3, random_state=seed, stratify=labels)
    rng = np.random.default_rng(seed)
    pos, neg = train[labels[train] == 1], train[labels[train] == 0]
    n = min(len(pos), len(neg))
    train = np.concatenate([rng.choice(pos, n, replace=False), rng.choice(neg, n, replace=False)])
    vec = TfidfVectorizer()
    x_train = vec.fit_transform([texts[i] for i in train])
    x_test = vec.transform([texts[i] for i in test])
    selector = SelectKBest(chi2, k=min(K_FEATURES, x_train.shape[1])).fit(x_train, labels[train])
    x_train, x_test = selector.transform(x_train), selector.transform(x_test)
    out = {}
    for name, model in classifiers(seed).items():
        pred = model.fit(x_train, labels[train]).predict(x_test)
        y = labels[test]
        out[name] = dict(accuracy=accuracy_score(y, pred),
                         bug_f1=f1_score(y, pred, pos_label=1),
                         nonbug_f1=f1_score(y, pred, pos_label=0),
                         macro_f1=f1_score(y, pred, average="macro"),
                         weighted_f1=f1_score(y, pred, average="weighted"))
    return out


def summarise(runs):
    return {name: {m: dict(mean=float(np.mean([r[name][m] for r in runs])),
                           std=float(np.std([r[name][m] for r in runs])))
                   for m in runs[0][name]} for name in runs[0]}


def main():
    start = time.perf_counter()
    data = load()
    print(f"loaded {len(data)} raw reports", flush=True)
    settings = []
    # (setting name, input field, subset of projects)
    settings.append(("rq3-sample-8000", "title", list(PROJECTS)))
    settings.append(("pooled", "title", list(PROJECTS)))
    for p in PROJECTS:
        settings.append((f"project:{p}", "title", [p]))
    settings.append(("pooled", "title+description", list(PROJECTS)))
    results, rows = {}, []
    with threadpool_limits(limits=2):
        for setting, field, projects in settings:
            subset = [d for d in data if d["project"] in projects
                      and (d["title"] if field == "title" else d["full"])]
            texts = [d["title"] if field == "title" else d["full"] for d in subset]
            labels = np.array([d["label"] for d in subset])
            t0 = time.perf_counter()
            if setting.startswith("rq3"):
                # paper RQ3: 1,600 randomly selected reports per project in every run
                by_project = defaultdict(list)
                for i, d in enumerate(subset):
                    by_project[d["project"]].append(i)
                runs = []
                for seed in range(RUNS):
                    rng = np.random.default_rng(1000 + seed)
                    pick = np.concatenate([rng.choice(v, RQ3_PER_PROJECT, replace=False)
                                           for v in by_project.values()])
                    runs.append(one_run([texts[i] for i in pick], labels[pick], seed))
                n_used = RQ3_PER_PROJECT * len(by_project)
            else:
                runs = [one_run(texts, labels, seed) for seed in range(RUNS)]
                n_used = len(labels)
            summary = summarise(runs)
            results[f"{setting}|{field}"] = dict(n=n_used, bug_share=float(labels.mean()),
                                                 summary=summary)
            for name, s in summary.items():
                rows.append(dict(setting=setting, input=field, model=name, n=n_used,
                                 paper_f=PAPER_PYTHON[name],
                                 **{f"{m}_mean": round(v["mean"], 4) for m, v in s.items()},
                                 **{f"{m}_std": round(v["std"], 4) for m, v in s.items()}))
            print(f"{setting:16s} {field:18s} n={n_used:6d} {time.perf_counter() - t0:6.0f}s  " +
                  "  ".join(f"{k}: bugF1 {v['bug_f1']['mean']:.3f} macro {v['macro_f1']['mean']:.3f} "
                            f"w {v['weighted_f1']['mean']:.3f}" for k, v in summary.items()), flush=True)
    # per-project average (alternative reading of the paper's Python row)
    per_project = defaultdict(lambda: defaultdict(list))
    for key, r in results.items():
        if key.startswith("project:"):
            for name, s in r["summary"].items():
                for m, v in s.items():
                    per_project[name][m].append(v["mean"])
    for name, ms in per_project.items():
        rows.append(dict(setting="per-project average", input="title", model=name, n="",
                         paper_f=PAPER_PYTHON[name],
                         **{f"{m}_mean": round(float(np.mean(v)), 4) for m, v in ms.items()}))
    out = ROOT / "results"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "paper_protocol.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out / "paper_protocol.json").write_text(json.dumps(dict(
        paper="Andrade et al., arXiv 2503.00660, Table 4 (Python row)", paper_f_measure=PAPER_PYTHON,
        runs=RUNS, k_features=K_FEATURES, results=results), indent=2), encoding="utf-8")
    print(f"done in {(time.perf_counter() - start) / 60:.1f} min; saved results/paper_protocol.*")


if __name__ == "__main__":
    main()

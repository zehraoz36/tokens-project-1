"""Reproducible CPU pilot. Run preprocess.py before this script."""
import copy
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
SEED = 42
MAX_FEATURES = 5000


def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average="macro")),
                bug_f1=float(f1_score(y, pred, pos_label=1)),
                confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
                per_class=classification_report(y, pred, labels=[0, 1],
                    target_names=["non-bug", "bug"], output_dict=True, zero_division=0))


def torch_lr(x_train, y_train, x_val, y_val):
    """Week 2: linear layer, sigmoid/BCE, SGD; use stable logits-based BCE."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = torch.nn.BCEWithLogitsLoss()
    labels = torch.tensor(y_train, dtype=torch.float32)
    best_score, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for indices in np.array_split(rng.permutation(len(y_train)),
                                      int(np.ceil(len(y_train) / 256))):
            x = torch.from_numpy(x_train[indices].toarray())
            optimizer.zero_grad()
            loss = criterion(model(x).flatten(), labels[indices])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        pred = torch_predict(model, x_val)
        score = float(f1_score(y_val, pred, average="macro"))
        history.append(dict(epoch=epoch, loss=total_loss / len(y_train), val_macro_f1=score))
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 25 == 0:
            print(f"  PyTorch epoch {epoch}: validation Macro-F1={score:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, dict(selected_epoch=best_epoch, trained_epochs=epoch, history=history)


@torch.inference_mode()
def torch_predict(model, matrix):
    model.eval()
    return np.concatenate([(model(torch.from_numpy(matrix[i:i+256].toarray())).flatten()
                            >= 0).numpy().astype(int)
                           for i in range(0, matrix.shape[0], 256)])


def main():
    torch.set_num_threads(2)
    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    all_results = []
    details = dict(seed=SEED, python=platform.python_version(), sklearn=sklearn.__version__,
                   numpy=np.__version__, torch=torch.__version__,
                   positive_class="bug", primary_metric="macro_f1", experiments=[])
    for scenario in ["random", "cross_project"]:
        sets = {s: [r for r in rows if r[scenario + "_split"] == s]
                for s in ["train", "val", "test"]}
        ys = {s: np.array([r["label"] for r in records]) for s, records in sets.items()}
        texts = {s: [r["text"] for r in records] for s, records in sets.items()}
        print(f"{scenario}: " + str({s: len(v) for s, v in sets.items()}), flush=True)
        for name in ["Majority", "Count_NB", "TFIDF_LinearSVM", "TFIDF_TorchLR", "TFIDF_SklearnLR"]:
            start = time.perf_counter()
            vectorizer = (CountVectorizer(max_features=MAX_FEATURES, min_df=2)
                          if name == "Count_NB" else
                          TfidfVectorizer(max_features=MAX_FEATURES, min_df=2,
                                          ngram_range=(1, 2), dtype=np.float32))
            x = {"train": vectorizer.fit_transform(texts["train"])}
            x.update({s: vectorizer.transform(texts[s]) for s in ["val", "test"]})
            extra = {}
            with threadpool_limits(limits=2):
                if name == "TFIDF_TorchLR":
                    model, extra = torch_lr(x["train"], ys["train"], x["val"], ys["val"])
                    predict = lambda m: torch_predict(model, m)
                else:
                    models = {"Majority": lambda: DummyClassifier(strategy="most_frequent"),
                              "Count_NB": lambda: MultinomialNB(alpha=1.0),
                              "TFIDF_LinearSVM": lambda: LinearSVC(C=1.0, max_iter=5000,
                                                                   random_state=SEED),
                              "TFIDF_SklearnLR": lambda: LogisticRegression(C=1.0, max_iter=1000,
                                                                            random_state=SEED)}
                    model = models[name]()
                    model.fit(x["train"], ys["train"])
                    predict = model.predict
                val = metrics(ys["val"], predict(x["val"]))
                test_pred = predict(x["test"])
                test = metrics(ys["test"], test_pred)
            seconds = time.perf_counter() - start
            result = dict(scenario=scenario, model=name, n_train=len(ys["train"]),
                          n_val=len(ys["val"]), n_test=len(ys["test"]),
                          val_macro_f1=val["macro_f1"], accuracy=test["accuracy"],
                          macro_f1=test["macro_f1"], bug_f1=test["bug_f1"],
                          seconds=round(seconds, 2))
            all_results.append(result)
            details["experiments"].append(dict(**result, validation=val, test=test,
                                                 vocabulary_size=len(vectorizer.vocabulary_), **extra))
            (out / "metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
            print(f"{name}: accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}, "
                  f"{seconds:.1f}s", flush=True)
    for filename, values in [("benchmark.csv", all_results)]:
        with (out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)


if __name__ == "__main__":
    main()

"""Demo: classify a GitHub issue as bug / non-bug.

Two back-ends:
  classical (default)  TF-IDF word + character n-grams + logistic regression, trained
                       on the random-split training set in about a minute on CPU and
                       cached in models/. Also shows the words that pushed the decision.
  transformer          a fine-tuned model saved by
                       `python train_transformer.py --save-model models/deberta`

Examples
  python predict.py "TypeError when calling df.groupby with an empty list"
  python predict.py --title "Add support for zstd compression" --body "It would be nice..."
  python predict.py --interactive
  python predict.py --model models/deberta "Segfault in np.linalg.svd on large matrices"
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "models" / "classical_demo.pkl"


def first_3000(text):
    """Character n-grams are built from the first 3,000 characters (as in experiments.py)."""
    return text[:3000]


def train_classical():
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    train = [r for r in rows if r["random_split"] == "train"]
    texts, labels = [r["text"] for r in train], np.array([r["label"] for r in train])
    word = TfidfVectorizer(max_features=50_000, min_df=2, ngram_range=(1, 2), sublinear_tf=True)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=100_000,
                           sublinear_tf=True, preprocessor=first_3000)
    x = hstack([word.fit_transform(texts), char.fit_transform(texts)]).tocsr()
    model = LogisticRegression(C=8, max_iter=3000, class_weight="balanced", random_state=42).fit(x, labels)
    CACHE.parent.mkdir(exist_ok=True)
    with CACHE.open("wb") as f:
        pickle.dump(dict(word=word, char=char, model=model), f)
    return dict(word=word, char=char, model=model)


class Classical:
    def __init__(self):
        if CACHE.exists():
            with CACHE.open("rb") as f:
                self.p = pickle.load(f)
        else:
            print("Training the demo model once (about a minute)...", file=sys.stderr)
            self.p = train_classical()

    def __call__(self, text):
        from scipy.sparse import hstack
        w = self.p["word"].transform([text])
        x = hstack([w, self.p["char"].transform([text])]).tocsr()
        prob = float(self.p["model"].predict_proba(x)[0, 1])
        # word-level evidence: contribution = tf-idf weight x coefficient
        coef = self.p["model"].coef_[0][: w.shape[1]]
        names = self.p["word"].get_feature_names_out()
        contrib = {names[j]: float(w[0, j] * coef[j]) for j in w.nonzero()[1]}
        top = sorted(contrib.items(), key=lambda kv: kv[1])
        return prob, dict(towards_bug=[k for k, v in top[::-1][:5] if v > 0],
                          towards_non_bug=[k for k, v in top[:5] if v < 0])


class Transformer:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()

    def __call__(self, text):
        with self.torch.inference_mode():
            enc = self.tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
            prob = float(self.torch.softmax(self.model(**enc).logits, dim=-1)[0, 1])
        return prob, {}


def show(predictor, title, body=""):
    text, _ = clean_text(title, body)  # the same cleaning as the training data
    prob, why = predictor(text)
    label = "BUG" if prob >= 0.5 else "NON-BUG"
    print(f"\n{label}  (P(bug) = {prob:.2f})   {title[:80]}")
    for k, v in why.items():
        if v:
            print(f"  words {k.replace('_', ' ')}: {', '.join(v)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("text", nargs="?", help="issue title (and body) as one string")
    parser.add_argument("--title")
    parser.add_argument("--body", default="")
    parser.add_argument("--model", default="classical", help='"classical" or a saved transformer folder')
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    predictor = Classical() if args.model == "classical" else Transformer(args.model)
    if args.interactive:
        print("Type an issue title (empty line to quit).")
        while True:
            title = input("\ntitle> ").strip()
            if not title:
                break
            show(predictor, title, input("body (optional)> ").strip())
    elif args.title or args.text:
        show(predictor, args.title or args.text, args.body)
    else:
        for title in ["TypeError when calling df.groupby with an empty list",
                      "Add support for zstd compression in to_parquet",
                      "Improve the documentation of the axis argument in np.concatenate",
                      "Segfault in scipy.linalg.svd for very large matrices"]:
            show(predictor, title)


if __name__ == "__main__":
    main()

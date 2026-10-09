"""Week 3 neural baselines: TextCNN and BiLSTM on GloVe word embeddings.

Uses the same cleaned data, fixed splits, seed and metrics() function as
benchmark.py, so the scores are directly comparable. The epoch is chosen on
validation Macro-F1; the test set is scored once with that checkpoint.

Runs on CPU (slow, about 20-40 min per model) or GPU (Colab T4: a few minutes).
GloVe (glove.6B.100d, Wikipedia + Gigaword, 400k words) is downloaded on first use
into .download-cache/ (about 820 MB zip); pass --embeddings random to skip it.

Examples
  python train_neural.py --model cnn
  python train_neural.py --model bilstm
  python train_neural.py --model cnn --limit 300 --epochs 1     # quick check, saves nothing

Outputs (one file per model, so CNN and BiLSTM runs do not overwrite each other)
  results/neural_<model>_benchmark.csv, results/neural_<model>_metrics.json,
  results/neural_<model>_predictions_<scenario>.csv
"""
import argparse
import copy
import csv
import json
import platform
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from benchmark import SEED, metrics

ROOT = Path(__file__).resolve().parent
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
GLOVE_URLS = ["https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
              "https://nlp.stanford.edu/data/glove.6B.zip"]
PAD, UNK = 0, 1


def tokenize(text):
    return TOKEN.findall(text)


def build_vocab(texts, min_freq=2, max_size=50_000):
    counts = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}


def encode(texts, vocab, max_len):
    ids = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        seq = [vocab.get(tok, UNK) for tok in tokenize(t)[:max_len]]
        ids[i, :len(seq)] = seq
    return torch.from_numpy(ids)


def load_glove(vocab, dim):
    cache = ROOT / ".download-cache"
    cache.mkdir(exist_ok=True)
    target = cache / f"glove.6B.{dim}d.txt"
    if not target.exists():
        archive = cache / "glove.6B.zip"
        if not archive.exists():
            for url in GLOVE_URLS:
                try:
                    print(f"Downloading GloVe from {url} (about 820 MB)...", flush=True)
                    urllib.request.urlretrieve(url, archive.with_suffix(".partial"))
                    archive.with_suffix(".partial").replace(archive)
                    break
                except Exception as error:  # try the next mirror
                    print(f"  failed: {error}", flush=True)
            else:
                raise SystemExit("Could not download GloVe; use --embeddings random")
        with zipfile.ZipFile(archive) as z:
            target.write_bytes(z.read(target.name))
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, dim)).astype(np.float32)
    matrix[PAD] = 0
    found = 0
    with target.open(encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            if word in vocab:
                matrix[vocab[word]] = np.asarray(values, dtype=np.float32)
                found += 1
    print(f"GloVe coverage: {found:,} of {len(vocab):,} vocabulary words", flush=True)
    return torch.from_numpy(matrix), found


class TextCNN(nn.Module):
    """Kim (2014): parallel convolutions of widths 3, 4, 5, max-pooled over time."""

    def __init__(self, embeddings, filters=100, widths=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embeddings.shape[1], filters, w, padding=w // 2)
                                   for w in widths)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(filters * len(widths), 2)

    def forward(self, ids):
        x = self.embed(ids).transpose(1, 2)                      # batch, dim, time
        mask = (ids != PAD).unsqueeze(1)
        pooled = [torch.relu(conv(x))[..., :ids.shape[1]].masked_fill(~mask, -1e4).amax(dim=2)
                  for conv in self.convs]
        return self.out(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    """Bidirectional LSTM; max- and mean-pooling over the non-padding time steps."""

    def __init__(self, embeddings, hidden=128, dropout=0.5):
        super().__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=PAD)
        self.lstm = nn.LSTM(embeddings.shape[1], hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(4 * hidden, 2)

    def forward(self, ids):
        lengths = (ids != PAD).sum(dim=1).clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(self.dropout(self.embed(ids)), lengths.cpu(),
                                                   batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     total_length=ids.shape[1])
        mask = (ids != PAD).unsqueeze(2)
        maxed = output.masked_fill(~mask, -1e4).amax(dim=1)
        mean = (output * mask).sum(dim=1) / lengths.unsqueeze(1)
        return self.out(self.dropout(torch.cat([maxed, mean], dim=1)))


@torch.inference_mode()
def predict(model, ids, device, batch_size=256):
    model.eval()
    return np.concatenate([model(ids[i:i + batch_size].to(device)).argmax(dim=1).cpu().numpy()
                           for i in range(0, len(ids), batch_size)])


def run_scenario(scenario, rows, args, device):
    sets = {s: [r for r in rows if r[scenario + "_split"] == s] for s in ["train", "val", "test"]}
    if args.limit:
        sets = {s: random.Random(SEED).sample(v, min(args.limit, len(v))) for s, v in sets.items()}
    texts = {s: [r["text"] for r in v] for s, v in sets.items()}
    y = {s: np.array([r["label"] for r in v]) for s, v in sets.items()}
    print(f"{scenario}: " + str({s: len(v) for s, v in sets.items()}), flush=True)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    start = time.perf_counter()
    vocab = build_vocab(texts["train"])
    if args.embeddings == "glove":
        embeddings, covered = load_glove(vocab, args.dim)
    else:
        embeddings = torch.randn(len(vocab) + 2, args.dim) * 0.1
        embeddings[PAD] = 0
        covered = 0
    ids = {s: encode(texts[s], vocab, args.max_length) for s in texts}
    model = (TextCNN(embeddings) if args.model == "cnn" else BiLSTM(embeddings)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    labels = torch.from_numpy(y["train"])
    generator = torch.Generator().manual_seed(SEED)
    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        order = torch.randperm(len(labels), generator=generator)
        for i in range(0, len(order), args.batch_size):
            batch = order[i:i + args.batch_size]
            optimizer.zero_grad()
            loss = criterion(model(ids["train"][batch].to(device)), labels[batch].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        validation = metrics(y["val"], predict(model, ids["val"], device))
        history.append(dict(epoch=epoch, loss=total / len(labels), val_macro_f1=validation["macro_f1"]))
        print(f"  epoch {epoch}: loss={total / len(labels):.4f} "
              f"validation Macro-F1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_score:
            best_score, best_epoch, best_validation = validation["macro_f1"], epoch, validation
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)  # epoch chosen on validation, not on test
    test_pred = predict(model, ids["test"], device)
    test = metrics(y["test"], test_pred)
    seconds = round(time.perf_counter() - start, 2)
    name = f"{args.model.upper() if args.model == 'cnn' else 'BiLSTM'}_{args.embeddings}{args.dim}"
    if not args.limit:
        path = ROOT / "results" / f"neural_{args.model}_predictions_{scenario}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["record_id", "true_label", "predicted_label", "model"])
            for r, p in zip(sets["test"], test_pred):
                writer.writerow([r["record_id"], r["label"], int(p), name])
    result = dict(scenario=scenario, model=name, n_train=len(y["train"]), n_val=len(y["val"]),
                  n_test=len(y["test"]), val_macro_f1=best_score, accuracy=test["accuracy"],
                  macro_f1=test["macro_f1"], bug_f1=test["bug_f1"], seconds=seconds)
    print(f"{scenario}: accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}, "
          f"{seconds / 60:.1f} min", flush=True)
    return result, dict(**result, selected_epoch=best_epoch, history=history, vocabulary=len(vocab),
                        glove_covered=covered, validation=best_validation, test=test)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["cnn", "bilstm"], required=True)
    parser.add_argument("--embeddings", choices=["glove", "random"], default="glove")
    parser.add_argument("--dim", type=int, default=100, choices=[50, 100, 200, 300])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=256, help="tokens per report (as for BERT)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--limit", type=int, help="quick check: records per split; saves nothing")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    results, experiments = [], []
    for scenario in ["random", "cross_project"]:
        result, experiment = run_scenario(scenario, rows, args, device)
        results.append(result)
        experiments.append(experiment)
    if args.limit:
        print("Quick check finished; results were not saved.")
        return
    out = ROOT / "results"
    with (out / f"neural_{args.model}_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    details = dict(seed=SEED, model=args.model, embeddings=args.embeddings, dim=args.dim,
                   epochs=args.epochs, max_length=args.max_length, batch_size=args.batch_size,
                   learning_rate=args.lr, device=str(device), python=platform.python_version(),
                   torch=torch.__version__, positive_class="bug", primary_metric="macro_f1",
                   experiments=experiments)
    (out / f"neural_{args.model}_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"Saved results/neural_{args.model}_benchmark.csv and results/neural_{args.model}_metrics.json")


if __name__ == "__main__":
    main()

"""Figures for the README and the presentation. CPU only, a few seconds.

Reads whatever result files exist in results/ and writes PNGs to results/figures/:
  model_comparison.png    Macro-F1 of every model, random vs cross-project
  confusion_matrices.png  best transformer, both scenarios (row-normalised, with counts)
  class_distribution.png  bug / non-bug reports per project
  error_cues.png          how misleading wording drives the best model's errors
  paper_comparison.png    published F-measure vs our replication of the paper's protocol
Run after benchmark.py / experiments.py / error_analysis.py / paper_protocol.py.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = RES / "figures"

# Validated categorical palette (first three slots pass all-pairs CVD checks), light surface.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
FAMILY_COLOR = {"baseline": GRAY, "classical": BLUE, "neural": AQUA, "transformer": ORANGE}
FAMILY_LABEL = {"baseline": "Majority baseline", "classical": "Classical ML",
                "neural": "Neural (GloVe)", "transformer": "Transformer"}
SCENARIO_TITLE = {"random": "Random split", "cross_project": "Unseen project (scipy)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
})


def read_csv(name):
    path = RES / name
    return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []


NAMES = {"Majority": "Majority class", "Count_NB": "Naive Bayes (counts)",
         "TFIDF_LinearSVM": "Linear SVM (TF-IDF)", "TFIDF_TorchLR": "Log. regression (PyTorch)",
         "TFIDF_SklearnLR": "Log. regression (TF-IDF)", "bert-base-uncased": "BERT-base",
         "deberta-v3-small": "DeBERTa-v3-small"}


def collect_models():
    rows = []
    for r in read_csv("benchmark.csv"):
        rows.append((r["scenario"], NAMES[r["model"]], float(r["macro_f1"]),
                     "baseline" if r["model"] == "Majority" else "classical"))
    # best tuned classical model from experiments.py, chosen on validation
    exp = read_csv("experiments.csv")
    for scenario in ["random", "cross_project"]:
        cands = [r for r in exp if r["scenario"] == scenario]
        if cands:
            best = max(cands, key=lambda r: float(r["val_macro_f1"]))
            rows.append((scenario, f"Best tuned classical*", float(best["macro_f1"]), "classical"))
    for name in ["neural_cnn_benchmark.csv", "neural_bilstm_benchmark.csv"]:
        for r in read_csv(name):
            label = "TextCNN (GloVe)" if "cnn" in name else "BiLSTM (GloVe)"
            rows.append((r["scenario"], label, float(r["macro_f1"]), "neural"))
    for name in ["bert_benchmark.csv", "transformer_benchmark.csv"]:
        for r in read_csv(name):
            rows.append((r["scenario"], NAMES.get(r["model"], r["model"]), float(r["macro_f1"]),
                         "transformer"))
    return rows


def model_comparison():
    rows = collect_models()
    order = sorted({r[1] for r in rows if r[0] == "random"},
                   key=lambda m: next(r[2] for r in rows if r[0] == "random" and r[1] == m))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(order) + 1.6), sharey=True)
    for ax, scenario in zip(axes, ["random", "cross_project"]):
        values = {r[1]: (r[2], r[3]) for r in rows if r[0] == scenario}
        y = np.arange(len(order))
        for i, m in enumerate(order):
            if m not in values:
                continue
            v, fam = values[m]
            ax.barh(i, v * 100, height=0.62, color=FAMILY_COLOR[fam], edgecolor=SURFACE, linewidth=2)
            ax.text(v * 100 + 0.8, i, f"{v * 100:.1f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, order)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Macro-F1 on the test set (%)")
        ax.set_title(SCENARIO_TITLE[scenario])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    families = [f for f in FAMILY_COLOR if any(r[3] == f for r in rows)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in families]
    fig.legend(handles, [FAMILY_LABEL[f] for f in families], loc="upper center", ncol=len(families),
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    note = "* best of experiments.py, selected on validation Macro-F1" if any(
        r[1].startswith("Best tuned") for r in rows) else ""
    fig.text(0.01, 0.005, note, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(FIG / "model_comparison.png", dpi=200)
    plt.close(fig)


def confusion_matrices():
    path = RES / "transformer_metrics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    cmap = LinearSegmentedColormap.from_list("blue", ["#eef4fc", BLUE, "#0d3a73"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, exp in zip(axes, data["experiments"]):
        cm = np.array(exp["test"]["confusion_matrix"])
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{norm[i, j] * 100:.1f}%\n({cm[i, j]:,})", ha="center", va="center",
                        fontsize=10, color="white" if norm[i, j] > 0.5 else INK)
        ax.set_xticks([0, 1], ["non-bug", "bug"])
        ax.set_yticks([0, 1], ["non-bug", "bug"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{SCENARIO_TITLE[exp['scenario']]}  (Macro-F1 {exp['macro_f1'] * 100:.1f})")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{data['model'].split('/')[-1]}: test confusion matrices (row-normalised)",
                 x=0.01, ha="left", fontweight="bold", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices.png", dpi=200)
    plt.close(fig)


def class_distribution():
    counts = Counter()
    for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        counts[(r["project"], r["label"])] += 1
    projects = sorted({p for p, _ in counts}, key=lambda p: -(counts[(p, 0)] + counts[(p, 1)]))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(projects))
    bug = np.array([counts[(p, 1)] for p in projects])
    non = np.array([counts[(p, 0)] for p in projects])
    ax.barh(y, bug, height=0.6, color=BLUE, edgecolor=SURFACE, linewidth=2, label="bug")
    ax.barh(y, non, left=bug, height=0.6, color=ORANGE, edgecolor=SURFACE, linewidth=2, label="non-bug")
    for i, p in enumerate(projects):
        total = bug[i] + non[i]
        ax.text(total + 120, i, f"{total:,}  ({bug[i] / total * 100:.0f}% bug)", va="center", fontsize=9)
    ax.set_yticks(y, projects)
    ax.invert_yaxis()
    ax.set_xlim(0, max(bug + non) * 1.32)
    ax.set_xlabel("Reports after cleaning")
    ax.set_title(f"Class distribution: {bug.sum() + non.sum():,} reports, "
                 f"{bug.sum() / (bug.sum() + non.sum()) * 100:.0f}% bug")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=200)
    plt.close(fig)


def error_cues():
    path = RES / "error_analysis.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())["random"]
    model = next(m for m in ["bert", "deberta", "classical"] if m in data)  # matches the README text
    cues = data[model]["cues"]
    groups = [("Non-bug reports\nmentioning error words", "non-bug", "error / exception words"),
              ("Bug reports mentioning\nrequest / docs words", "bug", "request / docs words"),
              ("Non-bug reports filed with\nthe bug-report template", "non-bug", "bug-report template")]
    groups = [g for g in groups if f"{g[1]} WITH {g[2]}" in cues]
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(groups))
    w = 0.36
    for k, (key, color, label) in enumerate([("WITHOUT", GRAY, "without the cue"),
                                             ("WITH", ORANGE, "with the cue")]):
        vals = [cues[f"{cls} {key} {cue}"]["error_rate"] * 100 for _, cls, cue in groups]
        ns = [cues[f"{cls} {key} {cue}"]["n"] for _, cls, cue in groups]
        bars = ax.bar(x + (k - 0.5) * w, vals, width=w, color=color, edgecolor=SURFACE,
                      linewidth=2, label=label)
        for b, v, n in zip(bars, vals, ns):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.0f}%\nn={n:,}", ha="center",
                    va="bottom", fontsize=8.5)
    ax.set_xticks(x, [g[0] for g in groups])
    ax.set_ylabel("Error rate (%)")
    ax.set_ylim(0, max(cues[f"{c} WITH {q}"]["error_rate"] for _, c, q in groups) * 100 * 1.35)
    name = {"deberta": "DeBERTa-v3-small", "bert": "BERT-base", "classical": "best classical"}[model]
    ax.set_title(f"Misleading wording drives the errors ({name}, random split)", pad=26)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "error_cues.png", dpi=200)
    plt.close(fig)


def paper_comparison():
    path = RES / "paper_protocol.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    paper = data["paper_f_measure"]
    res = data["results"]
    models = list(paper)
    series = [("Published F-measure (paper, Table 4, Python)", GRAY, [paper[m] for m in models])]
    per_project = [k for k in res if k.startswith("project:")]
    for metric, label, color in [("nonbug_f1", "Our replication: non-bug F1", BLUE),
                                 ("macro_f1", "Our replication: Macro-F1", AQUA)]:
        if per_project:
            series.append((label, color, [np.mean([res[k]["summary"][m][metric]["mean"]
                                                   for k in per_project]) for m in models]))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(models))
    w = 0.8 / len(series)
    for k, (label, color, vals) in enumerate(series):
        bars = ax.bar(x + (k - (len(series) - 1) / 2) * w, np.array(vals) * 100, width=w, color=color,
                      edgecolor=SURFACE, linewidth=2, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6, f"{v * 100:.0f}", ha="center",
                    va="bottom", fontsize=8)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Classifier (paper's tuned settings), per-project runs averaged over the 5 projects")
    ax.set_ylabel("F-measure (%)")
    ax.set_title("Benchmark: source paper's protocol re-run on our five projects (titles, 30 runs)")
    ax.legend(frameon=False, loc="upper right", fontsize=8.5, ncol=1)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(FIG / "paper_comparison.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for f in [model_comparison, confusion_matrices, class_distribution, error_cues, paper_comparison]:
        f()
    print("Saved:", ", ".join(sorted(p.name for p in FIG.glob("*.png"))))


if __name__ == "__main__":
    main()

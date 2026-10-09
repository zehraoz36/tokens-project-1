"""Fine-tune a pretrained transformer on the bug / non-bug task (the "new" method).

Uses the cleaned data and the two fixed splits created by preprocess.py and the
same metrics as benchmark.py, so the scores are directly comparable with the
baselines. Needs a GPU: run it on Google Colab (see README).
"""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

from benchmark import SEED, metrics

ROOT = Path(__file__).resolve().parent


def load_scenario(rows, scenario, limit=None):
    """Texts and labels of the train/val/test parts of one experiment."""
    data = {}
    for split in ["train", "val", "test"]:
        subset = [r for r in rows if r[scenario + "_split"] == split]
        if limit:
            subset = random.Random(SEED).sample(subset, min(limit, len(subset)))
        data[split] = ([r["text"] for r in subset], np.array([r["label"] for r in subset]))
    return data


def encode(tokenizer, texts, labels, max_length):
    """Tokenize and keep only the first max_length tokens; padding is added per batch."""
    encoded = tokenizer(texts, truncation=True, max_length=max_length)
    return [dict({key: values[i] for key, values in encoded.items()}, labels=int(labels[i]))
            for i in range(len(texts))]


@torch.inference_mode()
def predict(model, loader, device, amp):
    model.eval()
    predictions = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items() if key != "labels"}
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(**batch).logits
        predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def run_scenario(scenario, data, args, device):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collate = DataCollatorWithPadding(tokenizer)
    items = {split: encode(tokenizer, texts, labels, args.max_length)
             for split, (texts, labels) in data.items()}
    loaders = {
        "train": DataLoader(items["train"], batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate, generator=torch.Generator().manual_seed(SEED)),
        "val": DataLoader(items["val"], batch_size=4 * args.batch_size, collate_fn=collate),
        "test": DataLoader(items["test"], batch_size=4 * args.batch_size, collate_fn=collate),
    }
    # The published checkpoint stores half-precision weights; train in float32.
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model = model.float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loaders["train"]) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp)

    best_score, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loaders["train"], start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device.type, dtype=torch.float16, enabled=args.amp):
                loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Loss is not finite. Run again with --no-amp.")
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item() * len(batch["labels"])
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(loaders['train'])} "
                      f"loss={total_loss / (step * args.batch_size):.4f}", flush=True)
        validation = metrics(data["val"][1], predict(model, loaders["val"], device, args.amp))
        history.append(dict(epoch=epoch, loss=total_loss / len(items["train"]),
                            val_macro_f1=validation["macro_f1"]))
        print(f"  epoch {epoch}: validation Macro-F1={validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_score:
            best_score, best_epoch, best_validation = validation["macro_f1"], epoch, validation
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)  # the epoch chosen on validation, not on test
    test_predictions = predict(model, loaders["test"], device, args.amp)
    test = metrics(data["test"][1], test_predictions)

    if not args.limit:
        # Same export format as train_bert.py, for error analysis (error_analysis.py)
        import hashlib
        prediction_path = ROOT / "results" / f"transformer_predictions_{scenario}.csv"
        prediction_path.parent.mkdir(exist_ok=True)
        with prediction_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["text_sha256", "true_label", "predicted_label"])
            for text, label, prediction in zip(data["test"][0], data["test"][1], test_predictions):
                writer.writerow([hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 int(label), int(prediction)])

    if args.save_model and scenario == "random" and not args.limit:
        # selected checkpoint of the random-split model, for predict.py --model <folder>
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
        print(f"Saved the selected random-split model to {args.save_model}", flush=True)

    seconds = time.perf_counter() - start
    result = dict(scenario=scenario, model=args.model.split("/")[-1],
                  n_train=len(items["train"]), n_val=len(items["val"]), n_test=len(items["test"]),
                  val_macro_f1=best_score, accuracy=test["accuracy"], macro_f1=test["macro_f1"],
                  bug_f1=test["bug_f1"], seconds=round(seconds, 2))
    print(f"{scenario}: accuracy={test['accuracy']:.4f}, macro-F1={test['macro_f1']:.4f}, "
          f"{seconds / 60:.1f} min", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result, dict(**result, selected_epoch=best_epoch, history=history,
                        validation=best_validation, test=test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--no-amp", dest="amp", action="store_false",
                        help="Disable mixed precision (slower, use if the loss is not finite)")
    parser.add_argument("--save-model", metavar="FOLDER",
                        help="Also save the selected random-split model (for the predict.py demo)")
    parser.add_argument("--limit", type=int,
                        help="Quick check: use this many records per split and do not save results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.amp = args.amp and device.type == "cuda"
    if device.type == "cpu" and not args.limit:
        raise SystemExit("No GPU found. In Colab choose Runtime > Change runtime type > T4 GPU, "
                         "or pass --limit 200 for a quick CPU check.")
    rows = [json.loads(line) for line in (ROOT / "data/cleaned.jsonl").open(encoding="utf-8")]
    results, experiments = [], []
    for scenario in ["random", "cross_project"]:
        data = load_scenario(rows, scenario, args.limit)
        print(f"{scenario}: " + str({split: len(texts) for split, (texts, _) in data.items()}),
              flush=True)
        result, experiment = run_scenario(scenario, data, args, device)
        results.append(result)
        experiments.append(experiment)
    if args.limit:
        print("Quick check finished; results were not saved.")
        return

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with (out / "transformer_benchmark.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    details = dict(seed=SEED, model=args.model, epochs=args.epochs, max_length=args.max_length,
                   batch_size=args.batch_size, learning_rate=args.lr, mixed_precision=args.amp,
                   device=torch.cuda.get_device_name(0), python=platform.python_version(),
                   torch=torch.__version__, transformers=transformers.__version__,
                   sklearn=sklearn.__version__, positive_class="bug", primary_metric="macro_f1",
                   experiments=experiments)
    (out / "transformer_metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print("Saved results/transformer_benchmark.csv and results/transformer_metrics.json")


if __name__ == "__main__":
    main()

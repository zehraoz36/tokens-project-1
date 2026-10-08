"""Download/extract five original CSVs, clean text, and create fixed splits."""
import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCE = "https://zenodo.org/records/7377402"
ARCHIVE_MD5 = "23d97199eced083cc169e31967bcf5ed"
PROJECTS = {
    "numpy": "github_numpy_numpy_labeled.csv",
    "pandas": "github_pandasdev_pandas_labeled.csv",
    "salt": "github_saltstack_salt_labeled.csv",
    "scipy": "github_scipy_scipy_labeled.csv",
    "weblate": "github_weblateorg_weblate_labeled.csv",
}
PREFIX = re.compile(r"^\s*(?:\[(?:BUG|ENH|ENHANCEMENT|FEAT|FEATURE|DOC|DOCS|FIX|MAINT|API|TST|BLD|REF|WIP)[^\]]*\]|(?:BUG|ENH|ENHANCEMENT|FEAT|FEATURE|DOC|DOCS|FIX|MAINT|API|TST|BLD|REF|WIP)(?:/[A-Z]+)*\s*:)\s*", re.I)


def clean_text(summary, description):
    changes = Counter()
    title = summary
    while PREFIX.match(title):
        title = PREFIX.sub("", title, count=1)
        changes["title_prefixes_removed"] += 1
    text = title + "\n" + description
    operations = [
        ("html_comments", r"<!--[\s\S]*?-->", " "),
        ("urls", r"https?://[^\s<>]+", " URL "),
        ("html_tags", r"</?[A-Za-z][^>\n]*>", " "),
        ("code_fence_markers", r"(?m)^\s*(?:```|~~~)[^\n]*$", " "),
        ("label_template_headings", r"(?im)^\s*#{1,6}\s*(?:bug report|feature request|enhancement request)\s*$", " "),
        ("markdown_heading_markers", r"(?m)^\s*#{1,6}\s+", " "),
    ]
    for name, pattern, replacement in operations:
        text, count = re.subn(pattern, replacement, text)
        changes[name] += count
    decoded = html.unescape(text)
    changes["html_entities_decoded"] = int(decoded != text)
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", decoded)).strip().lower()
    if len(text) > 20000:
        changes["long_texts_truncated"] += 1
        text = text[:20000]
    return text, changes


def prepare_raw(archive):
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    if archive is None and all((raw / name).exists() for name in PROJECTS.values()):
        return
    if archive is None:
        archive = ROOT / ".download-cache/Datasets.zip"
        archive.parent.mkdir(exist_ok=True)
        if not archive.exists():
            print("Downloading the official 207 MB archive...", flush=True)
            temporary = archive.with_suffix(".partial")
            urllib.request.urlretrieve(SOURCE + "/files/Datasets.zip?download=1", temporary)
            temporary.replace(archive)
    digest = hashlib.md5(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_MD5:
        raise ValueError(f"Archive checksum mismatch: {digest}")
    with zipfile.ZipFile(archive) as z:
        for filename in PROJECTS.values():
            (raw / filename).write_bytes(z.read("Datasets/" + filename))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Optional existing official Datasets.zip")
    args = parser.parse_args()
    prepare_raw(args.archive)
    csv.field_size_limit(10_000_000)
    raw_counts, labels, transformations, affected = Counter(), Counter(), Counter(), Counter()
    records, hashes = [], {}
    for project, filename in PROJECTS.items():
        path = ROOT / "data/raw" / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["pk", "summary", "description", "type"]:
                raise ValueError(f"Unexpected columns in {filename}: {reader.fieldnames}")
            for row in reader:
                raw_counts[project] += 1
                label = row["type"].strip()
                if label not in ["bug", "non-bug"]:
                    raise ValueError(f"Unexpected label: {label}")
                labels[label] += 1
                affected["empty_description"] += int(not row["description"].strip())
                text, counts = clean_text(row["summary"], row["description"])
                transformations.update(counts)
                affected.update({key: 1 for key, value in counts.items() if value})
                records.append(dict(record_id=project + ":" + row["pk"], project=project,
                                    label=int(label == "bug"), text=text,
                                    text_sha256=hashlib.sha256(text.encode()).hexdigest()))
    if len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Source IDs are not unique within projects")
    grouped = defaultdict(list)
    empty = 0
    for row in records:
        if not re.search(r"[a-z]", row["text"]):
            empty += 1
        else:
            grouped[row["text_sha256"]].append(row)
    cleaned, conflicts, duplicates = [], 0, 0
    for group in grouped.values():
        if len({r["label"] for r in group}) > 1:
            conflicts += len(group)
        else:
            cleaned.append(group[0])
            duplicates += len(group) - 1
    cleaned.sort(key=lambda r: r["record_id"])
    train_val, test = train_test_split(list(range(len(cleaned))), test_size=0.2,
                                     stratify=[r["label"] for r in cleaned], random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.125,
                                 stratify=[cleaned[i]["label"] for i in train_val], random_state=SEED)
    for split, indices in [("train", train), ("val", val), ("test", test)]:
        for i in indices:
            cleaned[i]["random_split"] = split
    for row in cleaned:
        row["cross_project_split"] = {"scipy": "test", "weblate": "val"}.get(row["project"], "train")
    split_counts = {}
    for scenario in ["random", "cross_project"]:
        split_counts[scenario] = {}
        for split in ["train", "val", "test"]:
            subset = [r for r in cleaned if r[scenario + "_split"] == split]
            split_counts[scenario][split] = dict(n=len(subset),
                classes=dict(Counter("bug" if r["label"] else "non-bug" for r in subset)),
                projects=dict(Counter(r["project"] for r in subset)))
    with (ROOT / "data/cleaned.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in cleaned:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = dict(source=SOURCE, source_archive_md5=ARCHIVE_MD5, source_files_sha256=hashes,
                 license="CC-BY-4.0", seed=SEED, raw_count=len(records), raw_projects=dict(raw_counts),
                 raw_labels=dict(labels), clean_count=len(cleaned),
                 removed=dict(empty_text=empty, duplicate_text=duplicates, conflicting_label_rows=conflicts),
                 transformation_occurrences=dict(transformations), affected_rows=dict(affected), splits=split_counts)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    (results / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

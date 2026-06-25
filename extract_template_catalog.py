#!/usr/bin/env python3
"""
Scans the FULL candidate pool (not just your submission) and extracts every
distinct career_history description string, with how many candidates use it
and a few sample (candidate_id, title) pairs per template.

If the dataset truly is built from a template library, this will show a
manageable number of unique blocks (likely dozens to low hundreds) instead
of 100,000 unique freeform paragraphs. That number tells you whether it's
worth hand-labeling each template once (much more reliable than re-deriving
a score from keywords per candidate every run).

Usage:
    python extract_template_catalog.py --candidates ./candidates.json --out ./template_catalog.csv
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict


def iter_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        first_line = f.readline()
        f.seek(0)
        try:
            json.loads(first_line)
            is_jsonl = True
        except (ValueError, json.JSONDecodeError):
            is_jsonl = False
        if is_jsonl:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        else:
            data = json.load(f)
            records = data.get("candidates", data) if isinstance(data, dict) else data
            for cand in records:
                yield cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out", default="template_catalog.csv")
    ap.add_argument("--field", choices=["description", "summary"], default="description",
                     help="Which text field to catalog: career_history descriptions, or profile summaries")
    args = ap.parse_args()

    desc_count = defaultdict(int)
    desc_samples = defaultdict(list)
    n_candidates = 0
    n_total_descs = 0

    for cand in iter_candidates(args.candidates):
        n_candidates += 1
        if args.field == "summary":
            texts = [cand.get("profile", {}).get("summary", "")]
        else:
            texts = [job.get("description", "") for job in cand.get("career_history", [])]

        for text in texts:
            text = text.strip()
            if not text:
                continue
            n_total_descs += 1
            desc_count[text] += 1
            if len(desc_samples[text]) < 3:
                desc_samples[text].append((cand["candidate_id"], cand.get("profile", {}).get("current_title", "")))

    n_unique = len(desc_count)
    print(f"Scanned {n_candidates} candidates, {n_total_descs} total '{args.field}' entries.")
    print(f"Unique '{args.field}' templates found: {n_unique}")
    print(f"Average reuse per template: {n_total_descs / n_unique:.1f}x")
    print()

    rows = sorted(desc_count.items(), key=lambda x: -x[1])
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reuse_count", "sample_candidate_ids_and_titles", "text"])
        for text, count in rows:
            samples = "; ".join(f"{cid} ({title})" for cid, title in desc_samples[text])
            writer.writerow([count, samples, text])

    print(f"Full catalog written to {args.out} — sorted by reuse count, most-reused first.")
    print("Open it and hand-label the top ~50-100 templates (the ones covering most of the pool);")
    print("that labeling effort covers far more candidates than reading individual profiles.")


if __name__ == "__main__":
    main()
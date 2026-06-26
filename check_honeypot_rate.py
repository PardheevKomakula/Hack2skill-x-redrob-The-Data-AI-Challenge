#!/usr/bin/env python3
"""
Validates the honeypot integrity of the final submission.
Checks what % of the top-100 ranked candidates have honeypot flags.
A healthy submission should have 0% honeypot rate.

Usage:
    python check_honeypot_rate.py --candidates ./candidates.json --submission ./submission.csv
"""
import argparse
import json
import csv
import sys


def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if raw.startswith("["):
            return {c.get("candidate_id", c.get("id")): c for c in json.loads(raw)}
        else:
            result = {}
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    c = json.loads(line)
                    result[c.get("candidate_id", c.get("id"))] = c
            return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", required=True)
    args = ap.parse_args()

    from src.honeypot import detect_honeypot_flags

    print("Loading submission IDs...")
    submission_ids = []
    with open(args.submission, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            submission_ids.append(row["candidate_id"])

    print(f"Loading candidate pool ({args.candidates})...")
    candidates = load_candidates(args.candidates)

    print("Checking honeypot flags on all ranked candidates...\n")
    flagged = []
    for cid in submission_ids:
        cand = candidates.get(cid)
        if cand is None:
            continue
        flags = detect_honeypot_flags(cand)
        if flags:
            flagged.append((cid, flags))

    total = len(submission_ids)
    n_flagged = len(flagged)
    rate = 100.0 * n_flagged / total if total > 0 else 0.0

    print("--- HONEYPOT REPORT ---")
    print(f"Submission size : {total}")
    print(f"Flagged         : {n_flagged}")
    print(f"Honeypot Rate   : {rate:.1f}% ({n_flagged}/{total})")

    if n_flagged == 0:
        print("[PASS] 0% honeypot rate -- no flagged candidates in top 100.")
    elif rate <= 5.0:
        print(f"[WARN] {rate:.1f}% honeypot rate -- review flagged candidates.")
        for cid, flags in flagged:
            print(f"   {cid}: {flags}")
    else:
        print(f"[FAIL] {rate:.1f}% honeypot rate exceeds threshold.")
        for cid, flags in flagged:
            print(f"   {cid}: {flags}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pulls full profile detail for specific candidate_ids so you can manually
read them and sanity-check a cluster (e.g. "are these 22 'AI Research
Engineer' candidates genuinely strong, or is that title a keyword trap?").

Usage:
    # Review every candidate in your submission with a given current_title:
    python review_candidates.py --candidates ./candidates.json --submission ./submission.csv --title "AI Research Engineer"

    # Or review specific candidate_ids directly:
    python review_candidates.py --candidates ./candidates.json --ids CAND_0008295 CAND_0011327
"""
import argparse
import csv
import gzip
import json


def load_candidates_by_id(path):
    opener = gzip.open if path.endswith(".gz") else open
    out = {}
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
                    cand = json.loads(line)
                    out[cand["candidate_id"]] = cand
        else:
            data = json.load(f)
            records = data.get("candidates", data) if isinstance(data, dict) else data
            for cand in records:
                out[cand["candidate_id"]] = cand
    return out


def print_candidate(cand):
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    print("=" * 90)
    print(f"{cand['candidate_id']}  |  {p.get('current_title')}  |  {p.get('location')}  |  {p.get('years_of_experience')} yrs")
    print(f"Headline: {p.get('headline')}")
    print(f"Summary: {p.get('summary')}")
    print("-- Career history --")
    for job in cand.get("career_history", []):
        print(f"  [{job.get('start_date')} -> {job.get('end_date') or 'present'}] {job.get('title')} @ {job.get('company')} ({job.get('duration_months')}mo)")
        print(f"    {job.get('description')}")
    print("-- Skills --")
    print("  " + ", ".join(f"{s['name']}({s.get('proficiency')},{s.get('duration_months')}mo)" for s in cand.get("skills", [])))
    print("-- Key signals --")
    print(f"  open_to_work={sig.get('open_to_work_flag')}  notice_days={sig.get('notice_period_days')}  "
          f"recruiter_resp={sig.get('recruiter_response_rate')}  last_active={sig.get('last_active_date')}  "
          f"github_score={sig.get('github_activity_score')}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--submission", help="submission.csv to pull candidate_ids from")
    ap.add_argument("--title", help="Only review submission rows whose current_title matches this (case-insensitive)")
    ap.add_argument("--ids", nargs="*", help="Specific candidate_ids to review directly")
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    by_id = load_candidates_by_id(args.candidates)

    target_ids = []
    if args.ids:
        target_ids = args.ids
    elif args.submission:
        with open(args.submission, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            cand = by_id.get(row["candidate_id"])
            if cand is None:
                continue
            if args.title:
                if args.title.lower() not in (cand.get("profile", {}).get("current_title") or "").lower():
                    continue
            target_ids.append(row["candidate_id"])
    else:
        print("Provide either --ids or --submission (optionally with --title to filter).")
        return

    target_ids = target_ids[: args.limit]
    print(f"Reviewing {len(target_ids)} candidate(s)...\n")
    for cid in target_ids:
        cand = by_id.get(cid)
        if cand:
            print_candidate(cand)
        else:
            print(f"{cid}: not found in candidate file")


if __name__ == "__main__":
    main()
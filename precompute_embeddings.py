#!/usr/bin/env python3
"""
Offline precompute step. NOT part of the 5-minute ranking budget —
run this once (or whenever candidates.jsonl changes) and commit the
resulting .npy files (or regenerate them via this script, documented
in the README's reproduce command).

Usage:
    python precompute_embeddings.py --candidates ./candidates.jsonl.gz
"""
import argparse
import gzip
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

import config as cfg
from features import build_narrative


def load_candidates(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl or .jsonl.gz")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    t0 = time.time()
    model = SentenceTransformer(cfg.EMBEDDING_MODEL_NAME)

    narratives, ids = [], []
    for cand in load_candidates(args.candidates):
        ids.append(cand["candidate_id"])
        narratives.append(build_narrative(cand))

    print(f"Loaded {len(ids)} candidates in {time.time() - t0:.1f}s")

    t1 = time.time()
    embeddings = model.encode(
        narratives,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # so cosine sim = dot product
        convert_to_numpy=True,
    )
    print(f"Encoded embeddings in {time.time() - t1:.1f}s, shape={embeddings.shape}")

    np.save(cfg.EMBEDDINGS_PATH, embeddings.astype(np.float32))
    np.save(cfg.CANDIDATE_IDS_PATH, np.array(ids))
    print(f"Saved to {cfg.EMBEDDINGS_PATH} and {cfg.CANDIDATE_IDS_PATH}")
    print(f"Total precompute time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Offline precompute step. NOT part of the 5-minute ranking budget.
Run this once (or whenever candidates.json changes) to generate
the .npy embedding files used by rank.py.

Usage:
    python precompute_embeddings.py --candidates ./candidates.json
"""
import argparse
import gzip
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from src import config as cfg
from src.features import build_narrative


def load_candidates(path):
    """Auto-detect: .gz → gzip, plain .json → list or JSONL."""
    if path.endswith(".gz"):
        opener = gzip.open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        with open(path, "rt", encoding="utf-8") as f:
            raw = f.read().strip()
            if raw.startswith("["):
                # Standard JSON array
                for item in json.loads(raw):
                    yield item
            else:
                # JSONL format (one JSON object per line)
                for line in raw.splitlines():
                    line = line.strip()
                    if line:
                        yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="Path to candidates.json or .jsonl or .jsonl.gz")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import yaml, os
    config_path = "config/ranking_config.yaml"
    device = "cpu"
    if os.path.exists(config_path):
        with open(config_path) as f:
            yml = yaml.safe_load(f)
        device = yml.get("model", {}).get("device", "cpu")

    t0 = time.time()
    print(f"Loading model '{cfg.EMBEDDING_MODEL_NAME}' on device={device}...")
    model = SentenceTransformer(cfg.EMBEDDING_MODEL_NAME, device=device)

    print("Reading candidate profiles...")
    narratives, ids = [], []
    for cand in load_candidates(args.candidates):
        ids.append(cand.get("candidate_id", cand.get("id")))
        narratives.append(build_narrative(cand))

    print(f"Loaded {len(ids)} candidates in {time.time() - t0:.1f}s")

    t1 = time.time()
    embeddings = model.encode(
        narratives,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine sim = dot product (faster at rank time)
        convert_to_numpy=True,
    )
    print(f"Encoded embeddings in {time.time() - t1:.1f}s, shape={embeddings.shape}")

    import os as _os
    _os.makedirs("data", exist_ok=True)
    np.save(cfg.EMBEDDINGS_PATH, embeddings.astype(np.float32))
    np.save(cfg.CANDIDATE_IDS_PATH, np.array(ids))
    print(f"Saved to {cfg.EMBEDDINGS_PATH} and {cfg.CANDIDATE_IDS_PATH}")
    print(f"Total precompute time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
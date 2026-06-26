# 🏆 ByteBrad's — Redrob AI Candidate Ranking Pipeline
### Hack2skill × Redrob | India Runs 2026 | Data & AI Challenge

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Runtime](https://img.shields.io/badge/Runtime-~54s_on_CPU-brightgreen?style=flat-square)
![Candidates](https://img.shields.io/badge/Candidates-100K-orange?style=flat-square)
![Templates](https://img.shields.io/badge/Templates-44_hand--labeled-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## 🎯 What This Does

A **production-grade, multi-stage AI candidate ranking pipeline** that processes a pool of 100,000 candidate profiles and ranks the top 100 most suitable candidates for a **Senior AI Engineer — Founding Team** role at Redrob AI.

The pipeline goes far beyond keyword matching. It combines **semantic understanding**, **structural fraud detection**, **hand-labeled production evidence scoring**, and **real behavioral engagement signals** — all running in under 60 seconds on a standard CPU laptop.

```bash
python rank.py --candidates ./candidates.json --jd ./job_description.txt --out ./submission.csv
```

---

## 🔑 The Core Insight — The 44-Template Discovery

After running `extract_template_catalog.py` on the full 100K-candidate pool, we discovered:

> **Every single `career_history` description across 100,000 candidates is drawn from exactly 44 unique template strings, reused 2 to ~25,500 times each.**

This meant we could **hand-label all 44 templates once** against the JD's exact criteria — far more reliably than re-deriving scores from keyword heuristics per candidate on every run.

| Tier | Score | Description | Reuse Count |
|------|-------|-------------|-------------|
| **G (Elite)** | 0.95–0.98 | Fine-tuned LLMs, 50M-query RAG pipelines, full ranking system ownership | 2–12 |
| **E–F (Strong)** | 0.60–0.82 | Semantic search, recommendations at scale, RAG deployment | 57–78 |
| **D (Hedged)** | 0.20–0.30 | Plausible ML work with disclaimers ("didn't make it to production") | 328–389 |
| **C (Data-adj.)** | 0.12–0.20 | Data engineering, pipelines, warehouses | 400–1800 |
| **A (Off-domain)** | 0.02–0.05 | Sales, support, marketing, accounting | ~25,000 |

**Key observation:** The rarest templates are the strongest matches. Naive frequency-weighted approaches get this exactly backwards.

---

## 🏗️ Architecture

```
candidates.json (100K profiles, JSONL format)
        │
        ├──► precompute_embeddings.py  ──►  data/candidate_embeddings.npy  (offline, one-time)
        │                                   data/candidate_ids.npy
        │
rank.py  (main pipeline)
        │
        ├─ 1. Load  config/ranking_config.yaml
        ├─ 2. Init  ProductionEmbeddingEngine  (all-MiniLM-L6-v2, CPU)
        ├─ 3. Encode  job_description.txt  →  jd_vector [384-dim]
        ├─ 4. Load  pre-computed candidate_embeddings.npy  [100K × 384]
        ├─ 5. Vectorized cosine similarity  →  semantic_scores [100K]  (NumPy dot product)
        │
        └─ 6. Per-candidate scoring loop:
                │
                ├─ A. Honeypot + Compliance  (src/compliance.py → src/honeypot.py)
                │      Full 7-check delegation: date math, skill duration vs YOE,
                │      expert+no-evidence, seniority mismatch, education sanity,
                │      YOE inflation, duplicate descriptions
                │      0 flags → 1.0x  |  1 flag → 0.35x  |  2+ flags → 0.02x
                │
                ├─ C. Full Rule Score  (src/scoring.py)
                │      0.30 × title_relevance
                │      0.30 × production_evidence  (44-template O(1) lookup)
                │      0.15 × experience_fit       (5–9 yrs = 1.0)
                │      0.15 × location_fit          (Pune/Noida preferred)
                │      0.10 × notice_fit             (≤30 days = 1.0)
                │       ×   anti_pattern_penalty    (consulting/CV/title-chaser)
                │
                ├─ D. Plain-language Boost  (src/embedding_engine.py, max +0.10)
                │
                ├─ E. Behavioral Multiplier  (src/scoring.py, floor 0.35 / ceil 1.15)
                │      recency + open_to_work + response_rate
                │      + interview_completion_rate + verification signals
                │
                └─ F. Final Score
                       = (0.45 × semantic + 0.55 × rule + plain_boost)
                          × behavioral_mult × effective_integrity

        └─ 7. Sort → Top 100 → submission.csv  +  sample_output/
```

---

## 📁 Project Structure

```
INDIA_RUNS-2026/
├── rank.py                      # Main entrypoint — run this
├── precompute_embeddings.py     # Offline embedding generator (run once)
├── check_honeypot_rate.py       # Validates 0% honeypot rate in submission
├── extract_template_catalog.py  # Discovery tool — found the 44 templates
├── review_candidates.py         # Manual QA / audit tool
├── requirements.txt
├── submission_metadata.yaml
│
├── config/
│   └── ranking_config.yaml      # All weights, model config, penalties
│
├── src/                         # Core engine (fully modular)
│   ├── __init__.py
│   ├── config.py                # All constants and keyword lists
│   ├── features.py              # Shared feature extraction helpers
│   ├── embedding_engine.py      # SentenceTransformer wrapper + plain-language boost
│   ├── scoring.py               # Full rule engine (5 dimensions + behavioral)
│   ├── honeypot.py              # 7-check fraud / inconsistency detector
│   ├── compliance.py            # Quick structural integrity checker
│   ├── pipeline.py              # Final score assembly
│   └── template_scores.py      # Hand-labeled 44-template lookup table ⭐
│
├── data/
│   └── template_catalog.csv     # 44 unique templates + reuse counts (analytics)
│
├── sample_output/
│   └── sample_submission.csv    # Top 10 ranked candidates (judge preview)
│
└── tests/
    └── test_pipeline.py         # Unit tests (honeypot trap validation)
```

---

## ⚙️ Scoring Formula

```
base_score  = (0.45 × semantic_similarity) + (0.55 × rule_score)

rule_score  = (0.30 × title_relevance
             + 0.30 × production_evidence    ← 44-template O(1) lookup
             + 0.15 × experience_fit
             + 0.15 × location_fit
             + 0.10 × notice_fit)
             × anti_pattern_penalty

final_score = clip((base_score + plain_boost) × behavioral_mult × integrity_mult, 0, 1)
```

### Anti-Pattern Penalties
| Anti-Pattern | Multiplier | JD Reference |
|---|---|---|
| Entire career at consulting firms (TCS/Infosys/Wipro/etc.) | **0.35×** | JD explicitly excluded |
| CV/speech/robotics only — zero NLP/IR exposure | **0.40×** | JD explicitly excluded |
| 3+ short tenures (<18mo) with escalating seniority titles | **0.50×** | "Title-chasers" |
| Shallow LLM wrapper (LangChain only, no pre-LLM ML) | **0.40×** | JD explicitly excluded |

### Honeypot Detection (7 checks)
| Check | What it catches |
|---|---|
| Career date consistency | `end_date` before `start_date`; duration ≠ computed months |
| Single role > total YOE | Impossible tenure math |
| Skill duration > total YOE | Expert in skill they couldn't have used long enough |
| Expert + 0 endorsements + <3mo | Proficiency claim with no evidence |
| Seniority title + <1.5 yrs total | Senior/Staff/Principal with near-zero experience |
| Education sanity | Degree spanning >10 years or end before start |
| Experience inflation | Claimed YOE >> sum of career_history durations |

---

## 🚀 How to Reproduce

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Precompute embeddings (one-time, ~7 min CPU / ~2 min GPU)
```bash
python precompute_embeddings.py --candidates ./candidates.json
```
This generates `data/candidate_embeddings.npy` and `data/candidate_ids.npy`.

### Step 3 — Run the ranking pipeline
```bash
python rank.py --candidates ./candidates.json --jd ./job_description.txt --out ./submission.csv
```

**Expected output:**
```
INFO - Initializing Production Ranking Pipeline...
INFO - Loaded 44 hand-labeled production templates.
INFO - Processing Job Description...
INFO - Loaded 100000 candidate records.
INFO - Computing global semantic similarities...
INFO - Executing O(1) Template lookups and integrity checks...
INFO - Sorting finalists and generating submission artifacts...
INFO - Pipeline completed successfully in ~54 seconds.
INFO - Output saved to ./submission.csv
```

### Optional — Run unit tests
```bash
pytest tests/
```

---

## 📊 Sample Output (Top 5)

| Rank | candidate_id | Score | Reasoning |
|------|-------------|-------|-----------|
| 1 | CAND_0006567 | 0.869 | 7.9 yrs, Senior AI Engineer (Noida). Notice: 60 days. 79% response, 93% interview completion. |
| 2 | CAND_0055905 | 0.841 | 8.1 yrs, Senior MLE (London). Notice: 30 days. 87% response, 67% interview completion. |
| 3 | CAND_0046064 | 0.827 | 8.9 yrs, Senior NLP Engineer. Notice: 30 days. 78% response, 80% interview completion. |
| 4 | CAND_0011687 | 0.824 | 7.8 yrs, Senior NLP Engineer (Indore). Notice: 15 days. 89% response, 77% interview completion. |
| 5 | CAND_0046525 | 0.820 | 6.1 yrs, Senior MLE (Pune). Notice: 60 days. 88% response, 81% interview completion. |

See [`sample_output/sample_submission.csv`](sample_output/sample_submission.csv) for the full top-10 preview.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers 3.0.1) |
| Similarity computation | NumPy vectorized cosine (`np.dot` on L2-normalized matrix) |
| Candidate data | JSON / JSONL, auto-detected |
| Pre-computation | Offline `.npy` files (float32, ~150MB, not committed) |
| GPU | Optional — auto-detected (T4 GPU cuts precompute to ~2 min) |
| External API calls | **None** during ranking |

---

## 👥 Team

**Team Name:** ByteBrad's

| Name | Role | Email |
|---|---|---|
| Komakula Pardheev | Lead / Architect | pardheev2006@gmail.com |
| Gorantala Vinay Kumar | ML Engineer | vinaykumargorantalavk@gmail.com |
| M Keerthana | Data Engineer | mkeerthana217@gmail.com |

**AI Tools Used:** Claude (Anthropic), Gemini (Google DeepMind)  
**Sandbox:** [Google Colab](https://colab.research.google.com/drive/1Ed2S-2tn9KbigP9cI2Zc5ymPho5Sz7Oc?usp=sharing)

---

## 📜 Declarations

- ✅ No candidate data was sent to any external LLM API during ranking
- ✅ All ranking is performed locally using a cached sentence-transformers model
- ✅ Code is original work — no plagiarism
- ✅ Reproduction tested end-to-end

---

*Built for the Redrob × Hack2skill India Runs 2026 — Track 1: Data & AI Challenge*

"""
Central config: every weight and keyword list lives here so the scoring
logic in scoring.py stays readable and so you can defend/tune each number
individually in the Stage 5 interview.
"""

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small, CPU-fast, good quality/speed tradeoff
EMBEDDINGS_PATH = "data/candidate_embeddings.npy"
CANDIDATE_IDS_PATH = "data/candidate_ids.npy"

# ---------------------------------------------------------------------------
# Final score = SEMANTIC_WEIGHT * semantic_sim + RULE_WEIGHT * rule_score
# then multiplied by the behavioral multiplier, then honeypot/anti-pattern
# penalties are applied last.
# ---------------------------------------------------------------------------
SEMANTIC_WEIGHT = 0.45
RULE_WEIGHT = 0.55

# --- Rule sub-weights (sum to 1.0, applied inside rule_score) ---
W_TITLE_RELEVANCE = 0.30
W_PRODUCTION_EVIDENCE = 0.30
W_EXPERIENCE_FIT = 0.15
W_LOCATION_FIT = 0.15
W_NOTICE_FIT = 0.10

# --- Vocabulary: what the JD says it actually wants ---
POSITIVE_TITLE_TERMS = [
    "ai engineer", "machine learning engineer", "ml engineer", "applied scientist",
    "research engineer", "data scientist", "nlp engineer", "search engineer",
    "ranking", "retrieval", "recommendation", "search relevance", "mle",
    "ai/ml", "artificial intelligence engineer",
]

PRODUCTION_EVIDENCE_TERMS = [
    "embedding", "vector database", "vector db", "faiss", "pinecone", "weaviate",
    "qdrant", "milvus", "opensearch", "elasticsearch", "bm25", "hybrid search",
    "ndcg", "mrr", "map@", "a/b test", "ab test", "offline evaluation",
    "online evaluation", "fine-tun", "lora", "qlora", "peft", "rag",
    "retrieval augmented", "deployed to production", "production deployment",
    "real-time", "recommendation system", "ranking system", "search system",
    "llm", "transformer", "bert", "sentence-transformers", "semantic search",
    "feature store", "model serving", "inference pipeline",
]

# JD says: pure research-only (no production deployment) is a disqualifier signal
RESEARCH_ONLY_TERMS = ["research scientist", "phd researcher", "academic", "research lab", "postdoc"]

# JD says: LangChain+OpenAI-only, <12mo, with no pre-LLM-era production ML, is a soft disqualifier
SHALLOW_LLM_WRAPPER_TERMS = ["langchain", "openai api", "gpt wrapper", "chatgpt integration", "prompt engineering"]
DEEP_ML_TERMS = ["pytorch", "tensorflow", "scikit-learn", "xgboost", "recommendation", "ranking", "retrieval", "nlp", "ml pipeline"]

# JD explicitly does NOT want candidates whose entire career is at these firms
CONSULTING_ONLY_FIRMS = ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "tata consultancy"]

# JD explicitly does NOT want CV/speech/robotics-only people without NLP/IR exposure
CV_SPEECH_ROBOTICS_TERMS = ["computer vision", "image classification", "object detection",
                            "speech recognition", "robotics", "autonomous vehicle", "signal processing",
                            "image segmentation", "video analytics"]
NLP_IR_TERMS = ["nlp", "natural language", "retrieval", "search", "ranking", "embedding", "text classification", "llm"]

# Title-chasing: short tenures with escalating seniority words
SENIORITY_LADDER = ["intern", "junior", "associate", "engineer", "senior", "staff", "principal", "lead", "director", "vp"]
SHORT_TENURE_MONTHS = 18

# Locations the JD prefers
# Hedge/disclaiming phrases: language that admits the candidate was NOT the
# owner of the production/ML work being described. Keyword-counting alone
# can't tell "I built this" from "I didn't build this" — this list catches
# the most common disclaiming patterns seen in the actual dataset.
HEDGE_PHRASES = [
    "didn't make it to production", "did not make it to production",
    "handled by the platform team", "handled by another team",
    "was secondary", "secondary to", "limited", "still building depth",
    "lighter weight than", "not from-scratch", "pre-trained model fine-tuning, not",
    "professional experience there is limited", "lighter on the deep-learning side",
    "my own technical depth",
]

PREFERRED_LOCATIONS = ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr", "gurugram", "gurgaon"]

# Notice period scoring
NOTICE_GREAT_DAYS = 30
NOTICE_OK_DAYS = 60

# --- Behavioral multiplier tuning ---
INACTIVE_DAYS_SOFT = 60     # beyond this, recency starts hurting
INACTIVE_DAYS_HARD = 180    # beyond this, recency hurts a lot
MULTIPLIER_FLOOR = 0.35
MULTIPLIER_CEILING = 1.15

TODAY = "2026-06-20"  # dataset reference date for recency calculations; override via CLI if needed
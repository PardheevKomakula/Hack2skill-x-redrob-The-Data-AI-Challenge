"""
Shared helpers for turning a raw candidate JSON record into:
  1. a narrative text string (for embedding)
  2. a flattened feature dict (for rule scoring)

Used identically by precompute_embeddings.py and rank.py so the two
stages never drift out of sync.
"""
from datetime import date, datetime


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def build_narrative(candidate: dict) -> str:
    """Single text blob capturing everything semantically relevant about a candidate."""
    profile = candidate.get("profile", {})
    parts = [
        profile.get("headline", ""),
        profile.get("current_title", ""),
        profile.get("summary", ""),
    ]
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    skill_names = [s.get("name", "") for s in candidate.get("skills", [])]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))
    return " . ".join(p for p in parts if p)


def all_text_lower(candidate: dict) -> str:
    """Lowercased concatenation used for simple keyword scanning."""
    return build_narrative(candidate).lower()


def career_company_text(candidate: dict) -> list:
    return [job.get("company", "").lower() for job in candidate.get("career_history", [])]


def career_titles(candidate: dict) -> list:
    return [job.get("title", "").lower() for job in candidate.get("career_history", [])]    
"""
Rule-based scoring that directly encodes what job_description.docx says
it rewards and punishes. This is the part of the system that catches the
"perfect skill list but title is Marketing Manager" trap and the
"Tier-5 wording but real production IR experience" upside the JD
explicitly calls out.
"""
from datetime import date
from . import config as cfg
from .features import all_text_lower, career_company_text, career_titles


def _keyword_hits(text: str, terms: list) -> int:
    return sum(1 for t in terms if t in text)


def title_relevance_score(candidate: dict) -> float:
    """How much of the candidate's current + past titles match AI/ML/IR roles."""
    titles_text = " ".join(career_titles(candidate)) + " " + (candidate.get("profile", {}).get("current_title", "").lower())
    hits = _keyword_hits(titles_text, cfg.POSITIVE_TITLE_TERMS)
    return min(1.0, hits / 2.0)  # 2+ matching title terms = full score


def matched_terms(text: str, terms: list) -> list:
    """Returns the actual subset of terms found in the text, preserving config order."""
    return [t for t in terms if t in text]


def production_evidence_score(candidate: dict) -> float:
    """
    Primary signal: exact-match lookup against the manually-labeled template
    catalog (template_scores.py). This dataset's career_history descriptions
    are built from a confirmed-finite set of 44 templates (see
    extract_template_catalog.py output), so a direct lookup is far more
    reliable than re-deriving a score from keywords every time.

    Falls back to keyword heuristics only for any description NOT found in
    the lookup table (e.g. if the candidate pool changes).

    Aggregation across a candidate's career_history: most-recent role
    weighted heaviest (it's the best signal of current capability), blended
    with the single best score anywhere in their history (credits genuine
    standout experience) and the average (penalizes a pool of mostly-noise
    roles with one lucky strong entry).
    """
    from .template_scores import TEMPLATE_SCORES

    career = candidate.get("career_history", [])
    if not career:
        return 0.0

    job_scores = []
    for job in career:
        desc = (job.get("description") or "").strip()
        if desc in TEMPLATE_SCORES:
            job_scores.append(TEMPLATE_SCORES[desc])
        else:
            job_scores.append(_keyword_fallback_score(desc.lower()))

    most_recent = job_scores[0]  # career_history[0] is the most recent role in this dataset
    best = max(job_scores)
    avg = sum(job_scores) / len(job_scores)

    score = 0.5 * most_recent + 0.3 * best + 0.2 * avg

    # Still penalize shallow-LLM-wrapper language and pure-research-only language
    # on top of the template score, since those are summary-level patterns,
    # not career_history-template patterns.
    text = all_text_lower(candidate)
    shallow_hits = _keyword_hits(text, cfg.SHALLOW_LLM_WRAPPER_TERMS)
    deep_hits = _keyword_hits(text, cfg.DEEP_ML_TERMS)
    if shallow_hits >= 2 and deep_hits == 0:
        score *= 0.4

    return min(1.0, score)


def _keyword_fallback_score(text: str) -> float:
    """Used only when a career_history description isn't in the known template catalog."""
    hits = _keyword_hits(text, cfg.PRODUCTION_EVIDENCE_TERMS)
    score = min(1.0, hits / 6.0)
    hedge_hits = _keyword_hits(text, cfg.HEDGE_PHRASES)
    if hedge_hits >= 3:
        score *= 0.35
    elif hedge_hits >= 1:
        score *= 0.7
    return score


def anti_pattern_penalty(candidate: dict) -> float:
    """Returns a multiplier in (0, 1]; 1.0 = no penalty."""
    penalty = 1.0
    companies = career_company_text(candidate)
    text = all_text_lower(candidate)

    # Consulting-only career (JD: explicit no, unless prior product-company experience shown elsewhere)
    if companies and all(any(f in c for f in cfg.CONSULTING_ONLY_FIRMS) for c in companies):
        penalty *= 0.35

    # CV/speech/robotics-only with zero NLP/IR exposure
    cv_hits = _keyword_hits(text, cfg.CV_SPEECH_ROBOTICS_TERMS)
    nlp_hits = _keyword_hits(text, cfg.NLP_IR_TERMS)
    if cv_hits >= 2 and nlp_hits == 0:
        penalty *= 0.4

    # Title-chasing: 3+ short tenures (<18mo) each with an escalating seniority word
    short_escalations = 0
    for job in candidate.get("career_history", []):
        duration = job.get("duration_months", 0) or 0
        jt = (job.get("title") or "").lower()
        if duration < cfg.SHORT_TENURE_MONTHS and any(s in jt for s in cfg.SENIORITY_LADDER[3:]):
            short_escalations += 1
    if short_escalations >= 3:
        penalty *= 0.5

    return penalty


def experience_fit_score(candidate: dict) -> float:
    years = candidate.get("profile", {}).get("years_of_experience", 0) or 0
    if 5 <= years <= 9:
        return 1.0
    if 3 <= years < 5 or 9 < years <= 12:
        return 0.6
    if years < 3:
        return 0.25
    return 0.35  # very senior, outside band but JD says "case by case"


def location_fit_score(candidate: dict) -> float:
    loc = (candidate.get("profile", {}).get("location", "") or "").lower()
    sig = candidate.get("redrob_signals", {})
    if any(p in loc for p in cfg.PREFERRED_LOCATIONS):
        return 1.0
    if sig.get("willing_to_relocate"):
        return 0.65
    return 0.25


def notice_fit_score(candidate: dict) -> float:
    days = candidate.get("redrob_signals", {}).get("notice_period_days", 90) or 90
    if days <= cfg.NOTICE_GREAT_DAYS:
        return 1.0
    if days <= cfg.NOTICE_OK_DAYS:
        return 0.7
    return 0.4


def rule_score(candidate: dict) -> float:
    score = (
        cfg.W_TITLE_RELEVANCE * title_relevance_score(candidate)
        + cfg.W_PRODUCTION_EVIDENCE * production_evidence_score(candidate)
        + cfg.W_EXPERIENCE_FIT * experience_fit_score(candidate)
        + cfg.W_LOCATION_FIT * location_fit_score(candidate)
        + cfg.W_NOTICE_FIT * notice_fit_score(candidate)
    )
    return score * anti_pattern_penalty(candidate)


def behavioral_multiplier(candidate: dict, today: date) -> float:
    sig = candidate.get("redrob_signals", {})
    mult = 1.0

    # Recency of activity
    from .features import parse_date
    last_active = parse_date(sig.get("last_active_date"))
    if last_active:
        days_inactive = (today - last_active).days
        if days_inactive > cfg.INACTIVE_DAYS_HARD:
            mult *= 0.5
        elif days_inactive > cfg.INACTIVE_DAYS_SOFT:
            mult *= 0.8

    # Availability
    if sig.get("open_to_work_flag") is False:
        mult *= 0.75

    # Responsiveness / reliability
    rr = sig.get("recruiter_response_rate", 0.5)
    mult *= 0.85 + 0.3 * rr  # ranges ~0.85 to 1.15

    icr = sig.get("interview_completion_rate", 0.7)
    mult *= 0.9 + 0.15 * icr

    # Verification / trust signals (small nudge)
    verified_count = sum([
        bool(sig.get("verified_email")),
        bool(sig.get("verified_phone")),
        bool(sig.get("linkedin_connected")),
    ])
    mult *= 0.95 + 0.02 * verified_count

    return max(cfg.MULTIPLIER_FLOOR, min(cfg.MULTIPLIER_CEILING, mult))
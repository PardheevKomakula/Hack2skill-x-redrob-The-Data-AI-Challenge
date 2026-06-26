import logging
from . import honeypot

logger = logging.getLogger(__name__)


def evaluate_profile_integrity(candidate: dict, config: dict) -> tuple[bool, float]:
    """
    Delegates to honeypot.py's detect_honeypot_flags(), which implements all
    7 internal-consistency checks (career-history date math, skill-duration
    vs total experience, expert-proficiency-with-near-zero-use, seniority/
    experience mismatch, education sanity, YOE-vs-career-history-span
    inflation, and duplicate descriptions within one candidate's own
    history) — rather than re-implementing a smaller subset here, which is
    how the YOE-vs-history check silently went dead in an earlier version
    of this file (it read candidate["years_of_experience"] instead of
    candidate["profile"]["years_of_experience"]).

    Returns (is_valid_flag, score_multiplier).
    """
    flags = honeypot.detect_honeypot_flags(candidate)
    is_valid = len(flags) == 0

    if flags:
        n = len(flags)
        if n >= 2:
            multiplier = config['penalties']['honeypot_multiplier']  # 0.02 — crushed
        else:
            multiplier = 0.35  # soft penalty for single flag
        logger.debug(
            f"Honeypot flags for {candidate.get('candidate_id')}: {flags}"
        )
    else:
        multiplier = 1.0

    return is_valid, multiplier
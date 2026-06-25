import logging

logger = logging.getLogger(__name__)

def evaluate_profile_integrity(candidate: dict, config: dict) -> tuple[bool, float]:
    """
    Checks for the 7 internal consistency traps (e.g., chronological anomalies, duplication).
    Returns a tuple of (is_valid_flag, score_multiplier).
    """
    multiplier = 1.0
    is_valid = True
    
    yoe = candidate.get("years_of_experience", 0)
    history = candidate.get("career_history", [])
    
    # Example Trap: Calculated duration versus claimed YOE
    if yoe > 0 and not history:
        is_valid = False
        multiplier *= config['penalties']['honeypot_multiplier']
        
    # Example Trap: Empty/Duplicate descriptions check
    descriptions = [job.get("description", "") for job in history if job.get("description")]
    if len(descriptions) != len(set(descriptions)) and len(descriptions) > 1:
        is_valid = False
        multiplier *= config['penalties']['honeypot_multiplier']
        
    return is_valid, multiplier
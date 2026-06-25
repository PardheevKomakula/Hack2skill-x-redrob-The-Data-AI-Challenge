import pytest
from src.compliance import evaluate_profile_integrity

def test_honeypot_empty_history_trap():
    config = {'penalties': {'honeypot_multiplier': 0.02}}
    malicious_candidate = {
        "candidate_id": "CAND_TRAP_99",
        "years_of_experience": 12.5,
        "career_history": []  # Zero proof of roles
    }
    
    is_valid, multiplier = evaluate_profile_integrity(malicious_candidate, config)
    assert is_valid is False
    assert multiplier == 0.02
import numpy as np
from . import scoring


class RankingPipeline:
    def __init__(self, config: dict, template_mapping: dict):
        self.config = config
        self.template_mapping = template_mapping  # kept for reference/debugging only —
        # the actual lookup now happens inside scoring.production_evidence_score()
        self.w_semantic = config['weights']['semantic_similarity']
        self.w_rule = config['weights']['rule_score']

    def calculate_candidate_score(
        self,
        candidate: dict,
        semantic_sim: float,
        integrity_mult: float,
        plain_boost: float,
        behavioral_mult: float,
    ) -> float:
        """
        Takes the full `candidate` dict (not a single template_text string)
        so that rule_score() can apply the full 5-component weighted formula:
          0.30 × title_relevance
          0.30 × production_evidence  (44-template O(1) lookup, multi-role aggregated)
          0.15 × experience_fit       (5-9 yr sweet spot)
          0.15 × location_fit         (Pune/Noida preferred)
          0.10 × notice_fit           (≤30 days preferred)
        multiplied by anti_pattern_penalty (consulting-only 0.35x, CV-only 0.40x,
        title-chaser 0.50x, shallow-LLM-wrapper 0.40x).

        The old implementation did a flat template dict lookup on a single
        description string — which bypassed all 5 scoring dimensions and all
        anti-pattern penalties silently.
        """
        rs = scoring.rule_score(candidate)
        base_score = (self.w_semantic * semantic_sim) + (self.w_rule * rs)
        final_score = (base_score + plain_boost) * behavioral_mult * integrity_mult
        return max(0.0, min(1.0, final_score))
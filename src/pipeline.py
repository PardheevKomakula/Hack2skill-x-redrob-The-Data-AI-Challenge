import numpy as np

class RankingPipeline:
    def __init__(self, config: dict, template_mapping: dict):
        self.config = config
        self.template_mapping = template_mapping
        self.w_semantic = config['weights']['semantic_similarity']
        self.w_rule = config['weights']['rule_score']

    def calculate_candidate_score(self, semantic_sim: float, template_text: str, integrity_mult: float, plain_boost: float, behavioral_mult: float) -> float:
        # Resolve production-evidence mapping over template dictionary
        rule_score = self.template_mapping.get(template_text, 0.5)
        
        # Core base weight blend + boosts and operational filters
        base_score = (self.w_semantic * semantic_sim) + (self.w_rule * rule_score)
        final_score = (base_score + plain_boost) * behavioral_mult * integrity_mult
        
        return max(0.0, min(1.0, final_score))
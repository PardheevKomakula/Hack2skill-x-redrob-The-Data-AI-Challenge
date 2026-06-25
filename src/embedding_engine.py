import os
import re
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ProductionEmbeddingEngine:
    def __init__(self, config: dict):
        self.model_name = config['model']['name']
        self.cache_dir = config['model']['cache_dir']
        self.device = config['model']['device']
        self.boost_terms = config.get('plain_language_terms', [])
        self.model = self.load_model()

    def load_model(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            return SentenceTransformer(
                self.model_name, 
                cache_folder=self.cache_dir, 
                device=self.device
            )
        except Exception as e:
            logger.error(f"Failed to load local model: {e}. Falling back to default initialization.")
            return SentenceTransformer(self.model_name, device=self.device)

    def evaluate_plain_language_boost(self, text: str) -> float:
        if not text:
            return 0.0
        text_lower = text.lower()
        matches = sum(1 for term in self.boost_terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower))
        return min(0.1, matches * 0.02)
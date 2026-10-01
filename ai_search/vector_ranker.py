import math
from typing import List, Dict, Any, Optional
from .hybrid_vector_engine import HybridVectorEngine

class VectorRanker:
    """
    Production-Grade Hybrid Vector Ranker using Dense Embeddings + BM25 + Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_dim: int = 128):
        self.hybrid_engine = HybridVectorEngine(vector_dim=vector_dim)

    def tokenize(self, text: str) -> List[str]:
        return self.hybrid_engine._clean_tokens(text)

    def rank_candidates(self, query: Any, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Ranks candidate documents using Hybrid Dense + Sparse RRF Vector Reranking.
        """
        if not candidates:
            return []

        if isinstance(query, list):
            query_str = " ".join(query)
        else:
            query_str = str(query or "")

        if not query_str.strip():
            return candidates[:top_k]

        return self.hybrid_engine.rank_documents(query_str, candidates, top_k=top_k)

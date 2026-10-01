import pytest
import numpy as np
from ai_search.hybrid_vector_engine import HybridVectorEngine
from ai_search.vector_ranker import VectorRanker

def test_dense_vector_encoding():
    engine = HybridVectorEngine(vector_dim=128)
    vec1 = engine.encode_dense_vector("payment upi failure")
    vec2 = engine.encode_dense_vector("payment upi failure")
    vec3 = engine.encode_dense_vector("completely unrelated text hospital notice")

    assert vec1.shape == (128,)
    # Self-similarity should be ~1.0
    sim_self = engine.compute_dense_similarity(vec1, vec2)
    assert sim_self > 0.99

    # Unrelated text similarity should be lower
    sim_diff = engine.compute_dense_similarity(vec1, vec3)
    assert sim_diff < sim_self

def test_bm25_sparse_scoring():
    engine = HybridVectorEngine()
    q_tokens = ["payment", "failed"]
    doc1 = ["upi", "payment", "failed", "due", "to", "bank", "error"]
    doc2 = ["weather", "forecast", "sunny", "today"]

    score1 = engine.compute_sparse_bm25(q_tokens, doc1)
    score2 = engine.compute_sparse_bm25(q_tokens, doc2)

    assert score1 > 0.0
    assert score2 == 0.0

def test_hybrid_rrf_ranking():
    ranker = VectorRanker()
    candidates = [
        {"id": 1, "title": "Weather Report", "content": "Sunny today in Mumbai"},
        {"id": 2, "title": "UPI Payment Error", "content": "Money debited from bank account but recharge failed"},
        {"id": 3, "title": "Customer Support Ticket", "content": "Agent closed my ticket without resolving billing issue"}
    ]

    query = "paise cut gaye payment issue"
    results = ranker.rank_candidates(query, candidates, top_k=3)

    assert len(results) == 3
    # Top result should be doc #2 (UPI Payment Error)
    assert results[0]["id"] == 2
    assert "hybrid_score" in results[0]
    assert "dense_score" in results[0]
    assert "sparse_score" in results[0]
    assert results[0]["hybrid_score"] > results[-1]["hybrid_score"]

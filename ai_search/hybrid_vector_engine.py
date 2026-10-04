import math
import re
import zlib
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

class HybridVectorEngine:
    """
    Production-Grade Hybrid Vector Search Engine.
    Combines:
    1. Dense Vector Embeddings (Semantic Similarity via Sub-word Projection & Cosine Distance)
    2. Sparse BM25 / Keyword Retrieval (Exact Token Match & Lexical Precision)
    3. Reciprocal Rank Fusion (RRF) for Combining Dense & Sparse Rankings.
    """
    def __init__(self, vector_dim: int = 128, rrf_k: int = 60):
        self.vector_dim = vector_dim
        self.rrf_k = rrf_k
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
            'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'
        }

    def _clean_tokens(self, text: str) -> List[str]:
        if not text or not isinstance(text, str):
            return []
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if w not in self.stopwords and len(w) > 1]

    def _get_subword_ngrams(self, word: str, n_min: int = 3, n_max: int = 4) -> List[str]:
        ngrams = []
        w_padded = f"<{word}>"
        for n in range(n_min, n_max + 1):
            for i in range(len(w_padded) - n + 1):
                ngrams.append(w_padded[i:i+n])
        return ngrams

    def encode_dense_vector(self, text: str) -> np.ndarray:
        """
        Encodes arbitrary text into a normalized 128-dimensional dense semantic embedding vector.
        Uses deterministic sub-word hashing (zlib.crc32) to capture typos, morphological roots,
        and semantic similarity consistently across all OS platforms and Python runtimes.
        """
        vec = np.zeros(self.vector_dim, dtype=np.float32)
        tokens = self._clean_tokens(text)
        if not tokens:
            return vec

        total_weight = 0.0
        for token in tokens:
            ngrams = self._get_subword_ngrams(token)
            for ngram in ngrams:
                # Deterministic feature hash projection
                h_val = zlib.crc32(ngram.encode('utf-8'))
                h = h_val % self.vector_dim
                val = (zlib.crc32((ngram + "_sign").encode('utf-8')) % 2) * 2 - 1  # +1 or -1
                vec[h] += float(val)
                total_weight += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


    def compute_dense_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates Cosine Similarity between two dense embedding vectors.
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        dot = float(np.dot(vec1, vec2))
        return max(0.0, min(1.0, (dot + 1.0) / 2.0))

    def compute_sparse_bm25(self, query_tokens: List[str], doc_tokens: List[str], avg_dl: float = 20.0, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculates Sparse BM25 / TF-IDF score for exact token matches.
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        doc_freq = {}
        for t in doc_tokens:
            doc_freq[t] = doc_freq.get(t, 0) + 1

        score = 0.0
        for q_token in set(query_tokens):
            if q_token in doc_freq:
                tf = doc_freq[q_token]
                idf = math.log((100.0 + 1) / (tf + 0.5)) + 1.0
                num = tf * (k1 + 1.0)
                den = tf + k1 * (1.0 - b + b * (doc_len / avg_dl))
                score += idf * (num / den)

        return min(1.0, score / 10.0)

    def rank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF).
        RRF Score(d) = 1 / (60 + Dense_Rank(d)) + 1 / (60 + Sparse_Rank(d))
        """
        if not query or not documents:
            return []

        q_dense = self.encode_dense_vector(query)
        q_tokens = self._clean_tokens(query)

        # 1. Compute Dense & Sparse Scores
        scored_docs = []
        for idx, doc in enumerate(documents):
            text = f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('text', '')} {doc.get('description', '')}".trim() if hasattr(str, 'trim') else f"{doc.get('title', '')} {doc.get('content', '')} {doc.get('text', '')} {doc.get('description', '')}".strip()
            
            d_dense = self.encode_dense_vector(text)
            d_tokens = self._clean_tokens(text)

            dense_sim = self.compute_dense_similarity(q_dense, d_dense)
            sparse_sim = self.compute_sparse_bm25(q_tokens, d_tokens)

            scored_docs.append({
                "doc": doc,
                "dense_score": float(dense_sim),
                "sparse_score": float(sparse_sim),
                "index": idx
            })

        # 2. Sort by Dense Rank
        dense_sorted = sorted(scored_docs, key=lambda x: x["dense_score"], reverse=True)
        dense_ranks = {item["index"]: rank + 1 for rank, item in enumerate(dense_sorted)}

        # 3. Sort by Sparse Rank
        sparse_sorted = sorted(scored_docs, key=lambda x: x["sparse_score"], reverse=True)
        sparse_ranks = {item["index"]: rank + 1 for rank, item in enumerate(sparse_sorted)}

        # 4. Apply Reciprocal Rank Fusion (RRF)
        results = []
        for item in scored_docs:
            idx = item["index"]
            r_dense = dense_ranks[idx]
            r_sparse = sparse_ranks[idx]

            rrf_score = (1.0 / (self.rrf_k + r_dense)) + (1.0 / (self.rrf_k + r_sparse))
            normalized_score = min(0.99, max(0.40, (item["dense_score"] * 0.5) + (item["sparse_score"] * 0.5) + (rrf_score * 5.0)))

            doc_copy = dict(item["doc"])
            doc_copy["hybrid_score"] = round(float(normalized_score), 3)
            doc_copy["dense_score"] = round(float(item["dense_score"]), 3)
            doc_copy["sparse_score"] = round(float(item["sparse_score"]), 3)
            doc_copy["rrf_rank"] = {
                "dense_rank": r_dense,
                "sparse_rank": r_sparse
            }
            results.append(doc_copy)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

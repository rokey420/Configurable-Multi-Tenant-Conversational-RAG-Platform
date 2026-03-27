# backend/retrieval/rerank.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder

_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, passages: List[str], top_k: int = 5) -> List[Tuple[int, str, float]]:
    """
    Returns list of (original_index, passage_text, score) sorted by score desc.
    """
    if not passages:
        return []

    pairs = [(query, p) for p in passages]
    scores = _reranker.predict(pairs)

    ranked = sorted(
        [(i, passages[i], float(scores[i])) for i in range(len(passages))],
        key=lambda x: x[2],
        reverse=True
    )
    return ranked[:top_k]
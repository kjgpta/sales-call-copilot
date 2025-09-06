from typing import List, Tuple, Dict
from .config import RERANKER, RERANKER_MODEL

def cross_encoder_rerank(query: str, candidates: List[Tuple[int, str]]):
    """candidates: list of (seg_id, text) -> returns list[(seg_id, score)] sorted desc
    """
    if RERANKER != "cross-encoder":
        return None
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(RERANKER_MODEL)
        pairs = [[query, c[1]] for c in candidates]
        scores = model.predict(pairs)
        out = [(candidates[i][0], float(scores[i])) for i in range(len(candidates))]
        out.sort(key=lambda x: x[1], reverse=True)
        return out
    except Exception:
        return None

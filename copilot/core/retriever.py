from typing import List, Tuple, Dict
from .index_lex import BM25Index
from .index_dense import DenseIndex
from .config import INDEX_DIR, HYBRID_DENSE_WEIGHT
import os, json

def merge_scores(lex: List[Tuple[int,float]], dense: List[Tuple[int,float]]) -> List[Tuple[int,float,Dict[str,float]]]:
    # Normalize each list to [0,1], then weighted sum
    def normalize(pairs):
        if not pairs:
            return []
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [(sid, 1.0) for sid, _ in pairs]
        return [(sid, (s - lo) / (hi - lo + 1e-9)) for sid, s in pairs]

    lex_n = normalize(lex)
    dense_n = normalize(dense)

    dmap = {sid: s for sid, s in dense_n}
    lmap = {sid: s for sid, s in lex_n}
    keys = set(dmap) | set(lmap)
    out = []
    for k in keys:
        ls = lmap.get(k, 0.0)
        ds = dmap.get(k, 0.0)
        final = (1 - HYBRID_DENSE_WEIGHT) * ls + HYBRID_DENSE_WEIGHT * ds
        out.append((k, final, {"lex": ls, "dense": ds}))
    out.sort(key=lambda x: x[1], reverse=True)
    return out

class Retriever:
    def __init__(self, bm25: BM25Index, dense: DenseIndex | None):
        self.bm25 = bm25
        self.dense = dense

    def retrieve(self, query: str, qvec = None, topk: int = 12):
        lex = self.bm25.search(query, topk=40)
        dense = []
        if self.dense and qvec is not None:
            dense = self.dense.search(qvec, topk=40)
        merged = merge_scores(lex, dense)
        return merged[:topk]

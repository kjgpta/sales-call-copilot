import os, pickle, re
from typing import List, Tuple
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

class BM25Index:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.doc_tokens: List[List[str]] = []
        self.seg_ids: List[int] = []
        self.bm25: BM25Okapi | None = None

    def _tokenize(self, s: str) -> List[str]:
        return [t.lower() for t in TOKEN_RE.findall(s)]

    def build(self, corpus: List[Tuple[int, str]]):
        # corpus: list of (seg_id, text)
        self.seg_ids = [sid for sid, _ in corpus]
        self.doc_tokens = [self._tokenize(txt) for _, txt in corpus]
        self.bm25 = BM25Okapi(self.doc_tokens)

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        with open(os.path.join(self.index_dir, "bm25.pkl"), "wb") as f:
            pickle.dump({"seg_ids": self.seg_ids, "doc_tokens": self.doc_tokens}, f)

    def load(self) -> bool:
        path = os.path.join(self.index_dir, "bm25.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.seg_ids = data["seg_ids"]
        self.doc_tokens = data["doc_tokens"]
        if self.doc_tokens:
            self.bm25 = BM25Okapi(self.doc_tokens)
        return True

    def search(self, query: str, topk: int = 20) -> List[Tuple[int, float]]:
        if not self.bm25:
            return []
        q_toks = self._tokenize(query)
        scores = self.bm25.get_scores(q_toks)
        pairs = list(zip(self.seg_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:topk]

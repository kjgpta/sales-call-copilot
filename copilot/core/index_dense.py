import os, json, pickle
from typing import List, Tuple
import numpy as np

class DenseIndex:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.embeddings: np.ndarray | None = None  # shape (N, D)
        self.seg_ids: List[int] = []

    def load(self) -> bool:
        emb_path = os.path.join(self.index_dir, "embeddings.npy")
        map_path = os.path.join(self.index_dir, "embeddings_ids.json")
        if not (os.path.exists(emb_path) and os.path.exists(map_path)):
            return False
        self.embeddings = np.load(emb_path)
        with open(map_path, "r") as f:
            self.seg_ids = json.load(f)
        return True

    def save(self):
        os.makedirs(self.index_dir, exist_ok=True)
        np.save(os.path.join(self.index_dir, "embeddings.npy"), self.embeddings)
        with open(os.path.join(self.index_dir, "embeddings_ids.json"), "w") as f:
            json.dump(self.seg_ids, f)

    def build(self, seg_ids: List[int], emb_matrix: np.ndarray):
        # Expect emb_matrix normalized row-wise
        self.seg_ids = seg_ids
        self.embeddings = emb_matrix

    def search(self, q: np.ndarray, topk: int = 20) -> List[Tuple[int, float]]:
        if self.embeddings is None:
            return []
        # q is 1 x D normalized
        sims = (self.embeddings @ q.reshape(-1, 1)).ravel()  # cosine since normalized
        idxs = np.argsort(-sims)[:topk]
        return [(self.seg_ids[i], float(sims[i])) for i in idxs]

import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from schemas import GlobalRAGEntry

EMBEDDING_DIM = 384
INDEX_PATH = "global_rag.index"
DATA_PATH = "global_rag.json"

class GlobalRAG:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.entries = []
        self.index = faiss.IndexFlatL2(EMBEDDING_DIM)

        if os.path.exists(INDEX_PATH) and os.path.exists(DATA_PATH):
            self._load()

    def _load(self):
        self.index = faiss.read_index(INDEX_PATH)
        with open(DATA_PATH, "r") as f:
            raw = json.load(f)
            self.entries = [GlobalRAGEntry(**e) for e in raw]

    def _persist(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(DATA_PATH, "w") as f:
            json.dump([e.dict() for e in self.entries], f, indent=2)

    def ingest(self, entry: GlobalRAGEntry):
        embedding = self.model.encode([entry.content]).astype("float32")
        self.index.add(embedding)
        self.entries.append(entry)
        self._persist()

    def retrieve(self, query: str, k: int = 5, tags=None):
        q_emb = self.model.encode([query]).astype("float32")
        _, indices = self.index.search(q_emb, k * 2)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            entry = self.entries[idx]
            if tags and not set(tags).issubset(set(entry.tags)):
                continue
            results.append(entry)
            if len(results) >= k:
                break

        return results

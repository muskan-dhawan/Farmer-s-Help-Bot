import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class RAGService:
    def __init__(self,
                 index_path="data/faiss/faiss_index.bin",
                 texts_path="data/faiss/texts.npy",
                 meta_path="data/faiss/meta.npy"):

        # 🔥 FIX: MULTILINGUAL MODEL
        self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.texts = np.load(texts_path, allow_pickle=True)
            self.meta = np.load(meta_path, allow_pickle=True)
        else:
            self.index = None
            self.texts = []
            self.meta = []

    def search(self, query, state=None, crop=None, k=5):
        if self.index is None:
            return ["⚠️ RAG index not built. Run build_faiss.py"]

        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)

        results = []

        for idx in I[0]:
            if idx == -1:
                continue

            text = self.texts[idx]
            meta = self.meta[idx]

            if state and meta.get("state", "").lower() != state.lower():
                continue

            if crop and crop.lower() not in text.lower():
                continue

            results.append(text)

        if not results:
            results = [self.texts[i] for i in I[0][:3] if i != -1]

        return results[:3]
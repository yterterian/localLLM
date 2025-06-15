import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH", "./vector_stores/memory_bank.index"
)
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)


class VectorManager:
    def __init__(self, dim: int = 384, persist_path: str = VECTOR_STORE_PATH):
        self.persist_path = persist_path
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.dim = dim
        self.index = self._load_or_create_index()

    def _load_or_create_index(self):
        if os.path.exists(self.persist_path):
            print(f"Loading FAISS index from {self.persist_path}")
            return faiss.read_index(self.persist_path)
        else:
            print("Creating new FAISS index")
            return faiss.IndexFlatL2(self.dim)

    def save(self):
        faiss.write_index(self.index, self.persist_path)
        print(f"FAISS index saved to {self.persist_path}")

    def add_texts(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True)
        self.index.add(np.array(vectors, dtype=np.float32))
        self.save()

    def search(self, query, top_k=5):
        query_vec = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(
            np.array(query_vec, dtype=np.float32), top_k
        )
        return indices[0], distances[0]

    def update(self, texts):
        # For simplicity, just add new texts (no deduplication)
        self.add_texts(texts)


if __name__ == "__main__":
    # Example usage: initialize and update memory bank
    vm = VectorManager()
    print("Initialized memory bank.")

    # Example: add/update with sample texts
    sample_texts = [
        "Hello world!",
        "Local LLM memory bank initialized.",
        "Test entry.",
    ]
    vm.update(sample_texts)
    print("Memory bank updated with sample texts.")

# Library sentence_transformers based on "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
import pickle
import sqlite3
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer


class CachedLLMEmbedder:
    def __init__(
        self,
        # model_name="Qwen/Qwen3-Embedding-8B",
        model_name="all-MiniLM-L6-v2",
        cache_path="concept_cache/embeddings_cache.db",
    ):
        # Create directory if it doesn't exist
        cache_dir = Path(cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.model = SentenceTransformer(model_name, device="cpu")
        self.conn = sqlite3.connect(cache_path, timeout=30.0)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, embedding BLOB)"
        )
        self.conn.commit()

    def embed_triplet(
        self, obj1: str, relation: str, obj2: str, scene_type: str = None
    ) -> torch.Tensor:
        key = (
            f"{scene_type}|{obj1}|{relation}|{obj2}"
            if scene_type
            else f"{obj1}|{relation}|{obj2}"
        )

        # Check cache
        row = self.conn.execute(
            "SELECT embedding FROM embeddings WHERE key = ?", (key,)
        ).fetchone()
        if row:
            return torch.tensor(pickle.loads(row[0]))

        # Compute and cache
        text = (
            f"In a {scene_type}, {obj1} {relation} {obj2}"
            if scene_type
            else f"{obj1} {relation} {obj2}"
        )
        embedding = torch.tensor(self.model.encode(text))

        self.conn.execute(
            "INSERT INTO embeddings (key, embedding) VALUES (?, ?)",
            (key, pickle.dumps(embedding.numpy())),
        )
        self.conn.commit()

        return embedding

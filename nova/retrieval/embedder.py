"""Embedding protocols, hashing embedder, and neural embedder."""

import hashlib
import re
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for vector embedders."""

    def embed(self, text: str) -> list[float]:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        ...


def serialize_embedding(embedding: list[float] | np.ndarray) -> bytes:
    """Serialize embedding vector to float32 BLOB bytes."""
    return np.asarray(embedding, dtype=np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize float32 BLOB bytes into a 1D numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


class HashEmbedder:
    """Deterministic 256-dimensional hash-based embedder for testing."""

    DIMENSION = 256

    def __init__(self, dimension: int = 256) -> None:
        """Initialize hash embedder with dimension."""
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a deterministic vector."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of text strings into deterministic unit vectors."""
        results: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r"\w+", text.lower())
            for token in tokens:
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimension
                sign = (
                    1.0
                    if int(hashlib.sha1(token.encode("utf-8")).hexdigest(), 16) % 2 == 0
                    else -1.0
                )
                vec[idx] += sign
                for i in range(len(token) - 2):
                    trigram = token[i : i + 3]
                    t_idx = (
                        int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16) % self.dimension
                    )
                    vec[t_idx] += 0.5 * sign

            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
            results.append(vec.tolist())
        return results


class SentenceTransformerEmbedder:
    """Local neural embedder backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initialize embedder with model name."""
        self.model_name = model_name
        self._model: Any = None

    def _load_model(self) -> Any:
        """Lazily import sentence_transformers and load model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Embed single text using sentence-transformers."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed batch of texts using sentence-transformers."""
        if not texts:
            return []
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

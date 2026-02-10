from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.types import Documents, Embeddings
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME


class NomicEmbeddingFunction(chromadb.EmbeddingFunction):
    """
    Wrapper around a SentenceTransformer model to be used as a Chroma embedding function.

    This is functionally equivalent to your original NomicEmbeddingFunction class.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        embeddings = self.model.encode(list(input), normalize_embeddings=True)
        return embeddings.tolist()


@lru_cache()
def get_embedding_function() -> chromadb.EmbeddingFunction:
    """
    Return a singleton instance of the embedding function.

    Ensures the SentenceTransformer model is loaded only once per process.
    """
    return NomicEmbeddingFunction()
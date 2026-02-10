from functools import lru_cache
from typing import Any, Dict

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from .config import CHROMA_HOST, CHROMA_PORT, COLLECTION_NAME, DEFAULT_TOP_K
from .embeddings import get_embedding_function


@lru_cache()
def get_chroma_client() -> ClientAPI:
    """
    Return a singleton Chroma client, matching your previous HttpClient config.
    """
    return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)


@lru_cache()
def get_collection() -> Collection:
    """
    Return the Chroma collection, creating it if it does not exist.

    Uses the shared embedding function, as before.
    """
    client = get_chroma_client()
    embedding_function = get_embedding_function()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )


def retrieve_documents(query: str, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    """
    Query the Chroma collection for the top_k relevant chunks.
    """
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    return results
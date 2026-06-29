"""Initialize the shared embeddings and per-notebook vector stores.

Each notebook owns one Chroma collection living inside the same
`settings.chroma_path` persist directory, selected by `collection_name`.
`lru_cache` memoizes one wrapper per collection name (and one embeddings
model per process).
"""

from functools import lru_cache

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from rag.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """Return the embeddings model (cached per process)."""
    return OllamaEmbeddings(model=settings.model_embeddings)


@lru_cache(maxsize=None)
def get_vectorstore(collection_name: str) -> Chroma:
    """Return the Chroma store for a notebook's collection (cached per name).

    `collection_name` is required on purpose — a default would silently reunite
    every notebook in one global collection and leak documents across them.
    """
    return Chroma(
        collection_name=collection_name,
        persist_directory=settings.chroma_path,
        embedding_function=get_embeddings(),
    )


def delete_collection(collection_name: str) -> None:
    """Drop a notebook's Chroma collection and evict it from the cache."""
    try:
        get_vectorstore(collection_name).delete_collection()
    finally:
        # lru_cache has no per-key eviction; clear all (entries rebuild lazily).
        get_vectorstore.cache_clear()

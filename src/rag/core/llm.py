"""Shared chat LLM (Ollama) for the agentic RAG pipeline.

Cached per process so we don't rebuild the client on every call.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from rag.config import settings


@lru_cache(maxsize=1)
def get_chat_llm() -> ChatOllama:
    """Return the chat LLM (cached per process)."""
    return ChatOllama(
        model=settings.model_llm,
        base_url=settings.ollama_base_url,
        temperature=0.0,
        num_ctx=settings.llm_num_ctx,
    )

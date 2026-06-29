"""Agentic RAG pipeline (local, no web).

An explicit reflection loop over a notebook's documents:

    retrieve(query) -> grade(question, context) -> [rewrite query & retrieve again
    if insufficient and budget remains] -> generate(answer + citations)

This replaces the old linear CrewAI flow (`crew.run_crew_ask`). It runs entirely
on the local Ollama LLM and the notebook's own Chroma collection — nothing leaves
the machine and no web search is performed.
"""

import os
import re
from typing import Callable

from langchain_core.documents import Document

from rag.config import settings
from rag.core.llm import get_chat_llm
from rag.core.vectorstore import get_vectorstore


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _doc_source_name(doc: Document) -> str:
    """Original upload name if known, else the on-disk path's basename."""
    return os.path.basename(
        doc.metadata.get("display_name") or doc.metadata.get("source", "Unknown")
    )


def _doc_key(doc: Document) -> tuple:
    """Stable identity for dedup across retrieval rounds."""
    ident = doc.metadata.get("source_id") or doc.metadata.get("source", "")
    return (ident, doc.metadata.get("page", 0), doc.page_content[:64])


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return ""
    blocks = []
    for doc in docs:
        page = doc.metadata.get("page", 0) + 1
        blocks.append(f"[Source: {_doc_source_name(doc)}, Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return "(no previous messages)"
    lines = []
    for m in history[-settings.crew_history_turns:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def _invoke(prompt: str) -> str:
    resp = get_chat_llm().invoke(prompt)
    return getattr(resp, "content", str(resp)).strip()


# --------------------------------------------------------------------------- #
# Reflection steps (one LLM call each)
# --------------------------------------------------------------------------- #
def _grade(question: str, context: str) -> bool:
    """Is the retrieved context enough to answer the question? (YES/NO)."""
    if not context.strip():
        return False
    prompt = (
        "You are a strict grader. Based ONLY on the context below, decide whether "
        "there is enough relevant information to answer the user's question.\n"
        "Answer with a single word: YES or NO.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Answer (YES or NO):"
    )
    out = _invoke(prompt).lower()
    m = re.search(r"\b(yes|no)\b", out)
    return m.group(1) == "yes" if m else False


def _rewrite_query(question: str, history: list[dict] | None, prev_query: str, context: str) -> str:
    """Produce a better search query when retrieval came up short."""
    prompt = (
        "The previous document search did not return enough relevant information.\n"
        "Rewrite the user's question into an improved search query: add precise keywords "
        "and synonyms, drop filler words. Return ONLY the query text, nothing else.\n\n"
        f"Conversation (context only):\n{_format_history(history)}\n\n"
        f"Original question: {question}\n"
        f"Previous query: {prev_query}\n"
        "Improved query:"
    )
    out = _invoke(prompt)
    # Keep first non-empty line, strip surrounding quotes.
    line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    line = line.strip().strip('"').strip("'").strip()
    return line or question


def _generate(question: str, history: list[dict] | None, context: str) -> str:
    """Compose the final answer, grounded strictly in the context."""
    prompt = (
        "You are a precise assistant answering ONLY from the provided document context.\n"
        "Rules:\n"
        "- Use ONLY the context. Do NOT use outside knowledge.\n"
        "- If the context does not contain the answer, say clearly that the information "
        "was not found in the documents (in the user's language).\n"
        "- Answer in the SAME language as the question (Vietnamese or English).\n"
        "- At the end, list the references used as 'References: <file> (page N)'.\n"
        "- The conversation history is for context/pronoun resolution only, not a source.\n\n"
        f"Conversation:\n{_format_history(history)}\n\n"
        f"Document context:\n{context if context.strip() else '(no documents retrieved)'}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return _invoke(prompt)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def answer_question(
    question: str,
    collection_name: str,
    history: list[dict] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict:
    """Answer a question against one notebook's documents using a reflection loop.

    Returns {"answer": str, "sources": [{"source","page"}], "iterations": int}.
    """

    def _say(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    vs = get_vectorstore(collection_name)
    query = question
    collected: list[Document] = []
    seen: set = set()
    iterations = 0

    # 1 initial retrieval + up to `max_reflection_iters` rewrites.
    for attempt in range(settings.max_reflection_iters + 1):
        iterations = attempt + 1
        _say("🔎 Đang tìm tài liệu liên quan…" if attempt == 0 else "🔎 Tìm lại với truy vấn mới…")
        try:
            docs = vs.similarity_search(query, k=settings.search_k)
        except Exception as e:
            docs = []
            _say(f"⚠️ Lỗi tìm kiếm: {e}")
        for d in docs:
            key = _doc_key(d)
            if key not in seen:
                seen.add(key)
                collected.append(d)

        context = _format_context(collected)
        _say("🧪 Đánh giá độ liên quan…")
        if _grade(question, context):
            break
        if attempt < settings.max_reflection_iters:
            _say("♻️ Tinh chỉnh truy vấn…")
            query = _rewrite_query(question, history, query, context)

    _say("✍️ Đang soạn câu trả lời…")
    answer = _generate(question, history, _format_context(collected))

    sources = [
        {
            "source": d.metadata.get("display_name") or d.metadata.get("source", "Unknown"),
            "page": d.metadata.get("page", 0),
        }
        for d in collected
    ]
    return {"answer": answer, "sources": sources, "iterations": iterations}

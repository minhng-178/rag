"""Service for ingesting PDF documents into the vector store.

Consolidates the ingestion logic that was duplicated between the API (main.py)
and the MCP server (mcp_server.py). The function raises plain exceptions
(FileNotFoundError / ValueError); the caller maps them to an HTTP response or
text depending on the interface.
"""

import os
from typing import Callable

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config import settings
from rag.core.vectorstore import get_vectorstore


def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )


def ingest_folder(folder_path: str | None = None) -> dict:
    """Load all PDFs in `folder_path` into the vector store.

    Returns:
        dict: {"chunks": <number of chunks ingested>, "loaded_files": [<file paths>]}

    Raises:
        FileNotFoundError: the folder does not exist.
        ValueError: no PDF files were found.
    """
    folder_path = folder_path or settings.default_folder

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path '{folder_path}' does not exist.")

    loader = DirectoryLoader(
        folder_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDF documents found in '{folder_path}'.")

    chunks = _make_splitter().split_documents(documents)

    # Batch adding documents to avoid Ollama connection errors / tokenization limits
    batch_size = settings.ingest_batch_size
    for i in range(0, len(chunks), batch_size):
        get_vectorstore("langchain").add_documents(chunks[i : i + batch_size])

    loaded_files = sorted({doc.metadata.get("source", "") for doc in chunks})

    return {"chunks": len(chunks), "loaded_files": loaded_files}


def ingest_files(
    file_paths: list[str],
    collection_name: str,
    source_ids: list[str] | None = None,
    progress_cb: Callable[[str, float], None] | None = None,
) -> dict:
    """Ingest one or more PDF files (by path) into a notebook's collection.

    Processes one file at a time to keep memory low. `source_ids[i]` ties the
    chunks of `file_paths[i]` to a `sources` row via metadata + deterministic
    Chroma ids (so a source can later be deleted precisely).

    Returns:
        {"chunks": <total>, "files": [{"path","source_id","chunks","pages","status","error"}]}
    """
    vs = get_vectorstore(collection_name)
    splitter = _make_splitter()
    batch_size = settings.ingest_batch_size

    results: list[dict] = []
    total_chunks = 0
    n = max(len(file_paths), 1)

    def _emit(label: str, file_idx: int, within: float) -> None:
        if progress_cb:
            progress_cb(label, min(1.0, (file_idx + within) / n))

    for idx, path in enumerate(file_paths):
        sid = source_ids[idx] if source_ids else None
        display_name = os.path.basename(path)
        info = {
            "path": path,
            "source_id": sid,
            "chunks": 0,
            "pages": 0,
            "status": "done",
            "error": None,
        }
        try:
            _emit(f"Loading {display_name}…", idx, 0.0)
            pages = PyPDFLoader(path).load()
            info["pages"] = len(pages)

            chunks = splitter.split_documents(pages)
            for c in chunks:
                c.metadata["source_id"] = sid
                c.metadata["display_name"] = display_name

            nbatches = max((len(chunks) + batch_size - 1) // batch_size, 1)
            for b, i in enumerate(range(0, len(chunks), batch_size)):
                batch = chunks[i : i + batch_size]
                ids = (
                    [f"{sid}:{i + j}" for j in range(len(batch))] if sid else None
                )
                vs.add_documents(batch, ids=ids)
                _emit(f"Embedding {display_name}…", idx, (b + 1) / nbatches)

            info["chunks"] = len(chunks)
            total_chunks += len(chunks)
        except Exception as e:  # one bad PDF must not abort the batch
            info["status"] = "error"
            info["error"] = str(e)
        results.append(info)

    _emit("Done", n, 0.0)
    return {"chunks": total_chunks, "files": results}


def delete_source_vectors(collection_name: str, source_id: str, num_chunks: int) -> None:
    """Remove a source's chunks from its collection using deterministic ids."""
    if not num_chunks:
        return
    ids = [f"{source_id}:{i}" for i in range(num_chunks)]
    get_vectorstore(collection_name).delete(ids=ids)

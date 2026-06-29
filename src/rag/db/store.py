"""SQLite persistence for notebooks, sources, conversations and messages.

Stdlib `sqlite3` only — no extra dependency. A short-lived connection is opened
per call (see `_conn`) so nothing is shared across Streamlit's reruns/threads.

This module is UI-agnostic: do NOT import Streamlit here.
"""

import json
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from rag.config import settings

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _new_id() -> str:
    return uuid.uuid4().hex


@contextmanager
def _conn():
    """Open a connection for a single unit of work, committing on success."""
    con = sqlite3.connect(settings.db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")  # required for ON DELETE CASCADE
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create data dirs and tables. Idempotent; call once at app start."""
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    schema = _SCHEMA_PATH.read_text(encoding="utf-8")
    with _conn() as con:
        con.execute("PRAGMA journal_mode = WAL")
        con.executescript(schema)


# --------------------------------------------------------------------------- #
# Notebooks
# --------------------------------------------------------------------------- #
def create_notebook(name: str) -> dict:
    nb_id = _new_id()
    collection_name = f"nb_{nb_id[:20]}"  # safe Chroma collection name
    with _conn() as con:
        con.execute(
            "INSERT INTO notebooks (id, name, collection_name) VALUES (?, ?, ?)",
            (nb_id, name.strip() or "Untitled notebook", collection_name),
        )
    return get_notebook(nb_id)


def list_notebooks() -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM notebooks ORDER BY updated_at DESC, created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_notebook(notebook_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
    return dict(row) if row else None


def rename_notebook(notebook_id: str, name: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE notebooks SET name = ?, updated_at = datetime('now') WHERE id = ?",
            (name.strip() or "Untitled notebook", notebook_id),
        )


def touch_notebook(notebook_id: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE notebooks SET updated_at = datetime('now') WHERE id = ?",
            (notebook_id,),
        )


def delete_notebook(notebook_id: str) -> dict:
    """Delete a notebook (cascades sources/conversations/messages in SQLite).

    Returns {"collection_name", "stored_paths"} so the caller can clean up the
    Chroma collection and uploaded files, which the SQL cascade does not touch.
    """
    with _conn() as con:
        nb = con.execute(
            "SELECT collection_name FROM notebooks WHERE id = ?", (notebook_id,)
        ).fetchone()
        paths = [
            r["stored_path"]
            for r in con.execute(
                "SELECT stored_path FROM sources WHERE notebook_id = ?", (notebook_id,)
            ).fetchall()
        ]
        con.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
    return {
        "collection_name": nb["collection_name"] if nb else None,
        "stored_paths": paths,
    }


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def add_source(notebook_id: str, filename: str, stored_path: str, bytes_: int) -> dict:
    src_id = _new_id()
    with _conn() as con:
        con.execute(
            "INSERT INTO sources (id, notebook_id, filename, stored_path, bytes) "
            "VALUES (?, ?, ?, ?, ?)",
            (src_id, notebook_id, filename, stored_path, bytes_),
        )
    return get_source(src_id)


def get_source(source_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
    return dict(row) if row else None


def update_source(source_id: str, **fields) -> None:
    if not fields:
        return
    allowed = {"filename", "stored_path", "num_chunks", "num_pages", "bytes", "status", "error"}
    cols = [k for k in fields if k in allowed]
    if not cols:
        return
    assignments = ", ".join(f"{c} = ?" for c in cols)
    values = [fields[c] for c in cols] + [source_id]
    with _conn() as con:
        con.execute(f"UPDATE sources SET {assignments} WHERE id = ?", values)


def list_sources(notebook_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sources WHERE notebook_id = ? ORDER BY created_at ASC",
            (notebook_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def delete_source(source_id: str) -> dict:
    """Delete a source row; returns it so the caller can drop vectors + file."""
    src = get_source(source_id)
    with _conn() as con:
        con.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    return src or {}


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #
def create_conversation(notebook_id: str, title: str = "New chat") -> dict:
    conv_id = _new_id()
    with _conn() as con:
        con.execute(
            "INSERT INTO conversations (id, notebook_id, title) VALUES (?, ?, ?)",
            (conv_id, notebook_id, title),
        )
    return get_conversation(conv_id)


def get_conversation(conversation_id: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return dict(row) if row else None


def list_conversations(notebook_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM conversations WHERE notebook_id = ? "
            "ORDER BY updated_at DESC, created_at DESC",
            (notebook_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def rename_conversation(conversation_id: str, title: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE conversations SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title.strip() or "New chat", conversation_id),
        )


def touch_conversation(conversation_id: str) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE conversations SET updated_at = datetime('now') WHERE id = ?",
            (conversation_id,),
        )


def delete_conversation(conversation_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# --------------------------------------------------------------------------- #
# Messages
# --------------------------------------------------------------------------- #
def add_message(
    conversation_id: str, role: str, content: str, sources: list | None = None
) -> dict:
    msg_id = _new_id()
    sources_json = json.dumps(sources) if sources else None
    with _conn() as con:
        con.execute(
            "INSERT INTO messages (id, conversation_id, role, content, sources_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, sources_json),
        )
    touch_conversation(conversation_id)
    return {"id": msg_id, "role": role, "content": content, "sources": sources or []}


def list_messages(conversation_id: str) -> list[dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC, rowid ASC",
            (conversation_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources_json"]) if d.get("sources_json") else []
        out.append(d)
    return out

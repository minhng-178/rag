"""NotebookLM-style RAG app — a single Streamlit process that calls the
`rag` services directly (no separate backend).

Each notebook owns its own sources (PDFs) and conversations. Sources are
isolated per notebook via a dedicated Chroma collection; chat history is
persisted in SQLite so it survives refreshes and restarts.
"""

import os
import sys
from pathlib import Path

# Ensure `rag` is importable even without an editable install.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

from rag.config import settings
from rag.core.vectorstore import delete_collection
from rag.db import store
from rag.services.agentic_rag import answer_question
from rag.services.ingestion import delete_source_vectors, ingest_files

# --- Page config & init ---
st.set_page_config(
    page_title="NotebookLM-style RAG",
    page_icon="📓",
    layout="wide",
    initial_sidebar_state="expanded",
)
store.init_db()

st.markdown(
    """
    <style>
      .title-gradient {
        background: linear-gradient(135deg, #06b6d4 0%, #0ea5e9 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        font-weight: 800; font-size: 2rem; margin-bottom: 0.1rem;
      }
      .subtitle { color: #64748b; font-size: 0.95rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State helpers — never store DB rows in session_state; always re-read.
# --------------------------------------------------------------------------- #
def _ensure_notebook_state() -> dict | None:
    """Return the current notebook dict, re-validating the id every rerun."""
    notebooks = store.list_notebooks()
    if not notebooks:
        st.session_state.current_notebook_id = None
        st.session_state.current_conversation_id = None
        return None

    ids = {nb["id"] for nb in notebooks}
    if st.session_state.get("current_notebook_id") not in ids:
        st.session_state.current_notebook_id = notebooks[0]["id"]
        st.session_state.current_conversation_id = None

    return next(nb for nb in notebooks if nb["id"] == st.session_state.current_notebook_id)


def _ensure_conversation_state(notebook_id: str) -> dict:
    """Return the current conversation, creating/selecting one if needed."""
    convs = store.list_conversations(notebook_id)
    if not convs:
        conv = store.create_conversation(notebook_id)
        st.session_state.current_conversation_id = conv["id"]
        return conv

    ids = {c["id"] for c in convs}
    if st.session_state.get("current_conversation_id") not in ids:
        st.session_state.current_conversation_id = convs[0]["id"]

    return next(c for c in convs if c["id"] == st.session_state.current_conversation_id)


def _select_notebook(notebook_id: str) -> None:
    st.session_state.current_notebook_id = notebook_id
    st.session_state.current_conversation_id = None


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("📎 References"):
        for s in sources:
            name = os.path.basename(s.get("source", "")) or s.get("source", "")
            page = (s.get("page", 0) or 0) + 1
            st.markdown(f"- 📄 **{name}** (page {page})")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 📓 Notebooks")

    notebooks = store.list_notebooks()
    if notebooks:
        labels = {nb["id"]: nb["name"] for nb in notebooks}
        ids = [nb["id"] for nb in notebooks]
        current_id = st.session_state.get("current_notebook_id") or ids[0]
        idx = ids.index(current_id) if current_id in ids else 0
        chosen = st.selectbox(
            "Active notebook",
            options=ids,
            index=idx,
            format_func=lambda i: labels.get(i, "?"),
            label_visibility="collapsed",
        )
        if chosen != st.session_state.get("current_notebook_id"):
            _select_notebook(chosen)
            st.rerun()
    else:
        st.info("Create your first notebook to get started.")

    with st.form("new_notebook", clear_on_submit=True):
        new_name = st.text_input("New notebook name", placeholder="e.g. Research papers")
        if st.form_submit_button("➕ Create notebook", use_container_width=True):
            nb = store.create_notebook(new_name or "Untitled notebook")
            _select_notebook(nb["id"])
            st.rerun()

current_notebook = _ensure_notebook_state()

if current_notebook is None:
    st.markdown('<h1 class="title-gradient">📓 NotebookLM-style RAG</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Create a notebook in the sidebar, upload PDFs, and chat with them.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

nb_id = current_notebook["id"]
collection = current_notebook["collection_name"]

with st.sidebar:
    # --- Notebook actions ---
    with st.expander("⚙️ Notebook settings"):
        renamed = st.text_input("Rename notebook", value=current_notebook["name"], key=f"rn_{nb_id}")
        col_a, col_b = st.columns(2)
        if col_a.button("Save name", use_container_width=True, key=f"save_{nb_id}"):
            store.rename_notebook(nb_id, renamed)
            st.rerun()
        if col_b.button("🗑️ Delete", use_container_width=True, key=f"del_{nb_id}"):
            st.session_state[f"confirm_del_{nb_id}"] = True
        if st.session_state.get(f"confirm_del_{nb_id}"):
            st.warning("Delete this notebook and all its sources & chats?")
            if st.button("Yes, delete permanently", type="primary", key=f"yesdel_{nb_id}"):
                info = store.delete_notebook(nb_id)
                if info.get("collection_name"):
                    try:
                        delete_collection(info["collection_name"])
                    except Exception as e:
                        st.warning(f"Could not drop vectors: {e}")
                for p in info.get("stored_paths", []):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
                st.session_state.pop(f"confirm_del_{nb_id}", None)
                _select_notebook(None)
                st.rerun()

    st.markdown("---")

    # --- Sources / upload ---
    st.markdown("### 📂 Sources")
    sources = store.list_sources(nb_id)

    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{nb_id}",
    )
    if uploaded and st.button("⚡ Ingest selected", use_container_width=True, key=f"ingest_{nb_id}"):
        existing_names = {s["filename"] for s in sources}
        nb_upload_dir = Path(settings.upload_dir) / nb_id
        nb_upload_dir.mkdir(parents=True, exist_ok=True)

        progress = st.progress(0.0, text="Starting…")

        def _cb(label: str, frac: float) -> None:
            progress.progress(frac, text=label)

        ingested, skipped = 0, 0
        for uf in uploaded:
            if uf.name in existing_names:
                skipped += 1
                continue
            data = uf.getvalue()
            src = store.add_source(nb_id, uf.name, "", len(data))
            stored_path = str(nb_upload_dir / f"{src['id']}__{uf.name}")
            with open(stored_path, "wb") as fh:
                fh.write(data)
            store.update_source(src["id"], stored_path=stored_path, status="ingesting")

            res = ingest_files([stored_path], collection, source_ids=[src["id"]], progress_cb=_cb)
            f = res["files"][0]
            store.update_source(
                src["id"],
                num_chunks=f["chunks"],
                num_pages=f["pages"],
                status=f["status"],
                error=f["error"],
            )
            ingested += 1

        progress.empty()
        store.touch_notebook(nb_id)
        msg = f"Ingested {ingested} file(s)."
        if skipped:
            msg += f" Skipped {skipped} already-present file(s)."
        st.success(msg)
        st.rerun()

    if sources:
        for s in sources:
            icon = {"done": "📄", "error": "⚠️", "ingesting": "⏳"}.get(s["status"], "📄")
            cols = st.columns([0.8, 0.2])
            label = f"{icon} {s['filename']}"
            if s["status"] == "done":
                label += f"  · {s['num_chunks']} chunks"
            elif s["status"] == "error":
                label += "  · failed"
            cols[0].markdown(label)
            if cols[1].button("✕", key=f"delsrc_{s['id']}", help="Remove source"):
                try:
                    delete_source_vectors(collection, s["id"], s["num_chunks"])
                except Exception as e:
                    st.warning(f"Could not remove vectors: {e}")
                if s["stored_path"]:
                    try:
                        os.remove(s["stored_path"])
                    except OSError:
                        pass
                store.delete_source(s["id"])
                st.rerun()
    else:
        st.caption("No sources yet. Upload a PDF above.")

    st.markdown("---")

    # --- Conversations ---
    st.markdown("### 💬 Conversations")
    if st.button("➕ New chat", use_container_width=True, key=f"newchat_{nb_id}"):
        conv = store.create_conversation(nb_id)
        st.session_state.current_conversation_id = conv["id"]
        st.rerun()

    conversations = store.list_conversations(nb_id)
    current_conv = _ensure_conversation_state(nb_id)
    for c in conversations:
        is_active = c["id"] == current_conv["id"]
        cols = st.columns([0.8, 0.2])
        if cols[0].button(
            ("▶ " if is_active else "") + (c["title"] or "New chat"),
            key=f"conv_{c['id']}",
            use_container_width=True,
        ):
            st.session_state.current_conversation_id = c["id"]
            st.rerun()
        if cols[1].button("✕", key=f"delconv_{c['id']}", help="Delete chat"):
            store.delete_conversation(c["id"])
            if st.session_state.get("current_conversation_id") == c["id"]:
                st.session_state.current_conversation_id = None
            st.rerun()


# --------------------------------------------------------------------------- #
# Main pane — chat
# --------------------------------------------------------------------------- #
current_conv = _ensure_conversation_state(nb_id)
conv_id = current_conv["id"]
n_sources = len(store.list_sources(nb_id))

st.markdown(f'<h1 class="title-gradient">📓 {current_notebook["name"]}</h1>', unsafe_allow_html=True)
st.markdown(
    f'<p class="subtitle">{n_sources} source(s) · chatting in "{current_conv["title"]}". '
    "Answers are grounded only in this notebook\'s documents.</p>",
    unsafe_allow_html=True,
)

# Rename current conversation inline
with st.expander("✏️ Rename this conversation"):
    new_title = st.text_input("Title", value=current_conv["title"], key=f"convtitle_{conv_id}")
    if st.button("Save title", key=f"savetitle_{conv_id}"):
        store.rename_conversation(conv_id, new_title)
        st.rerun()

# History from DB
for msg in store.list_messages(conv_id):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))

# Input
if prompt := st.chat_input("Ask something about this notebook's documents..."):
    if n_sources == 0:
        st.warning("Upload at least one PDF to this notebook before asking.")
        st.stop()

    store.add_message(conv_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Title the conversation from the first question.
    if (current_conv["title"] or "New chat") == "New chat":
        store.rename_conversation(conv_id, prompt[:40])

    history = store.list_messages(conv_id)[:-1]  # exclude the just-added question

    with st.chat_message("assistant"):
        status = st.status("Agent đang xử lý…", expanded=True)

        def _progress(msg: str) -> None:
            status.write(msg)
            status.update(label=msg)

        try:
            res = answer_question(prompt, collection, history, progress_cb=_progress)
            answer = res.get("answer", "")
            src = res.get("sources", [])
            iters = res.get("iterations", 1)
            status.update(
                label=f"✅ Xong ({iters} vòng truy vấn)", state="complete", expanded=False
            )
            st.markdown(answer)
            _render_sources(src)
            store.add_message(conv_id, "assistant", answer, src)
        except Exception as e:
            status.update(label="⚠️ Lỗi", state="error")
            err = f"⚠️ Error: {e}"
            st.error(err)
            store.add_message(conv_id, "assistant", err)
    st.rerun()

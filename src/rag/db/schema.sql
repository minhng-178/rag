-- Schema for the NotebookLM-style app. Idempotent: safe to executescript on every boot.

CREATE TABLE IF NOT EXISTS notebooks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    collection_name TEXT NOT NULL UNIQUE,        -- Chroma collection, e.g. nb_<uuid[:20]>
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,                   -- original display name
    stored_path TEXT NOT NULL,                   -- absolute path on disk
    num_chunks  INTEGER NOT NULL DEFAULT 0,
    num_pages   INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'pending', -- pending | ingesting | done | error
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_sources_nb ON sources(notebook_id);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New chat',
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_conv_nb ON conversations(notebook_id);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,               -- user | assistant
    content         TEXT NOT NULL,
    sources_json    TEXT,                        -- JSON list of {"source","page"}
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, created_at);

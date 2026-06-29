"""Centralized configuration for the whole project.

Reads from environment variables / the `.env` file (see `.env.example`).
Every other module imports `settings` from here instead of hardcoding values.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root directory (…/rag), relative to src/rag/config.py
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama / model ---
    model_llm: str = "llama3"
    model_embeddings: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"

    # --- Data paths ---
    chroma_path: str = str(BASE_DIR / "chroma_db")
    default_folder: str = str(BASE_DIR / "papers")

    # --- App data (notebooks / chat history / uploads) ---
    data_dir: str = str(BASE_DIR / "data")
    db_path: str = str(BASE_DIR / "data" / "app.db")
    upload_dir: str = str(BASE_DIR / "data" / "uploads")

    # --- Text splitting (RAG) ---
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- RAG / ingestion knobs ---
    ingest_batch_size: int = 16
    crew_history_turns: int = 6
    search_k: int = 5

    # --- Agentic RAG (reflection loop) ---
    max_reflection_iters: int = 2  # max query rewrites after the first retrieval
    llm_num_ctx: int = 8192

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Backend URL the Streamlit frontend calls
    api_base_url: str = "http://localhost:8000"


settings = Settings()

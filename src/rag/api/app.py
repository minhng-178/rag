"""FastAPI app: document ingestion + RAG question answering.

Run: `uvicorn rag.api.app:app --reload`
"""

from fastapi import FastAPI, HTTPException

from rag.config import settings
from rag.api.schemas import AskRequest, AskResponse, IngestResponse
from rag.services.crew import run_crew_ask
from rag.services.ingestion import ingest_folder

app = FastAPI(
    title="RAG Backend API - Batch Ingestion",
    description="This API allows you to ingest documents and perform retrieval-augmented generation (RAG) queries.",
    version="1.0.0",
)


@app.post("/ingest-local-folder", response_model=IngestResponse)
def ingest_local_folder(folder_path: str = settings.default_folder):
    """Ingest documents from a local folder into the vector store."""
    try:
        result = ingest_folder(folder_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message=f"Successfully ingested {result['chunks']} chunks from '{folder_path}'.",
        loaded_files=result["loaded_files"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Ask a question and get an answer based on the ingested documents."""
    try:
        res = run_crew_ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        question=request.question,
        answer=res.get("answer", ""),
        sources=res.get("sources", []),
    )

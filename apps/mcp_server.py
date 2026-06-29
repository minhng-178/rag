"""RAG MCP Server.

Run: `python apps/mcp_server.py`
"""

import asyncio
import os

from mcp import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from rag.config import settings
from rag.services.crew import run_crew_ask
from rag.services.ingestion import ingest_folder

server = Server("rag-server")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """MCP client."""
    return [
        types.Tool(
            name="ingest_documents",
            description=(
                "Load all PDF files from a folder into ChromaDB vector database. "
                "Call this tool before asking questions to ensure documents are indexed."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "folder_path": {
                        "type": "string",
                        "description": (
                            "Absolute or relative path to the folder containing PDF files. "
                            f"Default: {settings.default_folder}"
                        ),
                        "default": settings.default_folder,
                    }
                },
                "required": [],
            },
        ),
        types.Tool(
            name="ask_question",
            description=(
                "Ask a question about documents that have been ingested into the system. "
                "Returns an answer based on document content along with source references."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer based on the documents.",
                    }
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls from MCP client."""

    if name == "ingest_documents":
        folder_path = arguments.get("folder_path", settings.default_folder)

        try:
            result = ingest_folder(folder_path)
        except FileNotFoundError as exc:
            return [types.TextContent(type="text", text=f"Error: {exc}")]
        except ValueError as exc:
            return [types.TextContent(type="text", text=f"Warning: {exc}")]
        except Exception as exc:
            return [types.TextContent(type="text", text=f"Error during ingestion: {exc}")]

        file_names = [os.path.basename(f) for f in result["loaded_files"]]
        text = (
            f"Successfully ingested {result['chunks']} chunks from {len(file_names)} file(s):\n"
            + "\n".join(f"  - {name_}" for name_ in file_names)
        )
        return [types.TextContent(type="text", text=text)]

    elif name == "ask_question":
        question = arguments.get("question", "").strip()
        if not question:
            return [types.TextContent(type="text", text="Error: Question cannot be empty.")]

        try:
            res = run_crew_ask(question)
            answer = res.get("answer", "(No answer found)")
            sources = res.get("sources", [])

            # Collect unique sources
            seen = set()
            sources_list = []
            for src_item in sources:
                src = src_item.get("source", "Unknown")
                page = src_item.get("page", 0) + 1
                key = f"{src}:{page}"
                if key not in seen:
                    seen.add(key)
                    sources_list.append(f"  - {os.path.basename(src)} (page {page})")

            result = answer
            if sources_list:
                result += "\n\n--- Sources ---\n" + "\n".join(sources_list)

            return [types.TextContent(type="text", text=result)]

        except Exception as exc:
            return [types.TextContent(type="text", text=f"Error answering question: {exc}")]

    else:
        return [types.TextContent(type="text", text=f"Error: Unknown tool '{name}'")]


# ── Entry point ────────────────────────────────────────────────────────────
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())

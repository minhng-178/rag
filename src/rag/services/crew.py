"""CrewAI team (Researcher + Writer) that answers questions based on local documents."""

import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

# Monkeypatch: report supports_function_calling = False for llama3 to force CrewAI
# to use a prompt-based ReAct loop instead of native tool calling.
_original_supports_function_calling = LLM.supports_function_calling

def custom_supports_function_calling(self) -> bool:
    if self.model and "llama3" in self.model:
        return False
    return _original_supports_function_calling(self)

LLM.supports_function_calling = custom_supports_function_calling

from rag.config import settings  # noqa: E402
from rag.core.vectorstore import get_vectorstore  # noqa: E402


def _doc_source_name(doc) -> str:
    """Prefer the original upload name, falling back to the on-disk path."""
    return os.path.basename(
        doc.metadata.get("display_name") or doc.metadata.get("source", "Unknown")
    )


def _format_history(history: list[dict] | None) -> str:
    """Render recent turns as a short transcript for context only."""
    if not history:
        return "(no previous messages)"
    lines = []
    for m in history[-settings.crew_history_turns:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {m.get('content', '')}")
    return "\n".join(lines)


def run_crew_ask(
    question: str,
    collection_name: str,
    history: list[dict] | None = None,
) -> dict:
    """
    Runs a CrewAI team (Researcher + Writer) to answer a question using ONLY the
    documents in the given notebook's Chroma collection.
    """
    # 0. Search tool scoped to THIS notebook's collection (closure — no global state).
    @tool("Search Local Documents")
    def search_local_documents(query: str) -> str:
        """
        Search this notebook's PDF documents for information related to the query.
        Returns relevant chunks with source filenames and page numbers.
        """
        try:
            docs = get_vectorstore(collection_name).similarity_search(query, k=settings.search_k)
            if not docs:
                return "No matching documents found in the database."
            results = []
            for doc in docs:
                source = _doc_source_name(doc)
                page = doc.metadata.get("page", 0) + 1
                results.append(f"[Source: {source}, Page: {page}]\nContent: {doc.page_content}")
            return "\n\n---\n\n".join(results)
        except Exception as e:
            return f"Error searching documents: {str(e)}"

    # 1. Fetch source documents directly to return structured source list to the frontend
    sources = []
    try:
        docs = get_vectorstore(collection_name).similarity_search(question, k=settings.search_k)
        for doc in docs:
            sources.append({
                "source": doc.metadata.get("display_name") or doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0)
            })
    except Exception as e:
        print(f"Error fetching sources directly: {e}")

    # 2. Configure the LLM for CrewAI
    local_llm = LLM(
        model=f"ollama/{settings.model_llm}",
        base_url=settings.ollama_base_url,
        temperature=0.0,
        timeout=300,
        num_ctx=8192
    )

    # 3. Define Researcher Agent
    researcher = Agent(
        role="Document Research Specialist",
        goal="Search and gather the most accurate information from the local document database to answer the question",
        backstory=(
            "You are an outstanding document researcher. Your job is to analyze the user's "
            "question, search for relevant information in the internal documents using the search tool, "
            "and faithfully synthesize the core facts. You NEVER fabricate information that is not in the documents."
        ),
        tools=[search_local_documents],
        llm=local_llm,
        verbose=True
    )

    # 4. Define Writer Agent
    writer = Agent(
        role="Writer and answer editor",
        goal="Compose a complete, coherent, and professional answer based on the research information provided",
        backstory=(
            "You are an outstanding editor. You take the raw research information from the Research Specialist, "
            "restructure the layout, and refine the wording to produce the most accurate, coherent, and natural answer "
            "in the language of the user's query (usually Vietnamese or English). "
            "Always cite the reference sources (file name and page) at the end of the answer. "
            "If the research information contains no data to answer the question, state clearly that no information was found in the documents."
        ),
        llm=local_llm,
        verbose=True
    )

    # 5. Define Tasks
    research_task = Task(
        description=(
            "Recent conversation (for context and pronoun resolution only — do NOT treat it as a source):\n"
            "{history}\n\n"
            "Research the local documents to find an answer to the following question: '{question}'. "
            "Use the 'Search Local Documents' tool to find relevant information. "
            "Extract all relevant detailed information and note its origin clearly (file name, page number)."
        ),
        expected_output="Detailed information relevant to the question, extracted from the documents, together with specific sources (file name and page number).",
        agent=researcher
    )

    write_task = Task(
        description=(
            "Use the information from the Research Task to write a complete answer to the question: '{question}'. "
            "The answer must be clear, coherent, directly address the question, and be in the language of the user's query "
            "(if the user asks in Vietnamese, write the response in Vietnamese; if in English, write in English). "
            "At the end of the answer, list in detail the reference sources used (e.g., 'References: document.pdf (page 3)' or 'Tài liệu tham khảo: document.pdf (trang 3)'). "
            "If the information is not in the research documents, state clearly that no information was found."
        ),
        expected_output="A complete, coherent answer in the language of the user's query, with detailed references at the end.",
        agent=writer
    )

    # 6. Define Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute crew
    result = crew.kickoff(inputs={"question": question, "history": _format_history(history)})

    return {
        "answer": str(result),
        "sources": sources
    }

"""
Agentic RAG logic — Pydantic AI agent with registered search tools.

The agent is initialised with the configured LLM provider and exposes
tool-augmented methods for vector search, hybrid search, and document
retrieval.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT
from .providers import get_llm_model
from .tools import (
    vector_search_tool,
    hybrid_search_tool,
    get_document_tool,
    list_documents_tool,
    VectorSearchInput,
    HybridSearchInput,
    DocumentInput,
    DocumentListInput,
)

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent dependencies
# ---------------------------------------------------------------------------

@dataclass
class AgentDependencies:
    """Runtime dependencies injected into every agent invocation."""
    session_id: str
    user_id: Optional[str] = None
    search_preferences: Dict[str, Any] = None
    retrieved_chunks: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.search_preferences is None:
            self.search_preferences = {
                "use_vector": True,
                "default_limit": 10,
            }
        if self.retrieved_chunks is None:
            self.retrieved_chunks = []


# ---------------------------------------------------------------------------
# Agent initialisation
# ---------------------------------------------------------------------------

rag_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Tool registrations
# ---------------------------------------------------------------------------

@rag_agent.tool
async def vector_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search for relevant information using semantic similarity.

    This tool performs vector similarity search across document chunks
    to find semantically related content. Returns the most relevant results
    regardless of similarity score.

    Args:
        query: Search query to find similar content
        limit: Maximum number of results to return (1-50)

    Returns:
        List of matching chunks ordered by similarity (best first)
    """
    safe_limit = min(max(1, limit), 4)
    input_data = VectorSearchInput(query=query, limit=safe_limit)
    results = await vector_search_tool(input_data)

    formatted = [
        {
            "content": r.content[:1500] if len(r.content) > 1500 else r.content,
            "score": round(r.score, 4),
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]
    # Track retrieved chunks for evaluation
    ctx.deps.retrieved_chunks.extend(formatted)
    return formatted


@rag_agent.tool
async def hybrid_search(
    ctx: RunContext[AgentDependencies],
    query: str,
    limit: int = 4,
    text_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Perform both vector and keyword search for comprehensive results.

    This tool combines semantic similarity search with keyword matching
    for the best coverage. It ranks results using both vector similarity
    and text matching scores.

    Args:
        query: Search query for hybrid search
        limit: Maximum number of results to return (1-10)
        text_weight: Weight for text similarity vs vector similarity (0.0-1.0)

    Returns:
        List of chunks ranked by combined relevance score
    """
    safe_limit = min(max(1, limit), 4)
    input_data = HybridSearchInput(
        query=query, limit=safe_limit, text_weight=text_weight
    )
    results = await hybrid_search_tool(input_data)

    formatted = [
        {
            "content": r.content[:1500] if len(r.content) > 1500 else r.content,
            "score": round(r.score, 4),
            "document_title": r.document_title,
            "document_source": r.document_source,
            "chunk_id": r.chunk_id,
        }
        for r in results
    ]
    # Track retrieved chunks for evaluation
    ctx.deps.retrieved_chunks.extend(formatted)
    return formatted


@rag_agent.tool
async def get_document(
    ctx: RunContext[AgentDependencies],
    document_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Retrieve the complete content of a specific document.

    This tool fetches the full document content along with all its chunks
    and metadata.

    Args:
        document_id: UUID of the document to retrieve

    Returns:
        Complete document data with content and metadata, or None if not found
    """
    input_data = DocumentInput(document_id=document_id)
    document = await get_document_tool(input_data)

    if document:
        content = document.get("content", "")
        if len(content) > 3000:
            content = content[:3000] + " ... [truncated]"
        return {
            "id": document["id"],
            "title": document["title"],
            "source": document["source"],
            "content": content,
            "chunk_count": len(document.get("chunks", [])),
            "created_at": document["created_at"],
        }
    return None


@rag_agent.tool
async def list_documents(
    ctx: RunContext[AgentDependencies],
    limit: int = 20,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    List available documents with their metadata.

    This tool provides an overview of all documents in the knowledge base.

    Args:
        limit: Maximum number of documents to return (1-100)
        offset: Number of documents to skip for pagination

    Returns:
        List of documents with metadata and chunk counts
    """
    input_data = DocumentListInput(limit=limit, offset=offset)
    documents = await list_documents_tool(input_data)

    return [
        {
            "id": d.id,
            "title": d.title,
            "source": d.source,
            "chunk_count": d.chunk_count,
            "created_at": d.created_at.isoformat(),
        }
        for d in documents
    ]

"""
Agent tools — vector search, hybrid search, document retrieval.

Each tool function is a thin wrapper that embeds the query via the
configured embedding provider and delegates to the database layer.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from .db_utils import (
    vector_search,
    hybrid_search,
    get_document,
    list_documents,
    get_document_chunks,
)
from .models import ChunkResult, DocumentMetadata
from .providers import get_embedding_provider

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

async def generate_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for *text* using the active provider.

    Works transparently with both local (SentenceTransformer) and
    OpenAI embeddings depending on ``EMBEDDING_PROVIDER``.
    """
    try:
        provider = get_embedding_provider()
        return await provider.embed(text)
    except Exception as exc:
        logger.error("Failed to generate embedding: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Tool input schemas
# ---------------------------------------------------------------------------

class VectorSearchInput(BaseModel):
    """Input for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")


class HybridSearchInput(BaseModel):
    """Input for hybrid search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, description="Maximum number of results")
    text_weight: float = Field(
        default=0.3, description="Weight for text similarity (0-1)"
    )


class DocumentInput(BaseModel):
    """Input for document retrieval."""
    document_id: str = Field(..., description="Document ID to retrieve")


class DocumentListInput(BaseModel):
    """Input for listing documents."""
    limit: int = Field(default=20, description="Maximum number of documents")
    offset: int = Field(default=0, description="Number of documents to skip")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def vector_search_tool(input_data: VectorSearchInput) -> List[ChunkResult]:
    """Perform vector similarity search."""
    try:
        embedding = await generate_embedding(input_data.query)
        results = await vector_search(embedding=embedding, limit=input_data.limit)

        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["similarity"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]
    except Exception as exc:
        logger.error("Vector search failed: %s", exc)
        return []


async def hybrid_search_tool(input_data: HybridSearchInput) -> List[ChunkResult]:
    """Perform hybrid (vector + keyword) search."""
    try:
        embedding = await generate_embedding(input_data.query)
        results = await hybrid_search(
            embedding=embedding,
            query_text=input_data.query,
            limit=input_data.limit,
            text_weight=input_data.text_weight,
        )

        return [
            ChunkResult(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                content=r["content"],
                score=r["combined_score"],
                metadata=r["metadata"],
                document_title=r["document_title"],
                document_source=r["document_source"],
            )
            for r in results
        ]
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        return []


async def get_document_tool(
    input_data: DocumentInput,
) -> Optional[Dict[str, Any]]:
    """Retrieve a complete document with its chunks."""
    try:
        document = await get_document(input_data.document_id)
        if document:
            chunks = await get_document_chunks(input_data.document_id)
            document["chunks"] = chunks
        return document
    except Exception as exc:
        logger.error("Document retrieval failed: %s", exc)
        return None


async def list_documents_tool(
    input_data: DocumentListInput,
) -> List[DocumentMetadata]:
    """List available documents with metadata."""
    try:
        documents = await list_documents(
            limit=input_data.limit, offset=input_data.offset
        )
        return [
            DocumentMetadata(
                id=d["id"],
                title=d["title"],
                source=d["source"],
                metadata=d["metadata"],
                created_at=datetime.fromisoformat(d["created_at"]),
                updated_at=datetime.fromisoformat(d["updated_at"]),
                chunk_count=d.get("chunk_count"),
            )
            for d in documents
        ]
    except Exception as exc:
        logger.error("Document listing failed: %s", exc)
        return []

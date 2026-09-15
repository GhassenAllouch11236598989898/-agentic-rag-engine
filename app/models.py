"""Pydantic v2 request/response models for the Agentic RAG Engine API."""

from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict, field_validator
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SearchType(str, Enum):
    """Supported search strategies."""
    VECTOR = "vector"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# Chunk / Search result models
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    """Single chunk returned by a search query."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document_title: str
    document_source: str

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Clamp score to [0, 1]."""
        return max(0.0, min(1.0, v))


class SearchResponse(BaseModel):
    """Envelope for search results."""
    results: List[ChunkResult] = Field(default_factory=list)
    total_results: int = 0
    search_type: SearchType
    query_time_ms: float


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat message."""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    user_id: Optional[str] = Field(None, description="User identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    model_config = ConfigDict(use_enum_values=True)


class SearchRequest(BaseModel):
    """Standalone search request."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")
    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Lightweight document descriptor."""
    id: str
    title: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class ToolCall(BaseModel):
    """Record of a tool invocation made by the agent."""
    tool_name: str
    args: Dict[str, Any] = Field(default_factory=dict)
    tool_call_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Agent response envelope."""
    message: str
    session_id: str
    sources: List[DocumentMetadata] = Field(default_factory=list)
    tools_used: List[ToolCall] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Optional["EvaluationResult"] = Field(None, description="RAG evaluation metrics")


# ---------------------------------------------------------------------------
# RAG Evaluation models (RAG Triad + Semantic Alignment)
# ---------------------------------------------------------------------------

class ClaimVerification(BaseModel):
    """Verification result for a single atomic claim extracted from the response."""
    claim: str = Field(..., description="The extracted atomic claim")
    status: Literal["supported", "partially_supported", "unsupported"] = Field(
        ..., description="Whether the claim is grounded in the retrieved context"
    )
    supporting_chunk_index: Optional[int] = Field(
        None, description="Index of the context chunk that supports this claim"
    )
    reasoning: str = Field("", description="Brief justification for the status")


class RAGMetrics(BaseModel):
    """Individual metric scores for RAG Triad evaluation."""
    faithfulness: float = Field(..., ge=0.0, le=1.0, description="Groundedness of the answer in retrieved context (0–1)")
    answer_relevance: float = Field(..., ge=0.0, le=1.0, description="How directly the answer addresses the query (0–1)")
    context_relevance: float = Field(..., ge=0.0, le=1.0, description="Signal-to-noise ratio of retrieved chunks vs query (0–1)")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity between response and context embeddings (0–1)")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a RAG response."""
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Calibrated overall confidence (0–100)")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Hallucination risk tier")
    metrics: RAGMetrics = Field(..., description="Individual RAG Triad metric scores")
    claims: List[ClaimVerification] = Field(default_factory=list, description="Claim-by-claim verification")
    critique: str = Field("", description="Evaluator reasoning summary")
    evaluated_at: datetime = Field(default_factory=datetime.now)
    evaluation_time_ms: float = Field(0.0, description="Time taken for evaluation in ms")


class EvaluationRequest(BaseModel):
    """Request body for the standalone /evaluate endpoint."""
    query: str = Field(..., description="The original user query")
    response: str = Field(..., description="The generated response to evaluate")
    contexts: List[str] = Field(default_factory=list, description="Retrieved context chunks")
    session_id: Optional[str] = Field(None, description="Associated session ID")


# Update forward reference for ChatResponse
ChatResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Ingestion models
# ---------------------------------------------------------------------------

class IngestionConfig(BaseModel):
    """Configuration for the document ingestion pipeline."""
    chunk_size: int = Field(default=850, ge=100, le=5000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)
    max_chunk_size: int = Field(default=2000, ge=500, le=10000)
    use_semantic_chunking: bool = True

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Overlap must be strictly less than chunk_size."""
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"Chunk overlap ({v}) must be less than chunk size ({chunk_size})"
            )
        return v


class IngestionResult(BaseModel):
    """Summary of a single document ingestion."""
    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Error / health models
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standardised error payload."""
    error: str
    error_type: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None


class HealthStatus(BaseModel):
    """System health check response."""
    status: Literal["healthy", "unhealthy"]
    database: bool
    llm_provider: str
    embedding_provider: str
    version: str
    timestamp: datetime

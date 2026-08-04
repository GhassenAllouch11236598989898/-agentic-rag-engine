"""
Document chunking — semantic and recursive strategies.

Uses LangChain text splitters under the hood. Semantic chunking
uses the configured embedding provider for split-point detection.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChunkingConfig:
    """Configuration for the document chunking pipeline."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    use_semantic_splitting: bool = True

    def __post_init__(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


# ---------------------------------------------------------------------------
# Chunk data class
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """Represents a single document chunk with optional embedding."""
    content: str
    index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    token_count: Optional[int] = None

    def __post_init__(self):
        if self.token_count is None:
            self.token_count = len(self.content) // 4


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

class PDFSemanticChunker:
    """Chunk PDF content using semantic and/or recursive strategies."""

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self._semantic_splitter = None

        # Always available recursive fallback
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
        )

        # Lazy-init semantic splitter (requires embeddings)
        if config.use_semantic_splitting:
            self._init_semantic_splitter()

    def _init_semantic_splitter(self):
        """Initialise the semantic chunker with the configured embeddings."""
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            # Use the configured embedding provider via LangChain wrapper
            embedding_provider = self._get_langchain_embeddings()
            self._semantic_splitter = SemanticChunker(
                embeddings=embedding_provider,
                breakpoint_threshold_type="percentile",
            )
        except Exception as exc:
            logger.warning(
                "Semantic chunker init failed, will use recursive fallback: %s",
                exc,
            )
            self._semantic_splitter = None

    @staticmethod
    def _get_langchain_embeddings():
        """
        Build a LangChain-compatible embeddings instance.

        Prefers local SentenceTransformer; falls back to OpenAI if configured.
        """
        import os

        provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            )

        # Local SentenceTransformer via HuggingFace embeddings
        from langchain_community.embeddings import HuggingFaceEmbeddings

        model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        return HuggingFaceEmbeddings(model_name=model_name)

    def chunk_content(
        self,
        content: str,
        title: str = "PDF Document",
        source: str = "pdf",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Split *content* into chunks and return ``DocumentChunk`` objects.
        """
        if not content.strip():
            return []

        base_metadata = {
            "title": title,
            "source": source,
            "content_type": "pdf",
            **(metadata or {}),
        }

        doc = Document(page_content=content, metadata=base_metadata)

        try:
            if (
                self._semantic_splitter is not None
                and len(content) > self.config.chunk_size
            ):
                chunks = self._semantic_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "semantic"
            else:
                chunks = self.fallback_splitter.split_documents([doc])
                for chunk in chunks:
                    chunk.metadata["chunk_method"] = "recursive"
        except Exception as exc:
            logger.warning("Semantic chunking failed, using fallback: %s", exc)
            chunks = self.fallback_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata["chunk_method"] = "fallback"

        # Filter tiny chunks and convert to DocumentChunk
        final_chunks: List[DocumentChunk] = []
        for i, chunk in enumerate(chunks):
            text = chunk.page_content.strip()
            if len(text) >= self.config.min_chunk_size:
                chunk.metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(text),
                    }
                )
                final_chunks.append(
                    DocumentChunk(
                        content=text,
                        index=i,
                        start_char=0,
                        end_char=len(text),
                        metadata=chunk.metadata,
                    )
                )

        return final_chunks


def create_chunker(config: ChunkingConfig) -> PDFSemanticChunker:
    """Factory for ``PDFSemanticChunker``."""
    return PDFSemanticChunker(config)

"""
LLM and Embedding provider configuration.

Supports dual-provider architecture:
  - Local inference via Ollama (default)
  - Cloud inference via OpenAI (fallback)

Provider selection is controlled by environment variables:
  - LLM_PROVIDER: "ollama" | "openai"
  - EMBEDDING_PROVIDER: "local" | "openai"
"""

import os
import logging
from typing import List, Optional
from functools import lru_cache

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
import openai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------

def get_llm_model(model_override: Optional[str] = None) -> OpenAIChatModel:
    """
    Build an LLM model instance based on the configured provider.

    For Ollama the OpenAI-compatible endpoint is used transparently.

    Args:
        model_override: Optional model name that takes precedence over env.

    Returns:
        A configured ``OpenAIChatModel`` ready for Pydantic AI.
    """
    provider_name = os.getenv("LLM_PROVIDER", "ollama").lower()

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when LLM_PROVIDER=openai"
            )
        model_name = model_override or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
        provider = OpenAIProvider(api_key=api_key)
        logger.info("LLM provider: OpenAI (%s)", model_name)
    else:
        # Ollama exposes an OpenAI-compatible API on /v1
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = model_override or os.getenv("LLM_MODEL", "llama3.1:8b")
        provider = OpenAIProvider(
            api_key="ollama",  # Ollama ignores the key but the field is required
            base_url=f"{base_url}/v1",
        )
        logger.info("LLM provider: Ollama (%s @ %s)", model_name, base_url)

    return OpenAIChatModel(model_name, provider=provider)


# ---------------------------------------------------------------------------
# Embedding Provider
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Unified embedding interface that works with both local and OpenAI."""

    def __init__(self):
        self._provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
        self._model_name = self._resolve_model_name()
        self._dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self._local_model = None
        self._openai_client = None

        if self._provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY must be set when EMBEDDING_PROVIDER=openai"
                )
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("Embedding provider: OpenAI (%s)", self._model_name)
        else:
            logger.info(
                "Embedding provider: local SentenceTransformer (%s)",
                self._model_name,
            )

    # -- public API ----------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name

    async def embed(self, text: str) -> List[float]:
        """Generate an embedding vector for a single text."""
        if self._provider == "openai":
            return await self._embed_openai(text)
        return self._embed_local(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        if self._provider == "openai":
            return await self._embed_openai_batch(texts)
        return self._embed_local_batch(texts)

    # -- private helpers -----------------------------------------------------

    def _resolve_model_name(self) -> str:
        if self._provider == "openai":
            return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def _get_local_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._local_model = SentenceTransformer(self._model_name)
        return self._local_model

    def _embed_local(self, text: str) -> List[float]:
        model = self._get_local_model()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _embed_local_batch(self, texts: List[str]) -> List[List[float]]:
        model = self._get_local_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [v.tolist() for v in vectors]

    async def _embed_openai(self, text: str) -> List[float]:
        response = await self._openai_client.embeddings.create(
            model=self._model_name,
            input=text,
        )
        return response.data[0].embedding

    async def _embed_openai_batch(self, texts: List[str]) -> List[List[float]]:
        response = await self._openai_client.embeddings.create(
            model=self._model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_embedding_provider: Optional[EmbeddingProvider] = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the global ``EmbeddingProvider`` singleton (lazy-init)."""
    global _embedding_provider
    if _embedding_provider is None:
        _embedding_provider = EmbeddingProvider()
    return _embedding_provider


def get_embedding_dim() -> int:
    """Return the configured embedding dimension."""
    return int(os.getenv("EMBEDDING_DIM", "384"))

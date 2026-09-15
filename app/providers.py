"""
LLM and Embedding provider configuration optimized for local execution.

Supports dual-provider architecture:
  - Local inference via Ollama (default, free, no API key required)
  - Local Embeddings via SentenceTransformers (default) or Ollama embeddings
  - Cloud inference via OpenAI (optional fallback)

Provider selection is controlled by environment variables:
  - LLM_PROVIDER: "ollama" (default) | "openai"
  - EMBEDDING_PROVIDER: "local" (default) | "ollama" | "openai"
"""

import os
import logging
import asyncio
from typing import List, Optional

from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Provider (Local Ollama by default)
# ---------------------------------------------------------------------------

def get_llm_model(model_override: Optional[str] = None) -> OpenAIChatModel:
    """
    Build an LLM model instance based on the configured provider.

    Defaults to Ollama (http://localhost:11434/v1).

    Args:
        model_override: Optional model name that takes precedence over env.

    Returns:
        A configured ``OpenAIChatModel`` ready for Pydantic AI.
    """
    provider_name = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            model_name = model_override or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
            provider = OpenAIProvider(api_key=api_key)
            logger.info("LLM provider: OpenAI (%s)", model_name)
            return OpenAIChatModel(model_name, provider=provider)
        else:
            logger.warning(
                "LLM_PROVIDER=openai specified but OPENAI_API_KEY is missing. "
                "Falling back to local Ollama provider."
            )

    # Local Ollama setup (default)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model_name = model_override or os.getenv("LLM_MODEL", "llama3.1:8b")

    # Ollama exposes an OpenAI-compatible API on /v1
    provider = OpenAIProvider(
        api_key="ollama",  # Dummy key required by OpenAI client
        base_url=f"{base_url}/v1",
    )
    logger.info("LLM provider: Local Ollama (%s @ %s)", model_name, base_url)
    return OpenAIChatModel(model_name, provider=provider)


# ---------------------------------------------------------------------------
# Embedding Provider (Local SentenceTransformers or Ollama)
# ---------------------------------------------------------------------------

class EmbeddingProvider:
    """Unified embedding interface optimized for local execution."""

    def __init__(self):
        self._provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        # Fallback check if OpenAI requested without key
        if self._provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if not api_key:
                logger.warning(
                    "EMBEDDING_PROVIDER=openai specified but OPENAI_API_KEY is missing. "
                    "Falling back to local SentenceTransformers."
                )
                self._provider = "local"

        self._model_name = self._resolve_model_name()
        self._dim = int(os.getenv("EMBEDDING_DIM", "384"))
        self._local_model = None
        self._openai_client = None

        if self._provider == "openai":
            import openai
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self._openai_client = openai.AsyncOpenAI(api_key=api_key)
            logger.info("Embedding provider: OpenAI (%s)", self._model_name)
        elif self._provider == "ollama":
            logger.info("Embedding provider: Local Ollama (%s @ %s)", self._model_name, self._ollama_base_url)
        else:
            logger.info("Embedding provider: Local SentenceTransformer (%s)", self._model_name)

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
        elif self._provider == "ollama":
            return await self._embed_ollama(text)
        else:
            return await asyncio.to_thread(self._embed_local, text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embedding vectors for a batch of texts."""
        if not texts:
            return []
        if self._provider == "openai":
            return await self._embed_openai_batch(texts)
        elif self._provider == "ollama":
            return await self._embed_ollama_batch(texts)
        else:
            return await asyncio.to_thread(self._embed_local_batch, texts)

    # -- private helpers -----------------------------------------------------

    def _resolve_model_name(self) -> str:
        if self._provider == "openai":
            return os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        elif self._provider == "ollama":
            return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        return os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    def _get_local_model(self):
        """Lazy-load the SentenceTransformer model on demand."""
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for local embeddings. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            logger.info("Loading local SentenceTransformer model '%s'...", self._model_name)
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

    async def _embed_ollama(self, text: str) -> List[float]:
        url = f"{self._ollama_base_url}/api/embeddings"
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"model": self._model_name, "prompt": text})
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding", [])

    async def _embed_ollama_batch(self, texts: List[str]) -> List[List[float]]:
        tasks = [self._embed_ollama(text) for text in texts]
        return await asyncio.gather(*tasks)

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


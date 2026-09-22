"""Connected workflow checks with deterministic model and retrieval substitutes."""

import pytest
from pydantic_ai.models.test import TestModel
from app.support import SupportService
from app.workspace import Workspace


@pytest.mark.asyncio
async def test_live_structured_reply_and_citation_gate(tmp_path, monkeypatch):
    import app.providers as providers

    workspace = Workspace(tmp_path / "live.sqlite3")
    workspace.initialize()
    service = SupportService(workspace, "live")

    async def retrieve(*args):
        return [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "title": "API policy",
                "content": "The API allows 120 requests per minute.",
                "page": 2,
                "score": 0.5,
            }
        ]

    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(
        providers,
        "get_llm_model",
        lambda: TestModel(
            custom_output_args={
                "message": "The API allows 120 requests per minute. [1]",
                "sufficient_evidence": True,
            }
        ),
    )
    result = await service.draft("alice", "What is the API rate limit?")
    assert result["status"] == "draft" and result["sources"][0]["page"] == 2
    monkeypatch.setattr(
        providers,
        "get_llm_model",
        lambda: TestModel(
            custom_output_args={
                "message": "Unsupported statement [99]",
                "sufficient_evidence": True,
            }
        ),
    )
    result = await service.draft("alice", "What is the API rate limit?")
    assert result["status"] == "needs_escalation" and "[99]" not in result["message"]


@pytest.mark.asyncio
async def test_live_judge_can_escalate_despite_retrieval(tmp_path, monkeypatch):
    import app.providers as providers

    workspace = Workspace(tmp_path / "live.sqlite3")
    workspace.initialize()
    service = SupportService(workspace, "live")

    async def retrieve(*args):
        return [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "title": "Unrelated",
                "content": "A login guide.",
                "page": None,
                "score": 0.4,
            }
        ]

    monkeypatch.setattr(service, "retrieve", retrieve)
    monkeypatch.setattr(
        providers,
        "get_llm_model",
        lambda: TestModel(
            custom_output_args={
                "message": "Cannot confirm.",
                "sufficient_evidence": False,
            }
        ),
    )
    assert (await service.draft("alice", "Can you confirm an unsupported policy?"))[
        "status"
    ] == "needs_escalation"


def test_missing_provider_credentials_fail_explicitly(monkeypatch):
    from app.providers import get_llm_model, EmbeddingProvider

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        get_llm_model()
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        EmbeddingProvider()


@pytest.mark.asyncio
async def test_embedding_dimensions_and_batch_order(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from app.providers import EmbeddingProvider

    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    monkeypatch.setenv("EMBEDDING_DIM", "384")
    provider = EmbeddingProvider()
    request = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[2.0] * 384),
                SimpleNamespace(index=0, embedding=[1.0] * 384),
            ]
        )
    )
    provider._openai_client = SimpleNamespace(
        embeddings=SimpleNamespace(create=request)
    )
    vectors = await provider.embed_batch(["first", "second"])
    assert [v[0] for v in vectors] == [1.0, 2.0]
    assert request.call_args.kwargs["dimensions"] == 384
    request.return_value = SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0] * 1536)]
    )
    with pytest.raises(ValueError, match="dimension 384"):
        await provider.embed("wrong dimension")

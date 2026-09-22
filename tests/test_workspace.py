import pytest
from app.workspace import Workspace
from app.security import RateLimiter, authenticate, parse_keys
from app.support import citation_check


@pytest.fixture
def workspace(tmp_path):
    db = Workspace(tmp_path / "workspace.sqlite3")
    db.initialize()
    return db


def test_recent_conversation_is_returned_chronologically(workspace):
    session = workspace.session("alice")
    for i in range(12):
        workspace.save_draft(
            "alice",
            session,
            f"question {i}",
            {"draft_id": str(i), "message": f"answer {i}"},
        )
    history = workspace.history("alice", session, limit=6)
    assert [m["content"] for m in history] == [
        "question 9",
        "answer 9",
        "question 10",
        "answer 10",
        "question 11",
        "answer 11",
    ]


def test_session_and_feedback_require_ownership(workspace):
    session = workspace.session("alice")
    workspace.save_draft(
        "alice", session, "hello", {"draft_id": "draft1", "message": "hello"}
    )
    with pytest.raises(LookupError):
        workspace.history("bob", session)
    assert not workspace.feedback("bob", "draft1", "helpful")
    assert workspace.activity("bob") == []


def test_upload_duplicate_delete_and_page_provenance(workspace):
    pages = [
        (
            3,
            "The platinum subscription includes unlimited projects and priority assistance for all members.",
        )
    ]
    original = workspace.ingest("plans.pdf", pages)
    duplicate = workspace.ingest("renamed.pdf", pages)
    assert (
        original["document_id"] == duplicate["document_id"] and duplicate["duplicate"]
    )
    matches = workspace.search("platinum subscription")
    assert matches[0]["page"] == 3
    assert matches[0]["document_id"] == original["document_id"]
    assert workspace.delete_document(original["document_id"])
    assert workspace.search("platinum subscription") == []


def test_invalid_references_are_not_marked_valid():
    assert not citation_check("A policy [9]", [{"citation": 1}])["references_valid"]
    assert not citation_check("A policy", [{"citation": 1}])["references_valid"]
    assert citation_check("A policy [1]", [{"citation": 1}])["references_valid"]


def test_throttling_and_key_validation():
    limiter = RateLimiter(limit=2)
    assert limiter.allow("one") and limiter.allow("one")
    assert not limiter.allow("one") and limiter.allow("two")
    with pytest.raises(ValueError):
        parse_keys('{"alice":{"key":"short","role":"admin"}}')
    keys = parse_keys('{"alice":{"key":"aaaaaaaaaaaaaaaaaaaaaaaa","role":"admin"}}')
    assert authenticate("Bearer aaaaaaaaaaaaaaaaaaaaaaaa", keys).name == "alice"
    assert authenticate("Bearer incorrect", keys) is None

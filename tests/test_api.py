import json
import pytest
from fastapi.testclient import TestClient
from app.server import create_app

KEYS = json.dumps(
    {
        "alice": {"key": "a" * 32, "role": "admin"},
        "bob": {"key": "b" * 32, "role": "agent"},
    }
)
ALICE = {"Authorization": "Bearer " + "a" * 32}
BOB = {"Authorization": "Bearer " + "b" * 32}


@pytest.fixture
def client(tmp_path):
    app = create_app(mode="demo", data_path=tmp_path / "demo.sqlite3", keys="{}")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_complete_reply_has_real_sources(client):
    response = client.post(
        "/api/chat",
        json={"message": "Will an annual upgrade credit the unused monthly payment?"},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["status"] == "draft" and result["citation_check"]["references_valid"]
    assert "unused" in result["message"].lower()
    for source in result["sources"]:
        document = client.get("/api/documents/" + source["document_id"]).json()
        assert source["content"] in document["content"]
    assert len(client.get("/api/activity").json()["drafts"]) == 1


def test_unknown_compliance_question_escalates(client):
    result = client.post(
        "/api/chat",
        json={
            "message": "Is Northstar HIPAA certified and will it sign a business associate agreement?"
        },
    ).json()
    assert result["status"] == "needs_escalation"
    assert "specialist" in result["message"]


def test_upload_is_searchable_and_duplicate_is_idempotent(client):
    files = {
        "file": (
            "policy.md",
            b"# Polar migration\nPolar migration supports encrypted archives up to 700 MB and requires an administrator approval.",
        )
    }
    uploaded = client.post("/api/documents", files=files)
    assert uploaded.status_code == 201, uploaded.text
    duplicate = client.post("/api/documents", files=files).json()
    assert duplicate["duplicate"]
    result = client.post(
        "/api/chat", json={"message": "What archive size does Polar migration support?"}
    ).json()
    assert any(
        source["document_id"] == uploaded.json()["document_id"]
        for source in result["sources"]
    )
    assert "700 MB" in result["message"]


def test_bad_uploads_and_input_are_rejected(client):
    assert (
        client.post(
            "/api/documents", files={"file": ("binary.exe", b"bad")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/documents", files={"file": ("scan.pdf", b"not a pdf")}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/documents",
            files={"file": ("damaged.pdf", b"%PDF-1.7\nmissing objects")},
        ).status_code
        == 422
    )
    assert (
        client.post("/api/documents", files={"file": ("empty.txt", b"")}).status_code
        == 422
    )
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert (
        client.post(
            "/api/documents", files={"file": ("big.txt", b"x" * (10 * 1024 * 1024 + 1))}
        ).status_code
        == 413
    )


def test_authentication_roles_and_session_isolation(tmp_path):
    with TestClient(create_app("demo", tmp_path / "auth.sqlite3", keys=KEYS)) as client:
        assert client.get("/api/documents").status_code == 401
        assert client.get("/api/documents", headers=BOB).status_code == 200
        assert (
            client.post(
                "/api/documents",
                headers=BOB,
                files={"file": ("x.txt", b"Example policy document text.")},
            ).status_code
            == 403
        )
        draft = client.post(
            "/api/chat", headers=ALICE, json={"message": "How do we cancel renewal?"}
        ).json()
        assert (
            client.get("/api/sessions/" + draft["session_id"], headers=BOB).status_code
            == 404
        )
        assert (
            client.post(
                "/api/chat",
                headers=BOB,
                json={"message": "Tell me more", "session_id": draft["session_id"]},
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/drafts/" + draft["draft_id"] + "/feedback",
                headers=BOB,
                json={"value": "helpful"},
            ).status_code
            == 404
        )


def test_stream_uses_actual_result_and_reports_failure(client, monkeypatch):
    response = client.post(
        "/api/chat/stream", json={"message": "How do we cancel renewal?"}
    )
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["progress", "result", "end"]
    assert events[1]["sources"] and events[1]["citation_check"]["references_valid"]

    async def fail(*args, **kwargs):
        raise RuntimeError("private database credential")

    monkeypatch.setattr(client.app.state.service, "retrieve", fail)
    response = client.post(
        "/api/chat/stream", json={"message": "How do we cancel renewal?"}
    )
    assert '"type": "error"' in response.text
    assert "private database credential" not in response.text
    response = client.post("/api/chat", json={"message": "How do we cancel renewal?"})
    assert response.status_code == 503
    assert "private database credential" not in response.text


def test_quality_checks_and_feedback(client):
    result = client.post("/api/benchmark/run").json()
    assert result["passed"] == result["total"] == 6, result
    assert client.get("/api/activity").json()["drafts"] == []
    draft = client.post(
        "/api/chat", json={"message": "How do we cancel renewal?"}
    ).json()
    assert (
        client.post(
            "/api/drafts/" + draft["draft_id"] + "/feedback", json={"value": "helpful"}
        ).status_code
        == 200
    )
    assert client.get("/api/activity").json()["drafts"][0]["feedback"] == "helpful"


def test_live_mode_requires_access_keys(tmp_path):
    with pytest.raises(ValueError, match="require RELAY_API_KEYS"):
        create_app("live", tmp_path / "live.sqlite3", keys="{}")


def test_account_balance_and_refund_execution_are_escalated(client):
    for question in [
        "What is the exact outstanding balance on invoice INV-2026-8841?",
        "Please issue me a refund now",
    ]:
        result = client.post("/api/chat", json={"message": question}).json()
        assert result["status"] == "needs_escalation"
        assert "billing-system lookup" in result["message"]
        assert result["sources"] == []


def test_deleted_samples_do_not_reappear_on_restart(tmp_path):
    path = tmp_path / "restart.sqlite3"
    with TestClient(create_app("demo", path, keys="{}")) as client:
        documents = client.get("/api/documents").json()["documents"]
        assert client.delete("/api/documents/" + documents[0]["id"]).status_code == 200
    with TestClient(create_app("demo", path, keys="{}")) as client:
        assert (
            len(client.get("/api/documents").json()["documents"]) == len(documents) - 1
        )

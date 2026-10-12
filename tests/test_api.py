import pytest
from fastapi.testclient import TestClient
from helpers import build
from qdrant_client.http.exceptions import ResponseHandlingException

from paper_rag.api.app import create_app
from paper_rag.llm.chat import FakeChat


@pytest.fixture
def service():
    return build()[0]


@pytest.fixture
def client(service):
    with TestClient(create_app(service), raise_server_exceptions=False) as c:
        yield c


def test_health_reports_chunk_count(client):
    body = client.get("/health").json()
    assert body == {"status": "ok", "indexed_chunks": 3}


def test_ask_returns_answer_sources_and_usage(client):
    r = client.post("/ask", json={"question": "how do coating pigments shift colour?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answered"] and body["sources"][0]["file_name"] == "coating.pdf"
    assert body["usage"]["cost_nok"] > 0 and body["usage"]["total_ms"] >= 0
    assert r.headers["X-Request-ID"]


def test_ask_passes_filters(client):
    r = client.post("/ask", json={"question": "network coating brdf", "own_paper": False})
    assert {s["file_name"] for s in r.json()["sources"]} == {"neural.pdf"}


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": "  "},
        {"question": "ok question", "top_k": 0},
        {"question": "ok question", "year_min": 2025, "year_max": 2020},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_api_key_is_required_when_configured(service):
    service.settings.api_key = "secret"
    with TestClient(create_app(service)) as c:
        q = {"question": "coating pigments"}
        assert c.post("/ask", json=q).status_code == 401
        assert c.post("/ask", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/ask", json=q, headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/health").status_code == 200  # health stays open for the platform probe


def test_vector_store_down_gives_503(service, monkeypatch):
    def boom(*a, **k):
        raise ResponseHandlingException(ConnectionError("qdrant is down"))

    monkeypatch.setattr(service.store, "count", boom)
    with TestClient(create_app(service), raise_server_exceptions=False) as c:
        r = c.get("/health")
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"] and "qdrant is down" not in r.text


def test_llm_failure_gives_502_without_leaking_details(service):
    import httpx
    import openai

    class Broken(FakeChat):
        def complete(self, system, user):
            raise openai.APIConnectionError(request=httpx.Request("POST", "https://x"))

    service.llm = Broken()
    with TestClient(create_app(service), raise_server_exceptions=False) as c:
        r = c.post("/ask", json={"question": "coating pigments"})
    assert r.status_code == 502 and r.json()["request_id"]


def test_unexpected_error_gives_generic_500(service, monkeypatch):
    monkeypatch.setattr(service, "ask", lambda *a, **k: 1 / 0)
    with TestClient(create_app(service), raise_server_exceptions=False) as c:
        r = c.post("/ask", json={"question": "coating pigments"})
    assert r.status_code == 500 and "division" not in r.text


def test_ingest_indexes_pdfs_and_second_call_skips(service, make_pdf, tmp_path):
    make_pdf(
        tmp_path / "new.pdf", [f"Fresh paper sentence number {i} about optics." for i in range(8)]
    )
    service.settings.pdf_dir = str(tmp_path)
    service.settings.manifest_path = str(tmp_path / "none.csv")
    with TestClient(create_app(service)) as c:
        first = c.post("/ingest").json()
        second = c.post("/ingest").json()
    assert first["indexed"] == ["new.pdf"] and first["chunks_indexed"] > 0
    assert second["skipped"] == ["new.pdf"] and second["embedding_tokens"] == 0

"""API integration tests for regulatory policy search and document endpoints."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_get_policy_documents_catalog() -> None:
    """GET /api/v1/policy/documents should return manifest catalog of reference documents."""
    response = client.get("/api/v1/policy/documents")
    assert response.status_code == 200
    data = response.json()

    assert "total_documents" in data
    assert data["total_documents"] == 4
    assert len(data["documents"]) == 4

    # Verify fields on first document
    first_doc = data["documents"][0]
    assert "id" in first_doc
    assert "title" in first_doc
    assert "issuing_body" in first_doc
    assert "source_url" in first_doc
    assert "publication_date" in first_doc


def test_post_policy_search_valid_query() -> None:
    """POST /api/v1/policy/search with matching query returns relevant passages and disclaimer."""
    payload = {
        "query": "elder financial exploitation sudden withdrawal of funds",
        "top_k": 3,
        "min_similarity": 0.08,
    }
    response = client.post("/api/v1/policy/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == payload["query"]
    assert data["has_results"] is True
    assert data["total_retrieved"] >= 1
    assert "disclaimer" in data
    assert "decision support only" in data["disclaimer"].lower()

    top_passage = data["passages"][0]
    assert "chunk_id" in top_passage
    assert "document_title" in top_passage
    assert "issuing_body" in top_passage
    assert "section" in top_passage
    assert "source_url" in top_passage
    assert top_passage["similarity_score"] > 0.08


def test_post_policy_search_issuing_body_filter() -> None:
    """POST /api/v1/policy/search with issuing_body filter restricts results."""
    payload = {
        "query": "money laundering red flags structuring and smurfing",
        "issuing_body": "FATF",
        "top_k": 5,
    }
    response = client.post("/api/v1/policy/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_results"] is True
    for p in data["passages"]:
        assert "FATF" in p["issuing_body"]


def test_post_policy_search_irrelevant_query_no_results() -> None:
    """POST /api/v1/policy/search with irrelevant query returns has_results=False and empty list."""
    payload = {
        "query": "quantum mechanical particle wave duality physics experiments",
        "top_k": 3,
        "min_similarity": 0.10,
    }
    response = client.post("/api/v1/policy/search", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_results"] is False
    assert data["total_retrieved"] == 0
    assert data["passages"] == []


def test_post_policy_search_validation_errors() -> None:
    """Query that is too short or top_k out of range should return 422 validation error."""
    # Query too short (< 2 chars)
    resp_short = client.post("/api/v1/policy/search", json={"query": "a"})
    assert resp_short.status_code == 422

    # top_k too large (> 10)
    resp_large_k = client.post("/api/v1/policy/search", json={"query": "valid query", "top_k": 50})
    assert resp_large_k.status_code == 422


def test_post_policy_reindex() -> None:
    """POST /api/v1/policy/reindex should trigger rebuild and return chunk count."""
    response = client.post("/api/v1/policy/reindex")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert data["indexed_chunks"] >= 10


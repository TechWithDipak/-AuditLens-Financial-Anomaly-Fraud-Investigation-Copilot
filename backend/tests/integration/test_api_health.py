"""Tests for Health and Root Endpoints."""

from fastapi.testclient import TestClient


def test_root_endpoint(client: TestClient):
    """Verify root endpoint returns basic application metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "AuditLens"
    assert data["status"] == "online"
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/api/v1/health"


def test_health_endpoint(client: TestClient):
    """Verify health endpoint returns valid status and service metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "AuditLens"
    assert data["version"] == "1.0.0"
    assert data["environment"] == "development"
    assert "timestamp" in data
    assert "python_version" in data
    assert data["services"]["api"] == "healthy"
    assert data["services"]["database"] == "sqlite_ready"
    assert data["services"]["ml_engine"] == "ready"

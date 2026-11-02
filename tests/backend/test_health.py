"""Tests for backend health check endpoints."""

from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app

client = TestClient(app)


def test_get_health_root():
    """Verify GET /health returns 200 and expected health metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.PROJECT_NAME
    assert data["version"] == settings.VERSION
    assert data["environment"] == settings.ENVIRONMENT
    assert "timestamp" in data


def test_get_health_api_v1():
    """Verify GET /api/v1/health returns 200 and expected health metadata."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == settings.PROJECT_NAME


def test_get_root_welcome():
    """Verify GET / returns 200 and helpful navigation links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "welcome" in data["message"].lower()
    assert data["version"] == settings.VERSION
    assert data["health_check"] == "/health"

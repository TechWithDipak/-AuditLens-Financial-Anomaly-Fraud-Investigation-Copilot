"""Tests for consistent API error responses."""

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.app.core.errors import APIError, register_error_handlers


def create_test_app() -> FastAPI:
    """Create isolated test application to verify error handlers."""
    test_app = FastAPI()
    register_error_handlers(test_app)

    @test_app.get("/trigger-api-error")
    async def trigger_api_error():
        raise APIError(
            message="Invalid transaction parameter",
            code="INVALID_PARAMETER",
            status_code=400,
            details={"field": "amount", "issue": "must be positive"},
        )

    @test_app.get("/trigger-http-404")
    async def trigger_http_404():
        raise HTTPException(status_code=404, detail="Case record not found")

    @test_app.get("/trigger-unexpected")
    async def trigger_unexpected():
        raise RuntimeError("Simulated internal database connection drop")

    return test_app


test_client = TestClient(create_test_app(), raise_server_exceptions=False)


def test_api_error_format():
    """Verify custom APIError conforms to standardized JSON envelope."""
    response = test_client.get("/trigger-api-error")
    assert response.status_code == 400
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_PARAMETER"
    assert data["error"]["message"] == "Invalid transaction parameter"
    assert data["error"]["details"]["field"] == "amount"


def test_http_404_format():
    """Verify HTTPException maps to standard error structure."""
    response = test_client.get("/trigger-http-404")
    assert response.status_code == 404
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "NOT_FOUND"
    assert data["error"]["message"] == "Case record not found"


def test_unexpected_error_does_not_leak_internals():
    """Verify 500 errors hide stack traces and internal exception messages."""
    response = test_client.get("/trigger-unexpected")
    assert response.status_code == 500
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INTERNAL_SERVER_ERROR"
    # Ensure internal exception string is NOT leaked to user
    assert "Simulated internal database" not in data["error"]["message"]

"""Integration tests for Dataset API endpoints."""

import io

from fastapi.testclient import TestClient


def test_get_sample_dataset_endpoint(client: TestClient):
    """Verify sample dataset endpoint loads and profiles credit card fixture."""
    response = client.get("/api/v1/datasets/sample")
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "creditcard_sample_fixture.csv"
    assert data["quality"]["row_count"] == 1000
    assert data["quality"]["column_count"] == 31
    assert data["quality"]["class_imbalance"] is not None
    assert data["quality"]["class_imbalance"]["positive_class_ratio"] == 0.005
    assert len(data["preview_rows"]) == 10


def test_get_sequence_sample_endpoint(client: TestClient):
    """Verify sequential sample endpoint loads and profiles sequence account fixture."""
    response = client.get("/api/v1/datasets/sequence-sample")
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "sequence_account_fixture.csv"
    assert data["quality"]["row_count"] == 500
    assert data["quality"]["column_count"] == 7
    assert len(data["preview_rows"]) == 10


def test_validate_csv_upload_valid(client: TestClient):
    """Verify multipart CSV upload returns correct profile."""
    csv_content = (
        "Time,V1,V2,Amount,Class\n0.0,0.5,-1.2,100.0,0\n1.0,-0.8,0.9,45.5,0\n2.0,-2.1,3.4,950.0,1\n"
    )
    files = {"file": ("test_upload.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/datasets/validate-upload", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_upload.csv"
    assert data["quality"]["row_count"] == 3
    assert data["quality"]["column_count"] == 5


def test_validate_csv_upload_invalid_extension(client: TestClient):
    """Verify non-csv uploads return HTTP 400."""
    files = {"file": ("malicious.exe", io.BytesIO(b"binary content"), "application/octet-stream")}
    response = client.post("/api/v1/datasets/validate-upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files" in response.json()["detail"]

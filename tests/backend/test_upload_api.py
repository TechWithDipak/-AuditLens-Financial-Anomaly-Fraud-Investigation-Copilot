"""Tests for CSV dataset upload and validation API endpoint."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)
SAMPLE_DIR = Path("./data/sample")


def test_upload_valid_csv():
    """Verify uploading a valid CSV returns 200 and schema summary."""
    valid_csv_path = SAMPLE_DIR / "synthetic_creditcard_sample.csv"
    with open(valid_csv_path, "rb") as f:
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("synthetic_creditcard_sample.csv", f, "text/csv")},
            data={"is_training": "true"},
        )
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["data"]["row_count"] == 150
    assert res["data"]["column_count"] == 31
    assert "class_distribution" in res["data"]["summary"]


def test_upload_unsupported_file_extension():
    """Verify non-CSV file extension is rejected with 415."""
    response = client.post(
        "/api/v1/data/upload",
        files={
            "file": (
                "malicious_payload.exe",
                b"binary content",
                "application/octet-stream",
            )
        },
    )
    assert response.status_code == 415
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_empty_csv():
    """Verify empty CSV upload is rejected with 422."""
    response = client.post(
        "/api/v1/data/upload",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 422
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_DATASET"


def test_upload_missing_columns_csv():
    """Verify CSV with missing columns is rejected with 422 and lists missing columns."""
    missing_csv_path = SAMPLE_DIR / "corrupted_missing_columns.csv"
    with open(missing_csv_path, "rb") as f:
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("corrupted_missing_columns.csv", f, "text/csv")},
        )
    assert response.status_code == 422
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_DATASET"
    assert len(res["error"]["details"]["missing_columns"]) > 0


def test_upload_corrupted_types_csv():
    """Verify CSV with corrupted strings in numeric columns is rejected with 422."""
    corrupted_types_path = SAMPLE_DIR / "corrupted_types.csv"
    with open(corrupted_types_path, "rb") as f:
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("corrupted_types.csv", f, "text/csv")},
        )
    assert response.status_code == 422
    res = response.json()
    assert res["success"] is False
    assert res["error"]["code"] == "INVALID_DATASET"
    assert any("non-numeric" in err for err in res["error"]["details"]["errors"])


def test_upload_valid_csv_with_database_ingestion(client: TestClient):
    """Verify uploading a CSV with ingest_to_db=true validates, scores, and saves to database."""
    valid_csv_path = SAMPLE_DIR / "synthetic_creditcard_sample.csv"
    with open(valid_csv_path, "rb") as f:
        response = client.post(
            "/api/v1/data/upload",
            files={"file": ("synthetic_creditcard_sample.csv", f, "text/csv")},
            data={"is_training": "true", "ingest_to_db": "true"},
            headers={"X-Analyst-ID": "analyst_uploader"},
        )
    assert response.status_code == 200
    res = response.json()
    assert res["success"] is True
    assert res["data"]["ingested_count"] == 150

    # Verify transactions are queryable through the transactions endpoint
    tx_list = client.get("/api/v1/transactions")
    assert tx_list.status_code == 200
    assert tx_list.json()["total"] == 150

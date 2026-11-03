"""Tests for Transaction endpoints, scoring, and explainability."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.db.models import TransactionModel
from ml.data.schema import REQUIRED_FEATURE_COLUMNS


def populate_sample_transactions(db: Session, count: int = 5) -> list[TransactionModel]:
    """Helper to populate database with sample transaction records."""
    dummy_features = {col: 0.05 * i for i, col in enumerate(REQUIRED_FEATURE_COLUMNS)}
    import json

    feat_json = json.dumps(dummy_features)

    txs = []
    for i in range(1, count + 1):
        tx = TransactionModel(
            transaction_id=f"TX_TEST_{i:04d}",
            time=float(i * 100),
            amount=float(i * 50.0),
            features=feat_json,
            anomaly_score=0.1 * i,
            fraud_probability=0.15 * i,
            is_flagged=(i >= 3),
            score_type="dual_scoring",
            label=(1 if i >= 4 else 0),
        )
        txs.append(tx)

    db.add_all(txs)
    db.commit()
    return txs


def test_list_transactions_empty(client: TestClient):
    """Verify empty database returns clean paginated response."""
    response = client.get("/api/v1/transactions")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["items"] == []
    assert data["page"] == 1


def test_list_transactions_pagination_and_filters(client: TestClient, test_db: Session):
    """Verify transaction pagination and amount/flagged filters."""
    populate_sample_transactions(test_db, count=6)

    # Page 1 with page_size 2
    res1 = client.get("/api/v1/transactions?page=1&page_size=2")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total"] == 6
    assert len(data1["items"]) == 2
    assert data1["total_pages"] == 3

    # Filter by is_flagged=true (should match transactions 3, 4, 5, 6 -> 4 items)
    res_flagged = client.get("/api/v1/transactions?is_flagged=true")
    assert res_flagged.status_code == 200
    data_flagged = res_flagged.json()
    assert data_flagged["total"] == 4
    for item in data_flagged["items"]:
        assert item["is_flagged"] is True

    # Filter by amount range
    res_amt = client.get("/api/v1/transactions?min_amount=100.0&max_amount=200.0")
    assert res_amt.status_code == 200
    data_amt = res_amt.json()
    assert data_amt["total"] == 3  # 100, 150, 200


def test_get_single_transaction_and_404(client: TestClient, test_db: Session):
    """Verify retrieving transaction by ID and handling non-existent ID."""
    txs = populate_sample_transactions(test_db, count=2)
    tx = txs[0]

    # Fetch by string transaction_id
    res = client.get(f"/api/v1/transactions/{tx.transaction_id}")
    assert res.status_code == 200
    item = res.json()
    assert item["transaction_id"] == tx.transaction_id
    assert item["amount"] == 50.0

    # Fetch non-existent
    res_404 = client.get("/api/v1/transactions/TX_NONEXISTENT_999")
    assert res_404.status_code == 404
    assert "not found" in res_404.json()["error"]["message"].lower()


def test_score_transaction_realtime(client: TestClient):
    """Verify ad-hoc transaction scoring endpoint."""
    features = {col: 0.1 for col in REQUIRED_FEATURE_COLUMNS}
    payload = {
        "transaction_id": "ADHOC_TEST_01",
        "time": 450.0,
        "amount": 1250.0,
        "features": features,
    }

    response = client.post("/api/v1/transactions/score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "ADHOC_TEST_01"
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert isinstance(data["is_flagged"], bool)
    assert "decision_threshold" in data


def test_score_transaction_negative_amount_rejected(client: TestClient):
    """Verify negative monetary amount is rejected by schema validator."""
    features = {col: 0.1 for col in REQUIRED_FEATURE_COLUMNS}
    payload = {
        "transaction_id": "INVALID_01",
        "time": 10.0,
        "amount": -50.0,  # Negative
        "features": features,
    }
    response = client.post("/api/v1/transactions/score", json=payload)
    assert response.status_code == 422


def test_transaction_explanation_endpoint(client: TestClient, test_db: Session):
    """Verify explanation endpoint returns top contributions and disclaimers."""
    txs = populate_sample_transactions(test_db, count=1)
    tx = txs[0]

    response = client.get(f"/api/v1/transactions/{tx.transaction_id}/explanation")
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == tx.transaction_id
    assert len(data["top_contributions"]) > 0
    assert len(data["limitations"]) > 0
    assert "proof" not in data["summary_text"].lower()

    # Test 404 for unknown transaction explanation
    res_404 = client.get("/api/v1/transactions/UNKNOWN_TX/explanation")
    assert res_404.status_code == 404


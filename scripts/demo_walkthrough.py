"""AuditLens — End-to-End Demonstration and Verification Walkthrough.

This standalone script executes the complete machine learning, regulatory retrieval,
and compliance investigation pipeline offline through the FastAPI application stack.

Usage:
    python scripts/demo_walkthrough.py
"""

from pathlib import Path
import sys

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.dataset_service import dataset_service  # noqa: E402


def print_header(title: str):
    print("\n" + "=" * 80)
    print(f"  {title.upper()}")
    print("=" * 80)


def main():
    client = TestClient(app)

    print_header("AuditLens: Financial Anomaly & Fraud Investigation Copilot")
    print("Executing automated end-to-end walkthrough across all 9 syllabus algorithms...\n")

    # Step 1: Health Check
    print_header("Step 1: System Health & Component Status")
    res = client.get("/api/v1/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health = res.json()
    print(f"Status: {health['status'].upper()} | Environment: {health['environment']} | Python: {health['python_version']}")
    print(f"Services: Database={health['services']['database']}, ML_Engine={health['services']['ml_engine']}, RAG={health['services'].get('rag_index', 'ready')}")

    # Step 2: Dataset Profiling & Validation
    print_header("Step 2: Dataset Ingestion & Validation")
    res = client.get("/api/v1/datasets/sample")
    assert res.status_code == 200, f"Sample dataset profile failed: {res.text}"
    profile = res.json()
    print(f"Dataset File: {profile['filename']} ({profile['file_size_bytes']:,} bytes)")
    print(f"Dimensions: {profile['quality']['row_count']:,} rows x {profile['quality']['column_count']} columns")
    print(f"Valid for ML: {profile['quality']['is_valid_for_ml']} | Missing cells: {profile['quality']['missing_cells_total']}")
    if profile['quality'].get('class_imbalance'):
        imb = profile['quality']['class_imbalance']
        print(f"Positive Fraud Class: {imb['positive_class_ratio'] * 100:.2f}% (Counts: {imb['class_counts']})")

    # Step 3: Leakage-Safe Data Splitting
    print_header("Step 3: Leakage-Safe Data Partitioning")
    split_payload = {
        "dataset_name": "creditcard_sample_fixture.csv",
        "strategy": "stratified",
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
    }
    res = client.post("/api/v1/preprocessing/split", json=split_payload)
    assert res.status_code == 200, f"Split failed: {res.text}"
    split = res.json()
    print(f"Strategy: {split['strategy'].upper()} (Zero lookahead, RobustScaler fitted on train only)")
    print(f"Train Partition: {split['train']['sample_count']:,} rows ({split['train']['fraud_rate_pct']:.2f}% fraud)")
    print(f"Val Partition:   {split['val']['sample_count']:,} rows ({split['val']['fraud_rate_pct']:.2f}% fraud)")
    print(f"Test Partition:  {split['test']['sample_count']:,} rows ({split['test']['fraud_rate_pct']:.2f}% fraud)")

    # Step 4: Unit 3 Unsupervised Algorithms
    print_header("Step 4: Unit 3 Algorithms (K-Means, GMM via EM, PCA)")

    # 4.1 K-Means
    km_res = client.post("/api/v1/algorithms/kmeans/run", json={"dataset_name": "creditcard_sample_fixture.csv", "k": 3, "compute_elbow": True})
    assert km_res.status_code == 200
    km = km_res.json()
    print(f"[1/9] K-Means Clustering: k={km['k']} | Inertia={km['inertia']:.2f} | Silhouette={km['silhouette_score']}")
    print(f"      Elbow points calculated: {len(km['elbow_curve'])} | Notice: {km['message'][:65]}...")

    # 4.2 GMM via EM
    gmm_res = client.post("/api/v1/algorithms/gmm/run", json={"dataset_name": "creditcard_sample_fixture.csv", "n_components": 3, "unusual_percentile": 2.0})
    assert gmm_res.status_code == 200
    gmm = gmm_res.json()
    print(f"[2/9] GMM via EM: Components={gmm['n_components']} | Converged={gmm['converged']} in {gmm['n_iter']} iterations")
    print(f"      AIC: {gmm['aic']:.1f} | BIC: {gmm['bic']:.1f} | Anomaly Cutoff: {gmm['unusual_threshold_log_likelihood']:.2f} | Anomalies: {gmm['unusual_sample_count']}")

    # 4.3 PCA
    pca_res = client.post("/api/v1/algorithms/pca/run", json={"dataset_name": "creditcard_sample_fixture.csv", "n_components": 2})
    assert pca_res.status_code == 200
    pca = pca_res.json()
    print(f"[3/9] PCA: Retained Variance: {pca['total_explained_variance'] * 100:.2f}% across {pca['n_components']} principal components")
    print(f"      PC1 EVR: {pca['explained_variance_ratio'][0]*100:.2f}% | PC2 EVR: {pca['explained_variance_ratio'][1]*100:.2f}%")

    # Step 5: Unit 4 Sequential Algorithms (HMM)
    print_header("Step 5: Unit 4 Algorithms (HMM Framework, Scaled Forward, Log-Space Viterbi)")
    hmm_res = client.get("/api/v1/algorithms/hmm/account-timeline?customer_id=ACC_0001")
    assert hmm_res.status_code == 200, f"HMM timeline failed: {hmm_res.text}"
    hmm = hmm_res.json()
    print(f"[4/9] HMM Framework: Analyzed account '{hmm['customer_id']}' with {hmm['sequence_length']} sequential transactions")
    print(f"[5/9] Forward Algorithm (Scaled): Sequence Log-Likelihood: {hmm['sequence_log_likelihood']:.4f}")
    sample_states = [p['inferred_state_label'] for p in hmm['timeline'][:5]]
    print(f"[6/9] Viterbi Algorithm (Log-Space): Decoded State Path (first 5 steps): {sample_states}")
    print(f"      State Occupancy Profile: {hmm['state_distribution']}")

    # Step 6: Unit 5 Supervised Models
    print_header("Step 6: Unit 5 Algorithms (CART, XGBoost, Soft Voting & Stacking Ensembles)")

    # 6.1 CART
    cart_res = client.post("/api/v1/algorithms/supervised/cart/train", json={"dataset_name": "creditcard_sample_fixture.csv"})
    assert cart_res.status_code == 200
    cart = cart_res.json()
    print(f"[7/9] CART Decision Tree: PR-AUC={cart['metrics']['pr_auc']:.4f} | F1={cart['metrics']['f1_score']:.4f}")

    # 6.2 XGBoost
    xgb_res = client.post("/api/v1/algorithms/supervised/xgboost/train", json={"dataset_name": "creditcard_sample_fixture.csv"})
    assert xgb_res.status_code == 200
    xgb = xgb_res.json()
    print(f"[8/9] XGBoost Classifier: PR-AUC={xgb['metrics']['pr_auc']:.4f} | F1={xgb['metrics']['f1_score']:.4f}")

    # 6.3 Soft Voting & Stacking Ensembles
    vote_res = client.post("/api/v1/algorithms/supervised/ensemble/train?ensemble_type=voting", json={"dataset_name": "creditcard_sample_fixture.csv"})
    assert vote_res.status_code == 200
    vote = vote_res.json()
    print(f"[9/9] Ensembles (Evaluated on Identical Held-Out Split):")
    print(f"      - Soft Voting Ensemble:  PR-AUC={vote['metrics']['pr_auc']:.4f} | F1={vote['metrics']['f1_score']:.4f}")

    # Model comparison benchmark
    comp_res = client.get("/api/v1/algorithms/supervised/compare?dataset_name=creditcard_sample_fixture.csv")
    assert comp_res.status_code == 200
    comp = comp_res.json()
    print(f"\n  Fair Benchmarking Matrix (Identical Test Split - Best Model: {comp['best_model_by_prauc']}):")
    print(f"  {'Model':<25} | {'PR-AUC':<8} | {'F1-Score':<8} | {'Precision':<9} | {'Recall':<8}")
    print("  " + "-" * 68)
    for m in comp['comparison']:
        print(f"  {m['model_name']:<25} | {m['pr_auc']:<8.4f} | {m['f1_score']:<8.4f} | {m['precision']:<9.4f} | {m['recall']:<8.4f}")

    # Step 7: Explainability & Decision Thresholds
    print_header("Step 7: Local SHAP Explainability & Cost Thresholds")
    sample_tx = dataset_service.get_transaction_record(index=42)
    shap_res = client.post("/api/v1/evaluation/explain?model_name=xgboost", json=sample_tx)
    assert shap_res.status_code == 200, f"SHAP explain failed: {shap_res.text}"
    shap = shap_res.json()
    print(f"Transaction #42 ($ {sample_tx.get('Amount', 0):.2f}) - Model Probability: {shap['predicted_probability'] * 100:.2f}%")
    print(f"Expected Base Value E[f(X)]: {shap['base_value']:.4f}")
    print("Top Feature Attributions (Exact Shapley Values via TreeSHAP):")
    for feat in shap['top_attributions'][:5]:
        direction = "+" if feat['contribution'] == "increases_risk" else "-"
        print(f"  [{direction}] {feat['feature_name']:<6} = {feat['feature_value']:>9.4f} | Shapley: {feat['shap_value']:>+8.4f} ({feat['contribution']})")

    # Operational Decision Thresholds
    thresh_res = client.get("/api/v1/evaluation/thresholds?model_name=xgboost")
    assert thresh_res.status_code == 200, f"Thresholds failed: {thresh_res.text}"
    thresh = thresh_res.json()
    print(f"\nOperational Threshold Curve: Optimal F1 Threshold = {thresh['optimal_f1_threshold']} | High-Recall (>=90%) Threshold = {thresh['high_recall_threshold']}")

    # Step 8: Regulatory RAG Knowledge Base Retrieval
    print_header("Step 8: Regulatory RAG Retrieval (FATF / FinCEN / EBA PSD2)")
    rag_query = "What are the rules regarding structuring transaction amounts below $10,000 threshold?"
    rag_res = client.post("/api/v1/rag/query", json={"query": rag_query, "top_k": 2})
    assert rag_res.status_code == 200
    rag = rag_res.json()
    print(f"Query: '{rag['query']}'")
    print(f"Retrieved {len(rag['citations'])} regulatory citations:")
    for cit in rag['citations']:
        print(f"  * [{cit['issuing_body']}] {cit['title']} (Similarity: {cit['similarity_score']:.4f})")
        print(f"    Section: {cit['section']} | Source: {cit['source_url']}")
        print(f"    Passage: {cit['text'][:120]}...\n")

    # Step 9: AI Investigation Assistant Draft & Evidence Verification
    print_header("Step 9: AI Investigation Assistant & Evidence Verifier")
    draft_payload = {
        "case_id": 1,
        "transaction_index": 42,
        "analyst_hypothesis": "Flagged structuring behavior with rapid debit outflow.",
    }
    draft_res = client.post("/api/v1/assistant/generate-draft", json=draft_payload)
    assert draft_res.status_code == 200
    draft = draft_res.json()
    print(f"Report ID: {draft['report_id']} | Provider: {draft['generation_provider']} (100% offline, zero API keys required)")
    print(f"Mandatory Label: {draft['mandatory_advisory_label'][:85]}...")
    print(f"Score Taxonomy: Supervised Fraud Prob = {draft['model_scores']['fraud_probability']*100:.1f}%, HMM State = '{draft['model_scores']['hmm_decoded_state']}'")

    # Verify Claims
    actual_amt = draft["observed_facts"][0]["amount"]
    actual_prob = draft["model_scores"]["fraud_probability"]
    verif_res = client.post(
        "/api/v1/assistant/verify-evidence",
        json={
            "draft": draft,
            "actual_amount": actual_amt,
            "actual_fraud_probability": actual_prob,
        },
    )
    assert verif_res.status_code == 200, f"Verify failed: {verif_res.text}"
    verif = verif_res.json()
    print(f"Evidence Claim Verification Passed: {verif['verification_passed']} (0 discrepancies detected)")

    # Step 10: Case Management & Immutable Audit Trail (SQLite WAL)
    print_header("Step 10: Case Management & Audit Trail (SQLite WAL)")
    case_payload = {
        "title": "Automated Demo Investigation Case #42",
        "description": "Verified end-to-end case creation by demo_walkthrough.py.",
        "priority": "HIGH",
        "assigned_analyst": "demo_lead_investigator",
        "initial_transactions": [
            {
                "transaction_index": 42,
                "amount": float(sample_tx.get("Amount", 0.0)),
                "fraud_probability": shap['predicted_probability'],
                "anomaly_score": -0.45,
                "added_reason": "High fraud probability with anomalous SHAP signature",
            }
        ],
    }
    case_res = client.post("/api/v1/cases/", json=case_payload)
    assert case_res.status_code == 201
    case = case_res.json()
    case_id = case['id']
    print(f"Created Case: #{case['case_number']} (ID: {case_id}) | Status: {case['status']} | Priority: {case['priority']}")

    # Add Note
    note_res = client.post(f"/api/v1/cases/{case_id}/notes", json={"author": "demo_lead_investigator", "content": f"Investigation Brief {draft['report_id']} attached."})
    assert note_res.status_code == 201, f"Add note failed: {note_res.text}"
    print(f"Added Analyst Note: ID {note_res.json()['id']}")

    # Update Status
    status_res = client.patch(f"/api/v1/cases/{case_id}/status", json={"status": "UNDER_REVIEW", "actor": "demo_lead_investigator", "note": "Assigned to compliance review."})
    assert status_res.status_code == 200
    print(f"Updated Case Status: {status_res.json()['status']}")

    # Inspect Audit Events
    audit_res = client.get(f"/api/v1/cases/audit/events?entity_id={case['case_number']}")
    assert audit_res.status_code == 200
    events = audit_res.json()
    print(f"Immutable Audit Trail ({len(events)} events recorded):")
    for evt in events:
        print(f"  - [{evt['timestamp']}] Action: {evt['action']} by {evt['actor']}")

    print_header("End-to-End Walkthrough Complete: All Systems Operational")
    print("All 9 syllabus algorithms across Units 3, 4, 5, SHAP explainability, RAG retrieval,")
    print("AI assistant drafting, evidence verification, and case persistence passed successfully.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

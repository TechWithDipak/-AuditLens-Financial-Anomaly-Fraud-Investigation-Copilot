"""Full model training, evaluation, threshold tuning, and artifact generation for AuditLens."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_recall_curve, auc
from xgboost import XGBClassifier
import joblib

from src.config import (
    TRAIN_RAW_FILE,
    TEST_RAW_FILE,
    ARTIFACTS_DIR,
    RANDOM_SEED,
    DEFAULT_THRESHOLD_SUSPICIOUS,
)
from src.data import (
    check_raw_data_exists,
    load_raw_dataset,
    split_train_val,
    compute_split_stats,
    generate_data_docs,
)
from src.features import (
    compute_causal_customer_features,
    engineer_base_features,
    FeaturePipeline,
)
from src.em_gmm import HandWrittenGMM
from src.hmm import (
    encode_discrete_symbols,
    CustomerHMM,
    compute_customer_sequence_scores,
)
from src.models import (
    CustomerSegmentation,
    PCAModel,
    SoftVotingEnsemble,
    tune_decision_threshold,
    evaluate_model_metrics,
)


def train_pipeline() -> Dict[str, Any]:
    """Execute complete end-to-end training and evaluation pipeline."""
    start_time = time.time()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Checking and Ingesting Raw Data ===")
    exists, msg = check_raw_data_exists()
    if not exists:
        raise FileNotFoundError(msg)

    print("Loading raw CSV datasets...")
    df_train_raw = load_raw_dataset(TRAIN_RAW_FILE)
    df_test_raw = load_raw_dataset(TEST_RAW_FILE)

    n_train_full = len(df_train_raw)
    n_test = len(df_test_raw)
    n_train = int(n_train_full * 0.8)
    n_val = n_train_full - n_train

    print(f"Loaded: fraudTrain={n_train_full:,} rows, fraudTest={n_test:,} rows.")

    print("\n=== Step 2: Causal Feature Engineering across Full Chronological History ===")
    t_feat = time.time()
    # Concatenate chronologically to compute causal rolling customer histories
    df_all = pd.concat([df_train_raw, df_test_raw], ignore_index=True)
    df_all = compute_causal_customer_features(df_all)
    df_all = engineer_base_features(df_all)
    print(f"Causal features computed in {time.time() - t_feat:.2f}s.")

    # Chronological partition
    train_df = df_all.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df_all.iloc[n_train:n_train_full].copy().reset_index(drop=True)
    test_df = df_all.iloc[n_train_full:].copy().reset_index(drop=True)

    # Document verified splits in docs/DATA.md
    stats = {
        "train": compute_split_stats(train_df, "Train (80%)"),
        "val": compute_split_stats(val_df, "Validation (20%)"),
        "train_full": compute_split_stats(df_all.iloc[:n_train_full], "Full fraudTrain"),
        "test": compute_split_stats(test_df, "Test (fraudTest)"),
    }
    generate_data_docs(stats)
    print(f"Dataset splits verified: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}.")

    print("\n=== Step 3: Feature Preprocessor Pipeline (Fit on Train Only) ===")
    pipe = FeaturePipeline()
    pipe.fit(train_df)
    pipe.save(ARTIFACTS_DIR / "feature_pipeline.joblib")

    X_train_base = pipe.transform(train_df)
    X_val_base = pipe.transform(val_df)
    X_test_base = pipe.transform(test_df)
    base_feat_names = list(X_train_base.columns)
    print(f"Base feature matrix shape: {X_train_base.shape} ({len(base_feat_names)} features).")

    print("\n=== Step 4: Unsupervised Models (Fitted on Train Period Only) ===")
    # 4.1 PCA (5 components)
    print("Fitting PCA (5 components)...")
    pca = PCAModel(n_components=5, random_state=RANDOM_SEED)
    pca.fit(X_train_base.values)
    pca.save(ARTIFACTS_DIR / "pca_model.joblib")
    X_train_pca = pca.transform(X_train_base.values)
    X_val_pca = pca.transform(X_val_base.values)
    X_test_pca = pca.transform(X_test_base.values)

    # 4.2 Customer-level K-Means Segmentation
    print("Fitting Customer-level K-Means Segmentation...")
    kmeans = CustomerSegmentation(n_clusters=4, random_state=RANDOM_SEED)
    kmeans.fit(train_df)
    kmeans.save(ARTIFACTS_DIR / "kmeans_model.joblib")
    print(f"K-Means fitted on {len(kmeans.customer_profiles_)} cardholders. Silhouette: {kmeans.silhouette_:.4f}")

    train_clusters, train_dists = kmeans.get_customer_features(train_df["cc_num"].values)
    val_clusters, val_dists = kmeans.get_customer_features(val_df["cc_num"].values)
    test_clusters, test_dists = kmeans.get_customer_features(test_df["cc_num"].values)

    # 4.3 Hand-written EM-GMM on PCA components
    print("Fitting Hand-written EM-GMM on PCA components...")
    # Train EM-GMM on representative 50k subsample of train for speed and stable convergence
    rng = np.random.RandomState(RANDOM_SEED)
    gmm_sub_idx = rng.choice(len(X_train_pca), size=min(50000, len(X_train_pca)), replace=False)
    gmm = HandWrittenGMM(n_components=4, max_iter=30, reg_covar=1e-6, random_state=RANDOM_SEED)
    gmm.fit(X_train_pca[gmm_sub_idx])
    joblib.dump(gmm, ARTIFACTS_DIR / "gmm_model.joblib")

    gmm_train_score = gmm.score_samples(X_train_pca)
    gmm_val_score = gmm.score_samples(X_val_pca)
    gmm_test_score = gmm.score_samples(X_test_pca)

    # 4.4 HMM with Baum-Welch and Hand-written Forward/Viterbi
    print("Fitting Customer HMM on Discrete Observation Sequences...")
    train_symbols = encode_discrete_symbols(
        train_df["amt_zscore_vs_customer_history"].values,
        train_df["is_night"].values,
        train_df["distance_km"].values,
    )
    val_symbols = encode_discrete_symbols(
        val_df["amt_zscore_vs_customer_history"].values,
        val_df["is_night"].values,
        val_df["distance_km"].values,
    )
    test_symbols = encode_discrete_symbols(
        test_df["amt_zscore_vs_customer_history"].values,
        test_df["is_night"].values,
        test_df["distance_km"].values,
    )

    train_df["symbol"] = train_symbols
    val_df["symbol"] = val_symbols
    test_df["symbol"] = test_symbols

    # Train HMM on customer sequences from train split
    cust_seqs = [grp["symbol"].values for _, grp in train_df.groupby("cc_num")]
    # Sample 200 customer sequences for robust, fast Baum-Welch training
    sample_seqs = [cust_seqs[i] for i in rng.choice(len(cust_seqs), size=min(200, len(cust_seqs)), replace=False)]
    hmm = CustomerHMM(n_components=3, random_state=RANDOM_SEED)
    hmm.fit(sample_seqs)
    hmm.save(ARTIFACTS_DIR / "hmm_model.joblib")

    # Compute rolling sequence anomaly scores using scaled forward
    hmm_train_score = compute_customer_sequence_scores(train_df, hmm.startprob_, hmm.transmat_, hmm.emissionprob_)
    hmm_val_score = compute_customer_sequence_scores(val_df, hmm.startprob_, hmm.transmat_, hmm.emissionprob_)
    hmm_test_score = compute_customer_sequence_scores(test_df, hmm.startprob_, hmm.transmat_, hmm.emissionprob_)

    # Assemble augmented feature matrices
    unsupervised_cols = ["gmm_score", "hmm_score", "kmeans_cluster", "kmeans_distance"]
    X_train_aug = np.hstack([
        X_train_base.values,
        gmm_train_score[:, None],
        hmm_train_score[:, None],
        train_clusters[:, None],
        train_dists[:, None],
    ])
    X_val_aug = np.hstack([
        X_val_base.values,
        gmm_val_score[:, None],
        hmm_val_score[:, None],
        val_clusters[:, None],
        val_dists[:, None],
    ])
    X_test_aug = np.hstack([
        X_test_base.values,
        gmm_test_score[:, None],
        hmm_test_score[:, None],
        test_clusters[:, None],
        test_dists[:, None],
    ])
    augmented_feature_names = base_feat_names + unsupervised_cols

    y_train = train_df["is_fraud"].values.astype(int)
    y_val = val_df["is_fraud"].values.astype(int)
    y_test = test_df["is_fraud"].values.astype(int)

    print("\n=== Step 5: Supervised Models & Soft Voting Ensemble ===")
    # 5.1 CART Decision Tree (depth 4-5)
    print("Training CART Decision Tree (depth=4)...")
    cart_model = DecisionTreeClassifier(
        max_depth=4,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    cart_model.fit(X_train_aug, y_train)
    joblib.dump(cart_model, ARTIFACTS_DIR / "cart_model.joblib")

    # 5.2 XGBoost (scale_pos_weight, no SMOTE)
    print("Training XGBoost Classifier...")
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    scale_pos_weight = float(neg_count / max(1, pos_count))
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    xgb_model.fit(X_train_aug, y_train)
    joblib.dump(xgb_model, ARTIFACTS_DIR / "xgb_model.joblib")

    # 5.3 Soft Voting Ensemble
    ensemble_model = SoftVotingEnsemble(xgb_model, cart_model)

    print("\n=== Step 6: Validation Model Selection & Threshold Tuning ===")
    # Predict probabilities on validation set
    p_val_cart = cart_model.predict_proba(X_val_aug)[:, 1]
    p_val_xgb = xgb_model.predict_proba(X_val_aug)[:, 1]
    p_val_ens = ensemble_model.predict_proba(X_val_aug)[:, 1]

    # Calculate validation PR-AUC for each model
    p_prec_c, p_rec_c, _ = precision_recall_curve(y_val, p_val_cart)
    val_pr_auc_cart = float(auc(p_rec_c, p_prec_c))

    p_prec_x, p_rec_x, _ = precision_recall_curve(y_val, p_val_xgb)
    val_pr_auc_xgb = float(auc(p_rec_x, p_prec_x))

    p_prec_e, p_rec_e, _ = precision_recall_curve(y_val, p_val_ens)
    val_pr_auc_ens = float(auc(p_rec_e, p_prec_e))

    print(f"Validation PR-AUC -> CART: {val_pr_auc_cart:.4f}, XGBoost: {val_pr_auc_xgb:.4f}, Ensemble: {val_pr_auc_ens:.4f}")

    # Code-driven selection of best model based on validation PR-AUC
    candidates = [
        ("XGBoost", val_pr_auc_xgb, xgb_model, p_val_xgb),
        ("CART", val_pr_auc_cart, cart_model, p_val_cart),
        ("Ensemble", val_pr_auc_ens, ensemble_model, p_val_ens),
    ]
    candidates.sort(key=lambda c: c[1], reverse=True)
    final_name, best_val_pr_auc, final_model, p_val_best = candidates[0]
    print(f"Selected FINAL model by validation PR-AUC: {final_name} ({best_val_pr_auc:.4f})")

    # Save final model
    joblib.dump(final_model, ARTIFACTS_DIR / "final_model.joblib")

    # Tune decision threshold on validation set only
    best_threshold, val_f1, val_prec, val_rec = tune_decision_threshold(y_val, p_val_best)
    print(f"Validation-tuned decision threshold: {best_threshold:.4f} (Val F1: {val_f1:.4f}, Prec: {val_prec:.4f}, Rec: {val_rec:.4f})")

    print("\n=== Step 7: Ablation Study (With vs. Without Unsupervised Columns) ===")
    # Train XGBoost without unsupervised columns
    xgb_base = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )
    xgb_base.fit(X_train_base.values, y_train)
    p_test_base = xgb_base.predict_proba(X_test_base.values)[:, 1]
    p_prec_base, p_rec_base, _ = precision_recall_curve(y_test, p_test_base)
    test_pr_auc_base = float(auc(p_rec_base, p_prec_base))

    p_test_xgb = xgb_model.predict_proba(X_test_aug)[:, 1]
    p_prec_xgb, p_rec_xgb, _ = precision_recall_curve(y_test, p_test_xgb)
    test_pr_auc_xgb_aug = float(auc(p_rec_xgb, p_prec_xgb))

    ablation_diff = test_pr_auc_xgb_aug - test_pr_auc_base
    if abs(ablation_diff) < 0.02:
        ablation_verdict = (
            f"Unsupervised columns (GMM, HMM, K-Means) add marginal improvement ({ablation_diff:+.4f} PR-AUC). "
            f"Supervised tree structures capture the majority of non-linear interactions directly from base features."
        )
    else:
        ablation_verdict = (
            f"Unsupervised features provide a noticeable gain ({ablation_diff:+.4f} PR-AUC) "
            f"by supplying sequence dynamics and density baselines."
        )
    print(f"Ablation result: Base XGBoost PR-AUC={test_pr_auc_base:.4f}, Augmented XGBoost PR-AUC={test_pr_auc_xgb_aug:.4f}")
    print(f"Honest Assessment: {ablation_verdict}")

    print("\n=== Step 8: Full Test Set Evaluation ===")
    p_test_cart = cart_model.predict_proba(X_test_aug)[:, 1]
    p_test_ens = ensemble_model.predict_proba(X_test_aug)[:, 1]

    # Evaluate final model
    if final_name == "XGBoost":
        p_test_final = p_test_xgb
    elif final_name == "CART":
        p_test_final = p_test_cart
    else:
        p_test_final = p_test_ens

    test_metrics = evaluate_model_metrics(y_test, p_test_final, best_threshold)
    cart_metrics = evaluate_model_metrics(y_test, p_test_cart, best_threshold)
    xgb_metrics = evaluate_model_metrics(y_test, p_test_xgb, best_threshold)
    ens_metrics = evaluate_model_metrics(y_test, p_test_ens, best_threshold)

    print(f"HEADLINE METRIC -> Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    print(f"Secondary Metrics -> ROC-AUC: {test_metrics['roc_auc']:.4f}, Precision: {test_metrics['precision']:.4f}, Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # Save metrics.json
    all_metrics = {
        "random_seed": RANDOM_SEED,
        "selected_model": final_name,
        "tuned_threshold": best_threshold,
        "threshold_suspicious": DEFAULT_THRESHOLD_SUSPICIOUS,
        "headline_metric": "PR-AUC",
        "final_test_metrics": test_metrics,
        "models_comparison": {
            "XGBoost": xgb_metrics,
            "CART": cart_metrics,
            "Ensemble": ens_metrics,
        },
        "validation_pr_auc": {
            "XGBoost": val_pr_auc_xgb,
            "CART": val_pr_auc_cart,
            "Ensemble": val_pr_auc_ens,
        },
        "ablation_study": {
            "xgboost_without_unsupervised_pr_auc": test_pr_auc_base,
            "xgboost_with_unsupervised_pr_auc": test_pr_auc_xgb_aug,
            "pr_auc_difference": ablation_diff,
            "assessment": ablation_verdict,
        },
        "kmeans_silhouette": kmeans.silhouette_,
        "em_log_likelihood_history": gmm.log_likelihood_history_,
        "dataset_splits": stats,
        "feature_names": augmented_feature_names,
    }

    metrics_file = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Saved complete metrics to {metrics_file}.")

    print("\n=== Step 9: Precomputing Test Scores and Parquet Generation ===")
    test_probs = p_test_final
    test_risk_scores = np.clip(test_probs * 100.0, 0.0, 100.0)

    # Risk verdicts
    verdicts = np.full(len(test_df), "Normal", dtype=object)
    verdicts[test_probs >= DEFAULT_THRESHOLD_SUSPICIOUS] = "Suspicious"
    verdicts[test_probs >= best_threshold] = "Fraud"

    # Assemble fast lightweight precomputed parquet
    parquet_df = pd.DataFrame(
        {
            "trans_num": test_df["trans_num"].astype(str),
            "trans_date_trans_time": test_df["trans_date_trans_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cc_num": test_df["cc_num"].astype(np.int64),
            "merchant": test_df["merchant"].astype(str),
            "category": test_df["category"].astype(str),
            "amt": test_df["amt"].astype(float),
            "log_amt": test_df["log_amt"].astype(float),
            "distance_km": test_df["distance_km"].round(1).astype(float),
            "hour": test_df["hour"].astype(int),
            "day_of_week": test_df["day_of_week"].astype(int),
            "is_night": test_df["is_night"].astype(int),
            "age": test_df["age"].astype(int),
            "gender": test_df["gender"].astype(int),
            "city_pop": test_df["city_pop"].astype(float),
            "is_fraud": test_df["is_fraud"].astype(int),
            "prob": test_probs.astype(float),
            "risk_score": test_risk_scores.round(1).astype(float),
            "verdict": verdicts.astype(str),
            "kmeans_cluster": test_clusters.astype(int),
            "kmeans_distance": test_dists.round(3).astype(float),
            "gmm_score": gmm_test_score.round(3).astype(float),
            "hmm_score": hmm_test_score.round(3).astype(float),
            "pca_1": X_test_pca[:, 0].round(3).astype(float),
            "pca_2": X_test_pca[:, 1].round(3).astype(float),
            "amt_zscore_vs_customer_history": test_df["amt_zscore_vs_customer_history"].round(2).astype(float),
            "secs_since_last_txn": test_df["secs_since_last_txn"].astype(float),
            "txn_count_last_24h": test_df["txn_count_last_24h"].astype(float),
        }
    )

    parquet_file = ARTIFACTS_DIR / "test_scored.parquet"
    parquet_df.to_parquet(parquet_file, index=False)
    print(f"Saved {len(parquet_df):,} scored test records to {parquet_file} ({parquet_file.stat().st_size / (1024*1024):.1f} MB).")

    # Also precompute and save CART tree text rules for the Algorithms tab
    tree_rules = export_text(cart_model, feature_names=augmented_feature_names, max_depth=4)
    with open(ARTIFACTS_DIR / "tree_rules.txt", "w", encoding="utf-8") as f:
        f.write(tree_rules)

    total_time = time.time() - start_time
    print(f"\nAuditLens Training Pipeline completed successfully in {total_time:.1f}s.")
    return all_metrics


if __name__ == "__main__":
    train_pipeline()

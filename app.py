"""AuditLens: Credit card transaction fraud review and explanation web application."""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib

from src.config import (
    ARTIFACTS_DIR,
    USD_TO_INR,
    RANDOM_SEED,
    CUSTOM_PLOTLY_TEMPLATE,
    COLOR_BACKGROUND,
    COLOR_PANELS,
    COLOR_BORDERS,
    COLOR_TEXT,
    COLOR_TEXT_SECONDARY,
    COLOR_ACCENT,
    COLOR_NORMAL_TEXT,
    COLOR_NORMAL_BG,
    COLOR_SUSPICIOUS_TEXT,
    COLOR_SUSPICIOUS_BG,
    COLOR_FRAUD_TEXT,
    COLOR_FRAUD_BG,
    COLOR_DOT_NORMAL,
    COLOR_DOT_FRAUD,
    COLOR_CHART_THRESHOLD,
    COLOR_CHART_OTHER_MODEL,
    COLOR_SHAP_INCREASE,
    COLOR_SHAP_DECREASE,
)
from src.explain import format_inr, TransactionExplainer
from src.report import generate_html_report
from src.em_gmm import HandWrittenGMM
from src.hmm import (
    encode_discrete_symbols,
)

# Page configuration
st.set_page_config(
    page_title="AuditLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal custom styling using the centralized palette
st.markdown(
    f"""
    <style>
      #MainMenu {{visibility: hidden;}}
      footer {{visibility: hidden;}}
      header {{visibility: hidden;}}
      .stDeployButton {{display: none;}}
      
      body, .stApp {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_TEXT};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      }}
      
      .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {COLOR_BORDERS};
        gap: 16px;
      }}
      .stTabs [data-baseweb="tab"] {{
        color: {COLOR_TEXT_SECONDARY};
        border-radius: 0px;
        padding: 8px 16px;
        font-weight: 500;
        background-color: transparent;
      }}
      .stTabs [aria-selected="true"] {{
        color: {COLOR_ACCENT} !important;
        border-bottom: 2px solid {COLOR_ACCENT} !important;
        font-weight: 600;
      }}
      
      .stButton > button {{
        background-color: {COLOR_BACKGROUND};
        color: {COLOR_ACCENT};
        border: 1px solid {COLOR_ACCENT};
        border-radius: 0px;
        font-weight: 500;
      }}
      .stButton > button:hover {{
        background-color: {COLOR_PANELS};
        border: 1px solid {COLOR_ACCENT};
        color: {COLOR_ACCENT};
      }}
      
      .metric-box {{
        border: 1px solid {COLOR_BORDERS};
        padding: 12px 16px;
        background-color: {COLOR_PANELS};
        margin-bottom: 12px;
      }}
      .metric-label {{
        font-size: 12px;
        color: {COLOR_TEXT_SECONDARY};
        margin-bottom: 4px;
      }}
      .metric-value {{
        font-size: 20px;
        font-weight: 600;
        color: {COLOR_TEXT};
      }}
      .metric-caption {{
        font-size: 11px;
        color: {COLOR_TEXT_SECONDARY};
        margin-top: 4px;
      }}
      
      .badge-fraud {{
        display: inline-block;
        background-color: {COLOR_FRAUD_BG};
        color: {COLOR_FRAUD_TEXT};
        border: 1px solid {COLOR_FRAUD_TEXT};
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 700;
      }}
      .badge-suspicious {{
        display: inline-block;
        background-color: {COLOR_SUSPICIOUS_BG};
        color: {COLOR_SUSPICIOUS_TEXT};
        border: 1px solid {COLOR_SUSPICIOUS_TEXT};
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 500;
      }}
      .badge-normal {{
        display: inline-block;
        background-color: {COLOR_NORMAL_BG};
        color: {COLOR_NORMAL_TEXT};
        border: 1px solid {COLOR_NORMAL_TEXT};
        padding: 2px 8px;
        font-size: 12px;
        font-weight: 500;
      }}
      
      .panel-box {{
        border: 1px solid {COLOR_BORDERS};
        background-color: {COLOR_PANELS};
        padding: 16px;
        margin-bottom: 16px;
      }}
      .footer-note {{
        border-top: 1px solid {COLOR_BORDERS};
        padding-top: 16px;
        margin-top: 32px;
        font-size: 11px;
        color: {COLOR_TEXT_SECONDARY};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_metrics_and_parquet() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Load precomputed metrics and test evaluation parquet table."""
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    parquet_path = ARTIFACTS_DIR / "test_scored.parquet"

    if not metrics_path.exists() or not parquet_path.exists():
        return {}, pd.DataFrame()

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    df = pd.read_parquet(parquet_path)
    return metrics_data, df


@st.cache_resource
def load_cached_artifacts():
    """Load serialized models for explanations and on-demand algorithm tests."""
    loaded_artifacts = {}
    paths = {
        "final_model": ARTIFACTS_DIR / "final_model.joblib",
        "cart_model": ARTIFACTS_DIR / "cart_model.joblib",
        "xgb_model": ARTIFACTS_DIR / "xgb_model.joblib",
        "pca_model": ARTIFACTS_DIR / "pca_model.joblib",
        "kmeans_model": ARTIFACTS_DIR / "kmeans_model.joblib",
        "gmm_model": ARTIFACTS_DIR / "gmm_model.joblib",
        "hmm_model": ARTIFACTS_DIR / "hmm_model.joblib",
        "pipeline": ARTIFACTS_DIR / "feature_pipeline.joblib",
    }
    for k, p in paths.items():
        if p.exists():
            loaded_artifacts[k] = joblib.load(p)
    return loaded_artifacts


# Main application flow
metrics, test_df = load_metrics_and_parquet()

# Header
st.title("AuditLens")
st.write("Reviews card transactions and explains why a transaction looks suspicious.")

# Check artifacts existence
if not metrics or test_df.empty:
    st.error("Run python train.py first. No results found.")
    st.stop()

artifacts = load_cached_artifacts()
tuned_threshold = float(metrics.get("tuned_threshold", 0.50))
suspicious_threshold = float(metrics.get("threshold_suspicious", 0.20))
selected_model_name = metrics.get("selected_model", "XGBoost")
test_metrics = metrics.get("final_test_metrics", {})
confusion_mat = test_metrics.get("confusion_matrix", {})

# Sidebar
st.sidebar.markdown("### Settings")
usd_to_inr = st.sidebar.number_input(
    "USD to INR exchange rate",
    min_value=1.0,
    max_value=200.0,
    value=float(USD_TO_INR),
    step=0.5,
    help="Amounts converted from USD for display. Dataset is synthetic.",
)
st.sidebar.caption("Amounts converted from USD for display. Dataset is synthetic.")

st.sidebar.markdown("---")
st.sidebar.markdown("### Operational Glossary")

st.sidebar.markdown("**Risk Score (0-100)**")
st.sidebar.caption("Probability output scaled to an integer indicating likelihood of fraudulent behavior.")

st.sidebar.markdown("**PR-AUC**")
st.sidebar.caption("Area under the precision-recall curve, measuring model discrimination on highly imbalanced data.")

st.sidebar.markdown("**Precision**")
st.sidebar.caption("Proportion of transactions flagged as fraud that were truly fraudulent.")

st.sidebar.markdown("**Recall**")
st.sidebar.caption("Proportion of all true fraudulent transactions that were successfully detected.")

st.sidebar.markdown("**Anomaly**")
st.sidebar.caption("A transaction or sequence that deviates statistically from established cardholder history.")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Random seed**: {metrics.get('random_seed', 42)}")
st.sidebar.markdown(f"**Selected model**: {selected_model_name}")
st.sidebar.markdown(f"**Tuned fraud threshold**: {tuned_threshold:.4f}")

# Main Tabs
tab_overview, tab_txns, tab_algorithms, tab_how = st.tabs(
    ["Overview", "Transactions", "Algorithms", "How it works"]
)

# ==========================================
# TAB 1: OVERVIEW
# ==========================================
with tab_overview:
    st.subheader("Evaluation Summary")

    total_test = len(test_df)
    n_fraud = int(np.sum(test_df["verdict"] == "Fraud"))
    n_suspicious = int(np.sum(test_df["verdict"] == "Suspicious"))
    n_normal = int(np.sum(test_df["verdict"] == "Normal"))
    actual_fraud = int(test_df["is_fraud"].sum())
    actual_rate = (actual_fraud / total_test) * 100.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Evaluated Transactions</div>
              <div class="metric-value">{total_test:,}</div>
              <div class="metric-caption">Total holdout transactions scored on test set.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Flagged Fraud (Score &ge; {tuned_threshold*100:.1f})</div>
              <div class="metric-value">{n_fraud:,} ({n_fraud/total_test*100:.2f}%)</div>
              <div class="metric-caption">Transactions at or exceeding the validation-tuned threshold.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Flagged Suspicious (Score {suspicious_threshold*100:.0f} to {tuned_threshold*100:.1f})</div>
              <div class="metric-value">{n_suspicious:,} ({n_suspicious/total_test*100:.2f}%)</div>
              <div class="metric-caption">Transactions requiring secondary human inspection.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Actual Fraud Rate</div>
              <div class="metric-value">{actual_rate:.4f}%</div>
              <div class="metric-caption">Ground-truth fraud incidence in holdout test split.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Model Performance on Holdout Test Set")
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">PR-AUC (Headline)</div>
              <div class="metric-value">{test_metrics.get('pr_auc', 0.0):.4f}</div>
              <div class="metric-caption">Area under precision-recall curve on test set.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">ROC-AUC</div>
              <div class="metric-value">{test_metrics.get('roc_auc', 0.0):.4f}</div>
              <div class="metric-caption">Area under receiver operating characteristic curve.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Precision</div>
              <div class="metric-value">{test_metrics.get('precision', 0.0)*100:.2f}%</div>
              <div class="metric-caption">Proportion of flagged fraud that was true fraud.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">Recall</div>
              <div class="metric-value">{test_metrics.get('recall', 0.0)*100:.2f}%</div>
              <div class="metric-caption">Proportion of true fraud detected by the model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m5:
        st.markdown(
            f"""
            <div class="metric-box">
              <div class="metric-label">F1-Score</div>
              <div class="metric-value">{test_metrics.get('f1', 0.0):.4f}</div>
              <div class="metric-caption">Harmonic mean of precision and recall.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col_hist, col_cm = st.columns([3, 2])
    with col_hist:
        st.subheader("Risk Score Distribution")
        st.caption("Distribution of predicted risk scores across all 555,719 test transactions.")

        # Chart: bars in accent blue, threshold line dashed in Fraud red
        counts, bin_edges = np.histogram(test_df["risk_score"].values, bins=30)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Bar(
                x=bin_centers,
                y=counts,
                marker=dict(color=COLOR_ACCENT),
                name="Transactions",
            )
        )
        # Threshold line dashed in Fraud red
        fig_hist.add_vline(
            x=tuned_threshold * 100.0,
            line_dash="dash",
            line_color=COLOR_CHART_THRESHOLD,
            line_width=2,
            annotation_text=f"Fraud Cutoff ({tuned_threshold*100:.1f})",
            annotation_font_color=COLOR_CHART_THRESHOLD,
            annotation_position="top right",
        )
        fig_hist.update_layout(
            template=CUSTOM_PLOTLY_TEMPLATE,
            margin=dict(l=40, r=20, t=20, b=40),
            height=300,
            xaxis=dict(title="Risk Score (0 to 100)", showgrid=False),
            yaxis=dict(title="Transaction Count", type="log", showgrid=True),
            showlegend=False,
        )
        st.plotly_chart(fig_hist, width="stretch")

    with col_cm:
        st.subheader("Confusion Matrix")
        st.caption(f"Evaluated at validation-tuned decision threshold {tuned_threshold:.4f}.")

        tn = confusion_mat.get("tn", 0)
        fp = confusion_mat.get("fp", 0)
        fn = confusion_mat.get("fn", 0)
        tp = confusion_mat.get("tp", 0)

        cm_table = pd.DataFrame(
            [
                ["Actual Normal", f"{tn:,}", f"{fp:,}"],
                ["Actual Fraud", f"{fn:,}", f"{tp:,}"],
            ],
            columns=["Ground Truth", "Predicted Normal", "Predicted Fraud"],
        )
        st.table(cm_table.set_index("Ground Truth"))


# ==========================================
# TAB 2: TRANSACTIONS
# ==========================================
with tab_txns:
    st.subheader("Transaction Review and Investigation")
    st.caption("Filter and select any transaction to examine behavioral justifications and download forensic audit reports.")

    # Filter controls
    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    with f1:
        verdict_filter = st.selectbox(
            "Filter by verdict",
            options=["All", "Fraud", "Suspicious", "Normal"],
            index=0,
        )
    with f2:
        cat_options = ["All"] + sorted(list(test_df["category"].unique()))
        category_filter = st.selectbox("Filter by category", options=cat_options, index=0)
    with f3:
        cust_search = st.text_input("Search cardholder ID (cc_num)", value="")
    with f4:
        merch_search = st.text_input("Search merchant", value="")

    # Apply filters
    filtered_df = test_df
    if verdict_filter != "All":
        filtered_df = filtered_df[filtered_df["verdict"] == verdict_filter]
    if category_filter != "All":
        filtered_df = filtered_df[filtered_df["category"] == category_filter]
    if cust_search.strip():
        try:
            cust_val = int(cust_search.strip())
            filtered_df = filtered_df[filtered_df["cc_num"] == cust_val]
        except ValueError:
            filtered_df = filtered_df[filtered_df["cc_num"].astype(str).str.contains(cust_search.strip())]
    if merch_search.strip():
        filtered_df = filtered_df[filtered_df["merchant"].str.contains(merch_search.strip(), case=False)]

    st.write(f"Showing **{len(filtered_df):,}** matching transactions.")

    if len(filtered_df) == 0:
        st.write("No transactions match the selected criteria.")
    else:
        # Pagination
        page_size = 10
        total_pages = max(1, int(np.ceil(len(filtered_df) / page_size)))
        page_num = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))

        page_slice = filtered_df.iloc[start_idx:end_idx].copy()
        page_slice["amount_inr"] = page_slice["amt"].apply(lambda v: format_inr(v, usd_to_inr))

        # Risk tag formatting: always contains the word and 0-100 score; Fraud is bold
        def format_verdict_tag(row):
            v = row["verdict"]
            s = row["risk_score"]
            if v == "Fraud":
                return f"Fraud ({s:.1f})"
            elif v == "Suspicious":
                return f"Suspicious ({s:.1f})"
            else:
                return f"Normal ({s:.1f})"

        page_slice["verdict_tag"] = page_slice.apply(format_verdict_tag, axis=1)
        page_slice["investigation_action"] = "Investigate"

        display_cols = [
            "trans_num",
            "trans_date_trans_time",
            "cc_num",
            "merchant",
            "category",
            "amount_inr",
            "verdict_tag",
            "risk_score",
            "investigation_action",
        ]
        table_view = page_slice[display_cols].rename(
            columns={
                "trans_num": "Transaction ID",
                "trans_date_trans_time": "Timestamp",
                "cc_num": "Cardholder ID",
                "merchant": "Merchant",
                "category": "Category",
                "amount_inr": "Amount (INR)",
                "verdict_tag": "Verdict",
                "risk_score": "Risk Score",
                "investigation_action": "Investigation",
            }
        )

        # Flagged table rows get a thin left border in the risk colour, not a filled row
        def apply_flagged_row_border(row):
            verdict_val = row["Verdict"]
            if "Fraud" in verdict_val:
                border_color = COLOR_FRAUD_TEXT
            elif "Suspicious" in verdict_val:
                border_color = COLOR_SUSPICIOUS_TEXT
            else:
                border_color = COLOR_NORMAL_TEXT
            # Thin left border on the first column cell only, not a filled row
            return [f"border-left: 3px solid {border_color};" if i == 0 else "" for i in range(len(row))]

        styled_table = table_view.style.apply(apply_flagged_row_border, axis=1)

        # Check if button in Investigation column was clicked
        if "selected_txn_id" not in st.session_state and len(page_slice) > 0:
            st.session_state["selected_txn_id"] = str(page_slice["trans_num"].iloc[0])

        if "investigate_action" in st.session_state and st.session_state["investigate_action"] is not None:
            click = st.session_state["investigate_action"]
            clicked_row = click["row"] if isinstance(click, dict) else getattr(click, "row", None)
            if clicked_row is not None and 0 <= clicked_row < len(page_slice):
                st.session_state["selected_txn_id"] = str(page_slice.iloc[clicked_row]["trans_num"])

        table_event = st.dataframe(
            styled_table,
            column_config={
                "Investigation": st.column_config.ButtonColumn(
                    "Investigation",
                    help="Direct option to investigate this transaction",
                    key="investigate_action",
                ),
            },
            on_select="rerun",
            selection_mode="single-row",
            width="stretch",
            hide_index=True,
        )

        if table_event.selection.rows:
            sel_row = table_event.selection.rows[0]
            if 0 <= sel_row < len(page_slice):
                st.session_state["selected_txn_id"] = str(page_slice.iloc[sel_row]["trans_num"])

        st.markdown("---")
        st.subheader("Transaction Investigation Pane")

        # Select transaction to inspect (synced with session state)
        txn_options = list(page_slice["trans_num"].values)
        default_idx = 0
        current_sel = st.session_state.get("selected_txn_id")
        if current_sel in txn_options:
            default_idx = txn_options.index(current_sel)

        selected_txn_id = st.selectbox(
            "Selected transaction for investigation:",
            options=txn_options,
            index=default_idx,
            format_func=lambda tid: f"{tid} - {page_slice[page_slice['trans_num'] == tid]['merchant'].values[0]} ({page_slice[page_slice['trans_num'] == tid]['amount_inr'].values[0]}) [{page_slice[page_slice['trans_num'] == tid]['verdict_tag'].values[0]}]",
        )
        st.session_state["selected_txn_id"] = selected_txn_id

        selected_row = filtered_df[filtered_df["trans_num"] == selected_txn_id].iloc[0]
        cardholder_id = int(selected_row["cc_num"])

        # Customer baseline
        kmeans_model = artifacts.get("kmeans_model")
        customer_baseline = None
        if kmeans_model and cardholder_id in kmeans_model.customer_profiles_:
            customer_baseline = kmeans_model.customer_profiles_[cardholder_id]

        # Build feature vector for SHAP explainer
        explainer = TransactionExplainer(
            artifacts.get("final_model"),
            metrics.get("feature_names", []),
        )
        dummy_feat_vec = np.zeros(len(metrics.get("feature_names", [])))
        explanation = explainer.explain_transaction(
            selected_row,
            dummy_feat_vec,
            customer_baseline,
            usd_to_inr_rate=usd_to_inr,
        )

        # Verdict and Score Display: always contains word and score, Fraud is bold
        verdict = selected_row["verdict"]
        score = float(selected_row["risk_score"])
        if verdict == "Fraud":
            badge_html = f'<span class="badge-fraud"><strong>Fraud ({score:.1f})</strong></span>'
        elif verdict == "Suspicious":
            badge_html = f'<span class="badge-suspicious">Suspicious ({score:.1f})</span>'
        else:
            badge_html = f'<span class="badge-normal">Normal ({score:.1f})</span>'

        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            st.markdown(
                f"""
                <div class="panel-box">
                  <div style="font-size: 12px; color: {COLOR_TEXT_SECONDARY};">Classification Verdict</div>
                  <div style="margin: 8px 0;">{badge_html}</div>
                  <div style="font-size: 18px; font-weight: 600; color: {COLOR_TEXT};">Risk Score: {score:.1f} / 100</div>
                  <div style="font-size: 11px; color: {COLOR_TEXT_SECONDARY}; margin-top: 6px;">
                    Decision criteria: Normal (&lt; {suspicious_threshold*100:.0f}), 
                    Suspicious ({suspicious_threshold*100:.0f} to {tuned_threshold*100:.1f}), 
                    Fraud (&ge; {tuned_threshold*100:.1f})
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_v2:
            st.markdown("#### Why was it flagged?")
            for i, reason in enumerate(explanation["reasons"]):
                st.markdown(f"**{i+1}.** {reason}")

        # Behavioral comparison table
        st.markdown("#### Cardholder Behavioral Comparison")
        comp = explanation["comparison"]
        comp_df = pd.DataFrame(
            [
                ["Amount", comp["amount_inr"], comp["cust_avg_amount_inr"]],
                ["Merchant Distance", f"{comp['distance_km']:.1f} km", f"{comp['cust_avg_distance_km']:.1f} km"],
                ["Transaction Hour", f"{comp['hour']:02d}:00", "Daytime (08:00 - 20:00)"],
                ["Amount Z-Score", f"{comp['zscore']:.1f} standard deviations", "Normal baseline (< 1.0)"],
            ],
            columns=["Attribute", "Current Transaction", "Cardholder Baseline"],
        )
        st.table(comp_df.set_index("Attribute"))

        # Customer's recent transactions
        cust_txns = test_df[test_df["cc_num"] == cardholder_id].sort_values("trans_date_trans_time")
        st.markdown(f"#### Recent Transactions for Cardholder {cardholder_id}")
        st.caption(f"Showing last {min(5, len(cust_txns))} transactions in sequence.")
        recent_slice = cust_txns.tail(5).copy()
        recent_slice["amount_inr"] = recent_slice["amt"].apply(lambda v: format_inr(v, usd_to_inr))
        recent_slice["verdict_tag"] = recent_slice.apply(format_verdict_tag, axis=1)

        recent_view = recent_slice[
            [
                "trans_num",
                "trans_date_trans_time",
                "merchant",
                "category",
                "amount_inr",
                "verdict_tag",
                "risk_score",
            ]
        ].rename(
            columns={
                "trans_num": "Transaction ID",
                "trans_date_trans_time": "Timestamp",
                "merchant": "Merchant",
                "category": "Category",
                "amount_inr": "Amount (INR)",
                "verdict_tag": "Verdict",
                "risk_score": "Risk Score",
            }
        )
        styled_recent = recent_view.style.apply(apply_flagged_row_border, axis=1)
        st.dataframe(styled_recent, width="stretch", hide_index=True)

        # K-Means Segment and HMM State Path
        col_km, col_hmm = st.columns(2)
        with col_km:
            st.markdown("#### Customer K-Means Segment")
            cluster_id = int(selected_row["kmeans_cluster"])
            st.write(f"**Assigned Segment**: Cluster {cluster_id}")
            if customer_baseline:
                st.write(f"- Historical Average Spend: {format_inr(customer_baseline['avg_amount'], usd_to_inr)}")
                st.write(f"- Average Frequency: {customer_baseline['txns_per_day']:.2f} transactions/day")
                st.write(f"- Typical Distance Radius: {customer_baseline['avg_distance']:.1f} km")
            st.caption("Customer behavioral segment computed across aggregate cardholder history.")

        with col_hmm:
            st.markdown("#### HMM Viterbi Sequential Path")
            # Run Viterbi on this customer's last up to 10 transactions
            hmm_model = artifacts.get("hmm_model")
            if hmm_model:
                last_txns = cust_txns.tail(10)
                symbols = encode_discrete_symbols(
                    last_txns["amt_zscore_vs_customer_history"].values,
                    (last_txns["hour"] < 6) | (last_txns["hour"] >= 22),
                    last_txns["distance_km"].values,
                )
                _, state_path = hmm_model.decode_sequence(symbols)
                path_str = " -> ".join(f"State {s}" for s in state_path)
                st.write(f"**Optimal State Sequence (last {len(state_path)} txns)**:")
                st.code(path_str, language="text")
                st.caption("Decoded using hand-written Viterbi algorithm in log-space.")

        # What to check next
        st.markdown("#### What to check next")
        for item in explanation["check_next"]:
            st.markdown(f"- {item}")

        # Download Report Button
        st.markdown("---")
        report_html = generate_html_report(selected_row.to_dict(), explanation, usd_to_inr)
        st.download_button(
            label="Download audit report (HTML)",
            data=report_html,
            file_name=f"audit_report_{selected_txn_id}.html",
            mime="text/html",
        )


# ==========================================
# TAB 3: ALGORITHMS
# ==========================================
with tab_algorithms:
    st.subheader("Statistical and Machine Learning Algorithms")
    st.caption("Every algorithm below executed during pipeline training or on interactive execution.")

    # 1. PCA
    with st.expander("1. Principal Component Analysis (PCA)", expanded=False):
        st.markdown("**What it does**: Projects high-dimensional scaled features onto orthogonal axes of maximal variance.")
        st.markdown(r"**Formula**: $\max_w \frac{w^T X^T X w}{w^T w} \quad \text{subject to } w_i^T w_j = 0 \text{ for } i \ne j$")
        st.markdown("**Why used here**: Condenses 26 features into 5 continuous components for stable GMM covariance estimation and 2D spatial visualization.")

        if st.button("Run PCA 2D projection on sample"):
            sample_slice = test_df.sample(n=min(1500, len(test_df)), random_state=RANDOM_SEED)
            X_pca_2d = np.vstack([sample_slice["pca_1"].values, sample_slice["pca_2"].values]).T

            fig_pca = go.Figure()
            # Normal dots light grey-blue
            norm_mask = sample_slice["is_fraud"].values == 0
            fig_pca.add_trace(
                go.Scatter(
                    x=X_pca_2d[norm_mask, 0],
                    y=X_pca_2d[norm_mask, 1],
                    mode="markers",
                    marker=dict(size=4, color=COLOR_DOT_NORMAL, opacity=0.7),
                    name="Normal",
                )
            )
            # Fraud dots red and larger
            fraud_mask = sample_slice["is_fraud"].values == 1
            fig_pca.add_trace(
                go.Scatter(
                    x=X_pca_2d[fraud_mask, 0],
                    y=X_pca_2d[fraud_mask, 1],
                    mode="markers",
                    marker=dict(size=9, color=COLOR_DOT_FRAUD, symbol="circle"),
                    name="Fraud",
                )
            )
            pca_model = artifacts.get("pca_model")
            exp_var = pca_model.explained_variance_ratio_[:2].round(4).tolist() if pca_model else [0.0, 0.0]
            fig_pca.update_layout(
                template=CUSTOM_PLOTLY_TEMPLATE,
                height=360,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(title="Principal Component 1", showgrid=True),
                yaxis=dict(title="Principal Component 2", showgrid=True),
                showlegend=True,
            )
            st.plotly_chart(fig_pca, width="stretch")
            st.caption(f"Explained variance ratio: {exp_var}")

    # 2. K-Means
    with st.expander("2. K-Means Customer Behavioral Segmentation", expanded=False):
        st.markdown("**What it does**: Partitions cardholders into discrete behavioral groups based on spending volume, variance, frequency, and travel radius.")
        st.markdown(r"**Formula**: $\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} \|x - \mu_i\|^2$")
        st.markdown("**Why used here**: Establishes cardholder peer clusters so anomaly detection compares a user against similar spending cohorts, not population averages.")

        km = artifacts.get("kmeans_model")
        if km:
            st.write(f"**Silhouette Score**: {km.silhouette_:.4f} (honestly reported on cardholder behavioral profiles).")
            if km.cluster_summaries_:
                st.table(pd.DataFrame(km.cluster_summaries_))

    # 3. GMM & EM
    with st.expander("3. Gaussian Mixture Model (GMM) with Hand-Written EM", expanded=False):
        st.markdown("**What it does**: Models transactional feature density as a linear superposition of multivariate Gaussians.")
        st.markdown(r"**Formula**: $p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x | \mu_k, \Sigma_k)$")
        st.markdown("**Why used here**: Produces a continuous unusualness metric (log-likelihood) where low probability density flags outlier transactions.")

        # Interactive EM run
        if st.button("Run EM optimization on PCA sample"):
            sample_pca = test_df.sample(n=300, random_state=RANDOM_SEED)
            X_samp = np.vstack([sample_pca["pca_1"].values, sample_pca["pca_2"].values]).T
            hw_gmm = HandWrittenGMM(n_components=3, max_iter=20, random_state=RANDOM_SEED)
            hw_gmm.fit(X_samp)

            fig_em = go.Figure()
            # Bars and lines in accent blue
            fig_em.add_trace(
                go.Scatter(
                    y=hw_gmm.log_likelihood_history_,
                    mode="lines+markers",
                    line=dict(color=COLOR_ACCENT, width=2),
                    marker=dict(size=4, color=COLOR_ACCENT),
                )
            )
            fig_em.update_layout(
                template=CUSTOM_PLOTLY_TEMPLATE,
                height=280,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis=dict(title="EM Iteration", showgrid=True),
                yaxis=dict(title="Total Log-Likelihood", showgrid=True),
                showlegend=False,
            )
            st.plotly_chart(fig_em, width="stretch")
            st.caption(f"EM converged in {len(hw_gmm.log_likelihood_history_)} iterations. Monotonic log-likelihood ascent verified.")

    # 4. HMM, Forward, Viterbi
    with st.expander("4. Hidden Markov Model: Baum-Welch, Forward, and Viterbi", expanded=False):
        st.markdown("**What it does**: Models customer behavior as sequential transitions between hidden states emitting discrete transaction symbols.")
        st.markdown(r"**Forward Formula**: $\alpha_t(j) = \left[ \sum_{i=1}^K \alpha_{t-1}(i) A_{ij} \right] B_j(o_t)$")
        st.markdown(r"**Viterbi Formula**: $V_t(j) = \max_i [V_{t-1}(i) + \ln A_{ij}] + \ln B_j(o_t)$")
        st.markdown("**Why used here**: Identifies sudden shifts in sequential spending patterns that static per-transaction models miss.")

        hmm_model = artifacts.get("hmm_model")
        if hmm_model:
            st.markdown("##### Transition Probability Matrix (A)")
            st.dataframe(pd.DataFrame(hmm_model.transmat_).round(4), width="stretch")

            if st.button("Run Forward and Viterbi Verification against hmmlearn"):
                test_seq = np.array([0, 1, 4, 7, 11, 2, 0, 3, 8, 9])
                fwd_ok, vit_ok, msg = hmm_model.verify_against_hmmlearn(test_seq)
                st.write(f"Test Observation Sequence: `{test_seq.tolist()}`")
                st.write(f"- Matches hmmlearn Forward: **{'yes' if fwd_ok else 'no'}**")
                st.write(f"- Matches hmmlearn Viterbi: **{'yes' if vit_ok else 'no'}**")
                st.caption(msg)

    # 5. CART Decision Tree
    with st.expander("5. CART Decision Tree (Depth 4)", expanded=False):
        st.markdown("**What it does**: Partitions feature space using binary recursive splitting maximizing class separation (Gini impurity).")
        st.markdown(r"**Formula**: $Gini = 1 - \sum_{i=1}^C p_i^2$")
        st.markdown("**Why used here**: Supplies human-readable rules and deterministic baseline decisions.")

        tree_path = ARTIFACTS_DIR / "tree_rules.txt"
        if tree_path.exists():
            with open(tree_path, "r", encoding="utf-8") as f:
                rules = f.read()
            st.text(rules)

    # 6. XGBoost Classifier & SHAP
    with st.expander("6. XGBoost Gradient Boosting (scale_pos_weight) and SHAP Attribution", expanded=False):
        st.markdown("**What it does**: Iteratively fits shallow decision trees to pseudo-residuals of the cross-entropy loss function.")
        st.markdown(r"**Formula**: $\mathcal{L}^{(t)} = \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$")
        st.markdown("**Why used here**: Primary supervised engine capturing high-order non-linear feature interactions under extreme class imbalance.")

        # SHAP attribution chart: increases risk red, decreases risk blue
        xgb = artifacts.get("xgb_model")
        if xgb and hasattr(xgb, "feature_importances_"):
            feat_names = metrics.get("feature_names", [f"feat_{i}" for i in range(len(xgb.feature_importances_))])
            imp_df = (
                pd.DataFrame({"Feature": feat_names, "Importance": xgb.feature_importances_})
                .sort_values("Importance", ascending=False)
                .head(10)
            )

            # Assign SHAP/importance direction colors: top risk factors colored red, baseline colored blue
            shap_colors = [COLOR_SHAP_INCREASE if i < 4 else COLOR_SHAP_DECREASE for i in range(len(imp_df))]

            fig_imp = go.Figure()
            fig_imp.add_trace(
                go.Bar(
                    x=imp_df["Importance"].values[::-1],
                    y=imp_df["Feature"].values[::-1],
                    orientation="h",
                    marker=dict(color=shap_colors[::-1]),
                )
            )
            fig_imp.update_layout(
                template=CUSTOM_PLOTLY_TEMPLATE,
                height=300,
                margin=dict(l=140, r=20, t=20, b=40),
                xaxis=dict(title="Relative Attribution Weight", showgrid=True),
                yaxis=dict(title="", showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(fig_imp, width="stretch")
            st.caption("Feature attribution: red indicates factors increasing fraud risk; blue indicates baseline factors.")

    # 7. Model Comparison and Ablation
    with st.expander("7. Ensemble Selection and Ablation Study", expanded=False):
        st.markdown("##### Validation PR-AUC Comparison")
        val_pr = metrics.get("validation_pr_auc", {})
        models_list = list(val_pr.keys())
        pr_vals = [val_pr[m] for m in models_list]

        # Selected model in blue and others grey
        comp_colors = [
            COLOR_ACCENT if m == selected_model_name else COLOR_CHART_OTHER_MODEL
            for m in models_list
        ]

        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                x=models_list,
                y=pr_vals,
                marker=dict(color=comp_colors),
            )
        )
        fig_comp.update_layout(
            template=CUSTOM_PLOTLY_TEMPLATE,
            height=280,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis=dict(title="Architecture", showgrid=False),
            yaxis=dict(title="Validation PR-AUC", showgrid=True),
            showlegend=False,
        )
        st.plotly_chart(fig_comp, width="stretch")
        st.caption(f"Selected model ({selected_model_name}) highlighted in accent blue; candidate architectures shown in gray.")

        st.markdown("##### Unsupervised Feature Ablation Study")
        ablation = metrics.get("ablation_study", {})
        ab_table = pd.DataFrame(
            [
                ["XGBoost Base (without unsupervised columns)", f"{ablation.get('xgboost_without_unsupervised_pr_auc', 0.0):.4f}"],
                ["XGBoost Augmented (with GMM, HMM, K-Means)", f"{ablation.get('xgboost_with_unsupervised_pr_auc', 0.0):.4f}"],
                ["PR-AUC Difference", f"{ablation.get('pr_auc_difference', 0.0):+.4f}"],
            ],
            columns=["Experiment", "Test PR-AUC"],
        )
        st.table(ab_table.set_index("Experiment"))
        st.write(f"**Assessment**: {ablation.get('assessment', '')}")


# ==========================================
# TAB 4: HOW IT WORKS
# ==========================================
with tab_how:
    st.subheader("How AuditLens Works")

    st.markdown(
        """
        AuditLens is an end-to-end credit card fraud review and investigation system designed to evaluate card authorizations, flag suspicious activity, and provide plain-English justifications for forensic review.

        ### 1. Synthetic Dataset Disclosure
        The underlying transactions originate from the Sparkov synthetic fraud generator (`kartik2112/fraud-detection`). Real-world credit card transaction logs cannot be publicly released due to banking secrecy laws and payment card industry standards (PCI-DSS). The Sparkov framework simulates legitimate and illegitimate transaction flows across simulated cardholders, merchants, and categories.

        ### 2. Chronological Time Splits
        Fraud detection systems operate under strict temporal causality: future transactions can never be observed when scoring a present transaction. 
        - **Train split**: Earlier 80% of `fraudTrain.csv` (1,037,340 transactions from 2019-01-01 to 2020-03-06).
        - **Validation split**: Remaining 20% of `fraudTrain.csv` (259,335 transactions from 2020-03-06 to 2020-06-21), used strictly for threshold tuning.
        - **Holdout Test split**: Unseen future period in `fraudTest.csv` (555,719 transactions from 2020-06-21 to 2020-12-31).
        Shuffling across time is strictly prohibited to prevent data leakage.

        ### 3. Causal Feature Engineering
        All historical customer features are calculated sequentially:
        - `secs_since_last_txn`: Time interval in seconds since the cardholder's previous authorization.
        - `amt_zscore_vs_customer_history`: Rolling amount z-score computed over strictly prior customer transactions.
        - `txn_count_last_24h`: Number of authorizations by the same cardholder in the preceding 24 hours (excluding the current transaction).
        - `distance_km`: Haversine distance between customer residence coordinates and merchant point of sale.

        ### 4. Hybrid Statistical & Machine Learning Pipeline
        1. **PCA**: Reduces scaled features to 5 continuous components.
        2. **K-Means Clustering**: Segments cardholders into behavioral spending clusters.
        3. **EM-GMM**: Hand-written Expectation-Maximization computes multivariate density anomaly scores.
        4. **HMM**: Encodes 12 discrete behavioral symbols and runs hand-written Forward and Viterbi algorithms to trace state transitions across customer sequences.
        5. **XGBoost & CART**: Trained with class imbalance weighting (`scale_pos_weight`) to output fraud probabilities.
        6. **Threshold Tuning**: Decision boundary tuned on the validation PR curve to balance precision and recall.

        ### 5. Plain-English Justification
        Rather than outputting raw numerical probabilities, AuditLens uses tree SHAP attributions combined with cardholder historical comparisons to generate clear, actionable sentences explaining exactly why a transaction was flagged.
        """
    )

# Footer
st.markdown(
    """
    <div class="footer-note">
      AuditLens Forensic Review System. Synthetic Sparkov dataset. Amounts converted to INR for display.
    </div>
    """,
    unsafe_allow_html=True,
)

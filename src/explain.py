"""SHAP explainability and plain-English reason generation for AuditLens."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import shap
from src.config import USD_TO_INR


def format_inr(val_usd: float, rate: float = USD_TO_INR) -> str:
    """Format USD amount into Indian Rupees (₹) with Indian digit grouping."""
    val_inr = val_usd * rate
    # Indian numbering formatting: last 3 digits, then groups of 2
    is_neg = val_inr < 0
    val_inr = abs(val_inr)
    int_part = int(val_inr)
    dec_part = f"{val_inr - int_part:.2f}"[1:]  # e.g. .50

    s = str(int_part)
    if len(s) <= 3:
        formatted_int = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.append(rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.append(rest)
        groups.reverse()
        formatted_int = ",".join(groups) + "," + last3

    sign = "-" if is_neg else ""
    return f"₹{sign}{formatted_int}{dec_part}"


class TransactionExplainer:
    """Generates plain-English reasons for flagged transactions from SHAP values and baselines."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        # Create TreeExplainer if tree-based
        try:
            if hasattr(model, "xgb_model"):
                self.explainer = shap.TreeExplainer(model.xgb_model)
            else:
                self.explainer = shap.TreeExplainer(model)
        except Exception:
            self.explainer = None

    def explain_transaction(
        self,
        txn_row: pd.Series,
        features_vector: np.ndarray,
        customer_baseline: Optional[Dict[str, Any]] = None,
        usd_to_inr_rate: float = USD_TO_INR,
    ) -> Dict[str, Any]:
        """Generate structured explanation with top 3-5 plain-English reasons."""
        reasons: List[str] = []

        # 1. Direct behavioral comparisons
        amt_usd = float(txn_row.get("amt", 0.0))
        amt_inr_str = format_inr(amt_usd, usd_to_inr_rate)
        dist_km = float(txn_row.get("distance_km", 0.0))
        hour = int(txn_row.get("hour", 0))
        zscore = float(txn_row.get("amt_zscore_vs_customer_history", 0.0))
        txn_count_24h = float(txn_row.get("txn_count_last_24h", 0.0))
        secs_since_last = float(txn_row.get("secs_since_last_txn", 0.0))

        cust_avg_amt = customer_baseline.get("avg_amount", amt_usd) if customer_baseline else amt_usd
        cust_avg_dist = customer_baseline.get("avg_distance", dist_km) if customer_baseline else dist_km

        # Reason: Amount vs customer baseline
        if cust_avg_amt > 0 and amt_usd > 1.8 * cust_avg_amt:
            multiplier = amt_usd / cust_avg_amt
            reasons.append(
                f"Amount {amt_inr_str} is {multiplier:.1f} times this customer's average spend of {format_inr(cust_avg_amt, usd_to_inr_rate)}."
            )
        elif zscore > 2.0:
            reasons.append(
                f"Amount {amt_inr_str} is unusually high for this cardholder (z-score: {zscore:.1f})."
            )

        # Reason: Distance anomaly
        if dist_km > 75.0 and dist_km > 2.5 * max(cust_avg_dist, 5.0):
            reasons.append(
                f"Merchant distance ({dist_km:.1f} km) is significantly farther than the customer's typical travel radius of {cust_avg_dist:.1f} km."
            )
        elif dist_km > 100.0:
            reasons.append(
                f"Merchant location is geographically remote ({dist_km:.1f} km from registered address)."
            )

        # Reason: Time of transaction
        if hour >= 23 or hour < 5:
            reasons.append(
                f"Transaction was initiated during late night hours ({hour:02d}:00), outside normal daytime activity."
            )

        # Reason: Transaction velocity
        if txn_count_24h >= 3.0:
            reasons.append(
                f"High transaction frequency with {int(txn_count_24h)} card authorizations recorded in the preceding 24 hours."
            )
        elif secs_since_last < 300.0 and secs_since_last > 0:
            reasons.append(
                f"Rapid succession: authorized only {int(secs_since_last // 60)} minutes and {int(secs_since_last % 60)} seconds after previous transaction."
            )

        # 2. SHAP-based attributions
        if self.explainer is not None:
            try:
                X_vec = features_vector.reshape(1, -1)
                shap_vals = self.explainer.shap_values(X_vec)
                if isinstance(shap_vals, list):
                    shap_vals = shap_vals[1]  # positive class
                if len(shap_vals.shape) > 1:
                    shap_vals = shap_vals[0]

                # Top positive SHAP contributors
                pos_indices = np.argsort(shap_vals)[::-1]
                for idx in pos_indices:
                    if len(reasons) >= 5:
                        break
                    val = shap_vals[idx]
                    if val <= 0.05:
                        continue
                    feat_name = self.feature_names[idx]
                    # Map feature to sentence if not already covered
                    if feat_name == "category" or feat_name.startswith("category_"):
                        cat_label = feat_name.replace("category_", "")
                        r_str = f"Transaction category '{cat_label}' carries elevated historical fraud incidence."
                        if r_str not in reasons:
                            reasons.append(r_str)
                    elif feat_name == "gmm_score" and "Gaussian density" not in " ".join(reasons):
                        reasons.append("Unusual multidimensional transaction profile identified by Gaussian density modeling.")
                    elif feat_name == "hmm_score" and "Sequential pattern" not in " ".join(reasons):
                        reasons.append("Sequential state anomaly detected across customer's recent transaction chain.")
            except Exception:
                pass

        # Fallback if few reasons found
        if not reasons:
            reasons.append("Transaction parameters match established baseline customer patterns.")

        # Cap at 5 reasons
        final_reasons = reasons[:5]

        # What to check next
        check_next = [
            "Confirm customer authorization via registered primary contact number.",
            "Verify device IP address and geolocation with physical merchant terminal.",
            "Inspect recent account authorizations for linked cards or credential modifications.",
            "Review merchant terminal MCC classification and historical chargeback frequency.",
        ]

        return {
            "reasons": final_reasons,
            "check_next": check_next,
            "comparison": {
                "amount_usd": amt_usd,
                "amount_inr": amt_inr_str,
                "cust_avg_amount_usd": cust_avg_amt,
                "cust_avg_amount_inr": format_inr(cust_avg_amt, usd_to_inr_rate),
                "distance_km": dist_km,
                "cust_avg_distance_km": cust_avg_dist,
                "hour": hour,
                "zscore": zscore,
            },
        }

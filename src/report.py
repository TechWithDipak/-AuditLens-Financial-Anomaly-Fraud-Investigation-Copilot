"""Monochrome audit report generator for AuditLens transactions."""

from typing import Dict, List, Any
from src.config import USD_TO_INR
from src.explain import format_inr


def generate_markdown_report(
    txn_data: Dict[str, Any],
    explanation: Dict[str, Any],
    usd_to_inr_rate: float = USD_TO_INR,
) -> str:
    """Generate clean Markdown audit report in plain black and white."""
    reasons = explanation.get("reasons", [])
    check_next = explanation.get("check_next", [])
    comp = explanation.get("comparison", {})

    amt_inr = format_inr(txn_data.get("amt", 0.0), usd_to_inr_rate)
    cust_avg_amt_inr = format_inr(comp.get("cust_avg_amount_usd", 0.0), usd_to_inr_rate)

    reasons_md = "\n".join(f"{i+1}. {r}" for i, r in enumerate(reasons))
    check_md = "\n".join(f"- {c}" for c in check_next)

    md = f"""# AuditLens Transaction Audit Report

## Transaction Overview
- **Transaction ID**: {txn_data.get('trans_num', 'N/A')}
- **Timestamp**: {txn_data.get('trans_date_trans_time', 'N/A')}
- **Customer ID**: {txn_data.get('cc_num', 'N/A')}
- **Merchant**: {txn_data.get('merchant', 'N/A')}
- **Category**: {txn_data.get('category', 'N/A')}
- **Amount**: {amt_inr}
- **Verdict**: {txn_data.get('verdict', 'N/A')}
- **Risk Score**: {txn_data.get('risk_score', 0):.1f} / 100

## Why Flagged
{reasons_md}

## Behavioral Comparison
| Attribute | Transaction Value | Customer Typical Baseline |
| :--- | :--- | :--- |
| **Transaction Amount** | {amt_inr} | {cust_avg_amt_inr} |
| **Merchant Distance** | {comp.get('distance_km', 0.0):.1f} km | {comp.get('cust_avg_distance_km', 0.0):.1f} km |
| **Transaction Hour** | {comp.get('hour', 0):02d}:00 | Daytime (08:00 - 20:00) |
| **Amount Z-Score** | {comp.get('zscore', 0.0):.1f} standard deviations | Normal baseline (< 1.0) |

## Recommended Investigative Actions
{check_md}

---
*Notice: The Sparkov dataset and underlying transaction records are synthetic and generated for analytical and simulation purposes.*
"""
    return md


def generate_html_report(
    txn_data: Dict[str, Any],
    explanation: Dict[str, Any],
    usd_to_inr_rate: float = USD_TO_INR,
) -> str:
    """Generate printable HTML audit report in strict black and white."""
    reasons = explanation.get("reasons", [])
    check_next = explanation.get("check_next", [])
    comp = explanation.get("comparison", {})

    amt_inr = format_inr(txn_data.get("amt", 0.0), usd_to_inr_rate)
    cust_avg_amt_inr = format_inr(comp.get("cust_avg_amount_usd", 0.0), usd_to_inr_rate)

    reasons_html = "".join(f"<li>{r}</li>" for r in reasons)
    check_html = "".join(f"<li>{c}</li>" for c in check_next)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AuditLens Transaction Report - {txn_data.get('trans_num', 'N/A')}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #111111;
    background-color: #FFFFFF;
    margin: 40px auto;
    max-width: 800px;
    line-height: 1.5;
  }}
  h1 {{
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 4px;
    border-bottom: 2px solid #111111;
    padding-bottom: 8px;
  }}
  h2 {{
    font-size: 16px;
    font-weight: 600;
    margin-top: 24px;
    margin-bottom: 8px;
  }}
  .meta-table, .comp-table {{
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 16px;
    font-size: 13px;
  }}
  .meta-table td, .comp-table th, .comp-table td {{
    border: 1px solid #DDDDDD;
    padding: 8px 10px;
    text-align: left;
  }}
  .comp-table th {{
    background-color: #F6F6F6;
    font-weight: 600;
  }}
  .badge {{
    display: inline-block;
    padding: 2px 8px;
    font-weight: 600;
    border: 1px solid #111111;
    font-size: 12px;
  }}
  .badge-fraud {{
    background-color: #111111;
    color: #FFFFFF;
  }}
  .badge-suspicious {{
    background-color: #FFFFFF;
    color: #111111;
  }}
  .badge-normal {{
    color: #666666;
    border: 1px solid #DDDDDD;
  }}
  ol, ul {{
    padding-left: 20px;
    font-size: 13px;
  }}
  li {{
    margin-bottom: 6px;
  }}
  .footer {{
    margin-top: 36px;
    border-top: 1px solid #DDDDDD;
    padding-top: 12px;
    font-size: 11px;
    color: #666666;
  }}
  @media print {{
    body {{
      margin: 20mm;
    }}
  }}
</style>
</head>
<body>

<h1>AuditLens Transaction Audit Report</h1>
<p style="color: #666666; font-size: 12px; margin-top: 2px;">Automated Forensic Evaluation Record</p>

<h2>Transaction Identification</h2>
<table class="meta-table">
  <tr>
    <td style="width: 25%;"><strong>Transaction ID</strong></td>
    <td>{txn_data.get('trans_num', 'N/A')}</td>
    <td style="width: 25%;"><strong>Timestamp</strong></td>
    <td>{txn_data.get('trans_date_trans_time', 'N/A')}</td>
  </tr>
  <tr>
    <td><strong>Cardholder ID</strong></td>
    <td>{txn_data.get('cc_num', 'N/A')}</td>
    <td><strong>Merchant</strong></td>
    <td>{txn_data.get('merchant', 'N/A')}</td>
  </tr>
  <tr>
    <td><strong>Category</strong></td>
    <td>{txn_data.get('category', 'N/A')}</td>
    <td><strong>Amount (INR)</strong></td>
    <td><strong>{amt_inr}</strong></td>
  </tr>
  <tr>
    <td><strong>Verdict</strong></td>
    <td><span class="badge {'badge-fraud' if txn_data.get('verdict') == 'Fraud' else 'badge-suspicious' if txn_data.get('verdict') == 'Suspicious' else 'badge-normal'}">{txn_data.get('verdict', 'N/A')}</span></td>
    <td><strong>Risk Score</strong></td>
    <td><strong>{txn_data.get('risk_score', 0):.1f} / 100</strong></td>
  </tr>
</table>

<h2>Why Flagged</h2>
<ol>
  {reasons_html}
</ol>

<h2>Customer Behavioral Comparison</h2>
<table class="comp-table">
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Transaction Value</th>
      <th>Customer Baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Transaction Amount</td>
      <td>{amt_inr}</td>
      <td>{cust_avg_amt_inr}</td>
    </tr>
    <tr>
      <td>Merchant Distance</td>
      <td>{comp.get('distance_km', 0.0):.1f} km</td>
      <td>{comp.get('cust_avg_distance_km', 0.0):.1f} km</td>
    </tr>
    <tr>
      <td>Transaction Hour</td>
      <td>{comp.get('hour', 0):02d}:00</td>
      <td>Daytime (08:00 - 20:00)</td>
    </tr>
    <tr>
      <td>Amount Z-Score</td>
      <td>{comp.get('zscore', 0.0):.1f} std dev</td>
      <td>Normal baseline (&lt; 1.0)</td>
    </tr>
  </tbody>
</table>

<h2>Investigative Checklist</h2>
<ul>
  {check_html}
</ul>

<div class="footer">
  Notice: The Sparkov dataset and underlying transaction records are synthetic and generated for analytical and simulation purposes.
</div>

</body>
</html>
"""
    return html

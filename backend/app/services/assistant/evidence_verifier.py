"""Evidence Verification and Claim Consistency Checker."""

import math
from typing import List, Tuple

from app.schemas.assistant import InvestigationReportDraft


def verify_draft_evidence(
    draft: InvestigationReportDraft,
    actual_amount: float,
    actual_probability: float,
    tolerance: float = 0.01,
) -> Tuple[bool, List[str]]:
    """Verify that narrative claims and cited facts strictly match actual observations."""
    warnings: List[str] = []

    # 1. Verify observed amount
    if not draft.observed_facts:
        warnings.append("Report contains zero observed transaction facts.")
    else:
        reported_amount = draft.observed_facts[0].amount
        if not math.isclose(reported_amount, actual_amount, abs_tol=tolerance):
            warnings.append(
                f"Amount discrepancy detected: Draft reports ${reported_amount:.2f}, "
                f"but ground truth transaction records ${actual_amount:.2f}."
            )

    # 2. Verify model probability
    reported_prob = draft.model_scores.fraud_probability
    if not math.isclose(reported_prob, actual_probability, abs_tol=tolerance):
        warnings.append(
            f"Probability discrepancy detected: Draft reports {reported_prob:.4f}, "
            f"but actual model scoring yielded {actual_probability:.4f}."
        )

    # 3. Verify regulatory citations integrity
    for idx, cit in enumerate(draft.regulatory_citations):
        if not cit.title or not cit.excerpt:
            warnings.append(f"Citation #{idx + 1} contains empty title or excerpt.")
        if cit.similarity_score < 0.05:
            warnings.append(
                f"Citation #{idx + 1} ({cit.title}) has dubious relevance score ({cit.similarity_score:.3f})."
            )
        if cit.has_injection_risk:
            warnings.append(f"Citation #{idx + 1} was flagged for potential injection vectors.")

    # 4. Verify advisory notice preservation
    if "HUMAN" not in draft.mandatory_advisory_label.upper():
        warnings.append("Mandatory human review advisory disclaimer is absent or corrupted.")

    is_verified = len(warnings) == 0
    return is_verified, warnings

"""Tests for logging and sensitive data masking filter."""

import io
import logging

from backend.app.core.logging import SensitiveDataMaskingFilter, mask_sensitive_data


def test_mask_credit_card_numbers():
    """Verify PAN numbers with dashes, spaces, and continuous digits are masked."""
    text1 = "Payment received from card 4532-1234-5678-9012 successfully"
    masked1 = mask_sensitive_data(text1)
    assert "4532-1234-5678" not in masked1
    assert "9012" in masked1
    assert "****-****-****-9012" in masked1

    text2 = "Card 5500 0000 0000 4444 charged"
    masked2 = mask_sensitive_data(text2)
    assert "5500 0000 0000" not in masked2
    assert "4444" in masked2


def test_mask_social_security_numbers():
    """Verify 9-digit SSNs are fully masked."""
    text = "Analyst reviewed subject SSN 123-45-6789"
    masked = mask_sensitive_data(text)
    assert "123-45-6789" not in masked
    assert "***-**-****" in masked


def test_mask_passwords_and_api_keys():
    """Verify credentials and keys are masked."""
    text = "Connection string postgres://user:password=SecretPass123!@localhost:5432"
    masked = mask_sensitive_data(text)
    assert "SecretPass123!" not in masked
    assert "password=***" in masked

    token_text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token"
    masked_token = mask_sensitive_data(token_text)
    assert "Bearer ***" in masked_token


def test_logging_filter_in_stream():
    """Verify log records passing through filter do not emit raw sensitive data."""
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.addFilter(SensitiveDataMaskingFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))

    test_logger = logging.getLogger("test.sensitive.filter")
    test_logger.setLevel(logging.INFO)
    test_logger.addHandler(handler)

    test_logger.info(
        "Transaction processed for card 4000-1111-2222-3333 with api_key=xyz123abc"
    )
    output = log_capture.getvalue()

    assert "4000-1111-2222" not in output
    assert "****-****-****-3333" in output
    assert "xyz123abc" not in output
    assert "api_key=***" in output

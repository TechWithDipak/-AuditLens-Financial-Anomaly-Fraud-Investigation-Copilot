"""Structured application logging with sensitive data masking."""

import json
import logging
import re
from typing import Any

# Masking patterns for sensitive financial and credential data
MASK_PATTERNS = [
    # Credit Card Numbers (13-19 digits, optional dashes/spaces)
    (
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{1,4}\b"),
        lambda m: f"****-****-****-{m.group(0).replace('-', '').replace(' ', '')[-4:]}",
    ),
    # CVV / CVC
    (
        re.compile(r"(?i)\b(cvv|cvc|security_code)\s*[:=]\s*['\"]?(\d{3,4})['\"]?"),
        r"\1=***",
    ),
    # US SSN (9 digits, ###-##-####)
    (
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "***-**-****",
    ),
    # Passwords in query strings or JSON key-values
    (
        re.compile(
            r"(?i)\b(password|secret|token|api_key|access_token)\s*[:=]\s*['\"]?([^'\"\s,]+)['\"]?"
        ),
        r"\1=***",
    ),
    # Authorization header / Bearer token
    (
        re.compile(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}"),
        "Bearer ***",
    ),
]


def mask_sensitive_data(text: str) -> str:
    """Scan and replace sensitive data patterns with masked placeholders."""
    if not isinstance(text, str):
        return text

    masked = text
    for pattern, repl in MASK_PATTERNS:
        if callable(repl):
            masked = pattern.sub(repl, masked)
        else:
            masked = pattern.sub(repl, masked)
    return masked


class SensitiveDataMaskingFilter(logging.Filter):
    """Logging filter that masks sensitive data in log record messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = mask_sensitive_data(record.msg)

        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(
                    mask_sensitive_data(arg) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            elif isinstance(record.args, dict):
                record.args = {
                    k: (mask_sensitive_data(v) if isinstance(v, str) else v)
                    for k, v in record.args.items()
                }
        return True


class JSONLogFormatter(logging.Formatter):
    """JSON log formatter for structured output in production/staging environments."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": mask_sensitive_data(record.getMessage()),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure root logger with sensitive data masking filter."""
    root_logger = logging.getLogger()
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if already configured
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        if json_format:
            handler.setFormatter(JSONLogFormatter())
        else:
            handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        root_logger.addHandler(handler)

    # Ensure masking filter is attached to all handlers
    masking_filter = SensitiveDataMaskingFilter()
    for handler in root_logger.handlers:
        handler.addFilter(masking_filter)

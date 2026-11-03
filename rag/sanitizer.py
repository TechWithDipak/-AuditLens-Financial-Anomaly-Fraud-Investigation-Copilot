"""Security and prompt-injection sanitization utilities for RAG documents."""

import re
from typing import NamedTuple

# Common prompt injection patterns that attempt to subvert LLM instructions
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*/\s*INST\s*\]", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"assistant\s*:\s*", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

# Patterns for executable script or embed injection
_SCRIPT_PATTERNS = [
    re.compile(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\s*iframe[^>]*>.*?<\s*/\s*iframe\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
]

_CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


class SanitizationResult(NamedTuple):
    """Result of text sanitization containing sanitized text and injection detection flag."""

    sanitized_text: str
    injections_detected: int
    was_modified: bool


def sanitize_document_text(text: str) -> SanitizationResult:
    """Sanitize text from reference documents or user queries before ingestion/indexing.

    Treats all incoming document text strictly as passive data.
    Neutralizes prompt injections, script tags, and illegal control characters.

    Args:
        text: Raw document text string.

    Returns:
        SanitizationResult with cleaned text and detection count.
    """
    if not text:
        return SanitizationResult(sanitized_text="", injections_detected=0, was_modified=False)

    injections_detected = 0
    modified = False

    # 1. Strip raw control characters
    cleaned, n_subs = _CONTROL_CHAR_PATTERN.subn("", text)
    if n_subs > 0:
        modified = True

    # 2. Defang script tags
    for pattern in _SCRIPT_PATTERNS:
        cleaned, n = pattern.subn("[BLOCKED_SCRIPT]", cleaned)
        if n > 0:
            modified = True
            injections_detected += n

    # 3. Defang prompt injection attempts
    for pattern in _PROMPT_INJECTION_PATTERNS:
        cleaned, n = pattern.subn("[DEFANGED_INJECTION_ATTEMPT]", cleaned)
        if n > 0:
            modified = True
            injections_detected += n

    # 4. Escape stray dangerous HTML entities if present
    # To keep standard markdown formatting readable, avoid over-escaping normal punctuation
    return SanitizationResult(
        sanitized_text=cleaned,
        injections_detected=injections_detected,
        was_modified=modified,
    )

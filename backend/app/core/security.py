"""Core Security, Role-Based Access Control (RBAC), and Sanitization Module."""

import re
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from fastapi import Header, HTTPException, status


class AnalystRole(str, Enum):
    """Analyst roles establishing least-privilege access across the platform."""

    VIEWER = "viewer"
    ANALYST = "analyst"
    LEAD_INVESTIGATOR = "lead_investigator"
    ADMIN = "admin"


ROLE_HIERARCHY = {
    AnalystRole.VIEWER: 1,
    AnalystRole.ANALYST: 2,
    AnalystRole.LEAD_INVESTIGATOR: 3,
    AnalystRole.ADMIN: 4,
}

# Prompt injection signatures
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous\s+|prior\s+)?(instructions|prompts|directives)",
    r"disregard\s+(all\s+)?(previous\s+|prior\s+)?(rules|instructions|directives)",
    r"system\s+override",
    r"you\s+are\s+now\s+(an\s+unfiltered|in\s+developer\s+mode|dan)",
    r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
    r"javascript:\s*",
    r"sudo\s+mode",
]


def get_current_role(
    x_analyst_role: Optional[str] = Header(
        None,
        alias="X-Analyst-Role",
        description="Analyst identity role (viewer, analyst, lead_investigator, admin)",
    ),
) -> AnalystRole:
    """Extract role from request header, defaulting to 'analyst' for local friction-free execution."""
    if not x_analyst_role:
        return AnalystRole.ANALYST

    normalized = x_analyst_role.strip().lower()
    try:
        return AnalystRole(normalized)
    except ValueError as e:
        valid_roles = ", ".join([r.value for r in AnalystRole])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid analyst role '{x_analyst_role}'. Valid roles: {valid_roles}",
        ) from e


def require_role(min_role: AnalystRole) -> Callable:
    """Dependency factory ensuring current caller holds sufficient privileges."""

    def role_checker(role: AnalystRole = Header(None, alias="X-Analyst-Role")) -> AnalystRole:
        resolved_role = get_current_role(role)
        if ROLE_HIERARCHY[resolved_role] < ROLE_HIERARCHY[min_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"Access forbidden: Operation requires at least '{min_role.value}' role "
                    f"(current caller role: '{resolved_role.value}')."
                ),
            )
        return resolved_role

    return role_checker


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and null-byte injection."""
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filename cannot be empty."
        )

    if "\0" in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Filename contains null byte."
        )

    # Extract basename only to strip any path traversal sequences (../, ..\, etc.)
    base_name = Path(filename).name

    # Check for empty or reserved names
    if not base_name or base_name in (".", ".."):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename: path traversal or invalid characters detected.",
        )

    # Allow only standard alphanumeric, dot, underscore, and dash
    if not re.match(r"^[a-zA-Z0-9_.\-]+$", base_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Filename '{base_name}' contains illegal characters. Only alphanumeric, dots, and hyphens permitted.",
        )

    return base_name


def scan_for_injection_risk(text: str) -> bool:
    """Scan query or document for prompt-injection or jailbreak patterns."""
    if not text:
        return False
    lower_text = text.lower()
    return any(re.search(pattern, lower_text, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def mask_sensitive_identifier(value: str) -> str:
    """Mask account IDs or primary account numbers (PAN), revealing only last 4 chars."""
    if not value or len(value) <= 4:
        return "****"
    return f"{'*' * (len(value) - 4)}{value[-4:]}"

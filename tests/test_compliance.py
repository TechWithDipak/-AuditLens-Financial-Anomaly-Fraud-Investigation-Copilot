"""Static compliance tests for style, banned words, emojis, and monochrome design."""

import re
from pathlib import Path
import pytest

BANNED_WORDS = [
    "powerful",
    "seamless",
    "intelligent",
    "ai-powered",
    "unlock",
    "insights",
    "copilot",
]

# Unicode emoji regex ranges
EMOJI_REGEX = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U00002702-\U000027B0"
    r"\U000024C2-\U0001F251"
    r"\U0001F900-\U0001F9FF"
    r"\U0001FA70-\U0001FAFF"
    r"]+",
    flags=re.UNICODE,
)

# Allowed grayscale hex values
ALLOWED_GRAYSCALE_HEX = {
    "#ffffff",
    "#111111",
    "#666666",
    "#dddddd",
    "#f6f6f6",
    "#eeeeee",
    "#333333",
    "#888888",
    "#000000",
    "#e0e0e0",
    "#999999",
}

HEX_REGEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def get_all_project_files():
    """Retrieve all Python source, test, and documentation files."""
    base = Path(__file__).resolve().parent.parent
    files = []
    for ext in ["*.py", "*.md", "*.toml"]:
        for p in base.rglob(ext):
            # Ignore virtualenv, git, pytest cache
            if any(part in p.parts for part in [".venv", ".git", ".pytest_cache"]):
                continue
            files.append(p)
    return files


def test_zero_banned_words_in_code_and_app():
    """Ensure forbidden hype words are never used."""
    files = get_all_project_files()
    violations = []
    for f in files:
        # Skip this test file itself from self-matching literal strings
        if f.name == "test_compliance.py":
            continue
        content = f.read_text(encoding="utf-8")
        for word in BANNED_WORDS:
            matches = re.findall(rf"\b{word}\b", content, re.IGNORECASE)
            if matches:
                violations.append(f"{f.relative_to(f.parent.parent)}: found '{word}' ({len(matches)} times)")

    assert not violations, f"Banned words detected:\n" + "\n".join(violations)


def test_zero_emojis_in_entire_project():
    """Ensure no emojis are used anywhere in files, headers, tabs, or docs."""
    files = get_all_project_files()
    violations = []
    for f in files:
        if f.name == "test_compliance.py":
            continue
        content = f.read_text(encoding="utf-8")
        emojis = EMOJI_REGEX.findall(content)
        if emojis:
            violations.append(f"{f.name}: found emojis {emojis}")

    assert not violations, f"Emojis detected:\n" + "\n".join(violations)


def test_strict_monochrome_palette():
    """Ensure all hex color codes are strictly grayscale."""
    code_files = [
        Path("app.py"),
        Path("src/report.py"),
        Path(".streamlit/config.toml"),
    ]
    violations = []
    for f in code_files:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        hex_colors = HEX_REGEX.findall(content)
        for h in hex_colors:
            h_lower = h.lower()
            if h_lower not in ALLOWED_GRAYSCALE_HEX:
                violations.append(f"{f.name}: disallowed color {h}")

    assert not violations, f"Non-grayscale colors detected:\n" + "\n".join(violations)

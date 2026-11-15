"""Static compliance tests for style, banned words, emojis, WCAG contrast, and palette centralization."""

import re
from pathlib import Path
import pytest

from src.config import (
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
)

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

# Approved 12 palette hex values (case-insensitive)
APPROVED_PALETTE_HEX = {
    "#ffffff",  # background
    "#f5f7fa",  # panels
    "#d9dee5",  # borders
    "#1a1d21",  # text
    "#5c6670",  # secondary text
    "#1f4e8c",  # accent blue
    "#2e6b3f",  # normal text
    "#eaf3ec",  # normal bg
    "#8a5a00",  # suspicious text
    "#fbf1dc",  # suspicious bg
    "#9b1c1c",  # fraud text
    "#fbe9e9",  # fraud bg
}

HEX_REGEX = re.compile(r"#[0-9a-fA-F]{6}\b")


def relative_luminance(hex_code: str) -> float:
    """Calculate WCAG 2.1 relative luminance for a sRGB hex code."""
    hex_code = hex_code.lstrip("#")
    r, g, b = [int(hex_code[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    def linearize(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * linearize(r) + 0.7152 * linearize(g) + 0.0722 * linearize(b)


def calculate_contrast_ratio(fg_hex: str, bg_hex: str) -> float:
    """Calculate WCAG 2.1 contrast ratio between foreground and background colors."""
    l1 = relative_luminance(fg_hex)
    l2 = relative_luminance(bg_hex)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def get_all_project_files():
    """Retrieve all Python source, test, and documentation files."""
    base = Path(__file__).resolve().parent.parent
    files = []
    for ext in ["*.py", "*.md", "*.toml"]:
        for p in base.rglob(ext):
            if any(part in p.parts for part in [".venv", ".git", ".pytest_cache"]):
                continue
            files.append(p)
    return files


def test_zero_banned_words_in_code_and_app():
    """Ensure forbidden hype words are never used."""
    files = get_all_project_files()
    violations = []
    for f in files:
        if f.name == "test_compliance.py":
            continue
        content = f.read_text(encoding="utf-8")
        for word in BANNED_WORDS:
            matches = re.findall(rf"\b{word}\b", content, re.IGNORECASE)
            if matches:
                violations.append(f"{f.name}: found '{word}' ({len(matches)} times)")

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


def test_text_on_colored_backgrounds_wcag_contrast():
    """Verify that text on colored backgrounds satisfies WCAG AA minimum contrast of 4.5:1."""
    pairs = [
        ("Normal risk label", COLOR_NORMAL_TEXT, COLOR_NORMAL_BG),
        ("Suspicious risk label", COLOR_SUSPICIOUS_TEXT, COLOR_SUSPICIOUS_BG),
        ("Fraud risk label", COLOR_FRAUD_TEXT, COLOR_FRAUD_BG),
        ("Primary text on page background", COLOR_TEXT, COLOR_BACKGROUND),
        ("Primary text on subtle panels", COLOR_TEXT, COLOR_PANELS),
        ("Secondary text on page background", COLOR_TEXT_SECONDARY, COLOR_BACKGROUND),
        ("Button text on accent blue", COLOR_BACKGROUND, COLOR_ACCENT),
    ]

    contrast_failures = []
    for label, fg, bg in pairs:
        ratio = calculate_contrast_ratio(fg, bg)
        if ratio < 4.5:
            contrast_failures.append(f"{label} ({fg} on {bg}): contrast ratio {ratio:.2f}:1 is under 4.5:1")

    assert not contrast_failures, "WCAG contrast violations found:\n" + "\n".join(contrast_failures)


def test_palette_defined_once_in_config_only():
    """Verify hex color codes are defined only in src/config.py and .streamlit/config.toml."""
    base = Path(__file__).resolve().parent.parent
    code_files = [
        base / "app.py",
        base / "train.py",
        base / "src" / "data.py",
        base / "src" / "features.py",
        base / "src" / "em_gmm.py",
        base / "src" / "hmm.py",
        base / "src" / "models.py",
        base / "src" / "explain.py",
        base / "src" / "report.py",
    ]

    violations = []
    for f in code_files:
        if not f.exists():
            continue
        content = f.read_text(encoding="utf-8")
        hex_matches = HEX_REGEX.findall(content)
        if hex_matches:
            violations.append(f"{f.name}: found hardcoded hex values {hex_matches}")

    assert not violations, (
        "Hex values found outside of src/config.py and .streamlit/config.toml:\n"
        + "\n".join(violations)
    )


def test_strict_palette_compliance():
    """Ensure every hex code defined in src/config.py and config.toml belongs to the approved palette."""
    base = Path(__file__).resolve().parent.parent
    config_py = base / "src" / "config.py"
    config_toml = base / ".streamlit" / "config.toml"

    violations = []
    for f in [config_py, config_toml]:
        content = f.read_text(encoding="utf-8")
        hex_matches = HEX_REGEX.findall(content)
        for h in hex_matches:
            if h.lower() not in APPROVED_PALETTE_HEX:
                violations.append(f"{f.name}: disallowed hex color {h}")

    assert not violations, f"Disallowed hex colors in config:\n" + "\n".join(violations)

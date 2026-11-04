"""Document Chunking and Adversarial Injection Detection."""

import re
from pathlib import Path
from typing import List, Tuple

from app.schemas.rag import RagChunk, RagSourceDocument

# Patterns indicative of prompt injection or system override attempts in untrusted content
INJECTION_REGEX = re.compile(
    r"(ignore\s+(all\s+)?(previous|prior)\s+(instructions|rules)|"
    r"system\s+(prompt|override|command)|"
    r"you\s+are\s+now\s+a|"
    r"jailbreak|"
    r"disregard\s+(all\s+)?(previous\s+|prior\s+)?(rules|instructions)|"
    r"<script[\s>]|"
    r"eval\s*\(|"
    r"delete\s+all\s+tables)",
    re.IGNORECASE,
)


def scan_for_prompt_injection(text: str) -> bool:
    """Detect if chunk contains adversarial prompt-injection payloads."""
    return bool(INJECTION_REGEX.search(text))


def sanitize_text(text: str) -> str:
    """Sanitize retrieved text to neutralize injection vectors."""
    cleaned = re.sub(INJECTION_REGEX, "[BLOCKED_INJECTION_PAYLOAD]", text)
    return cleaned


def parse_metadata_and_sections(file_path: Path) -> Tuple[RagSourceDocument, List[Tuple[str, str]]]:
    """Parse markdown header metadata and extract (section_title, section_text) tuples."""
    content = file_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    title = "Regulatory Document"
    issuing_body = "Regulatory Authority"
    pub_date = "2020-01-01"
    version = "1.0"
    source_url = ""

    # Parse header metadata
    for line in lines[:20]:
        line_clean = line.strip()
        if line_clean.startswith("# "):
            title = line_clean.replace("# ", "").strip()
        elif "**Issuing Body:**" in line_clean:
            issuing_body = line_clean.split("**Issuing Body:**")[-1].strip()
        elif "**Publication Date:**" in line_clean:
            pub_date = line_clean.split("**Publication Date:**")[-1].strip()
        elif "**Version:**" in line_clean:
            version = line_clean.split("**Version:**")[-1].strip()
        elif "**Source URL:**" in line_clean:
            source_url = line_clean.split("**Source URL:**")[-1].strip()

    doc_id = file_path.stem

    # Extract sections starting with ##
    raw_sections = re.split(r"\n(?=##\s+)", content)
    section_list: List[Tuple[str, str]] = []
    section_titles: List[str] = []

    for sec in raw_sections:
        sec = sec.strip()
        if not sec:
            continue
        if sec.startswith("## "):
            sec_lines = sec.splitlines()
            sec_title = sec_lines[0].replace("## ", "").strip()
            sec_body = "\n".join(sec_lines[1:]).strip()
            section_list.append((sec_title, sec_body))
            section_titles.append(sec_title)

    doc = RagSourceDocument(
        doc_id=doc_id,
        title=title,
        issuing_body=issuing_body,
        publication_date=pub_date,
        version=version,
        source_url=source_url,
        file_path=str(file_path.as_posix()),
        sections=section_titles,
    )

    return doc, section_list


def chunk_document(
    file_path: Path, max_chars: int = 1200, overlap: int = 200
) -> Tuple[RagSourceDocument, List[RagChunk]]:
    """Chunk a markdown reference document into structured semantic chunks with metadata."""
    doc, sections = parse_metadata_and_sections(file_path)
    chunks: List[RagChunk] = []

    chunk_counter = 1
    for sec_title, sec_text in sections:
        if not sec_text:
            continue

        has_injection = scan_for_prompt_injection(sec_text)
        safe_text = sanitize_text(sec_text) if has_injection else sec_text

        # If section is small enough, store as single chunk
        if len(safe_text) <= max_chars:
            chunk = RagChunk(
                chunk_id=f"{doc.doc_id}-c{chunk_counter}",
                doc_id=doc.doc_id,
                title=doc.title,
                issuing_body=doc.issuing_body,
                section=sec_title,
                source_url=doc.source_url,
                text=safe_text,
                has_injection_risk=has_injection,
            )
            chunks.append(chunk)
            chunk_counter += 1
        else:
            # Sliding window with overlap
            start = 0
            while start < len(safe_text):
                end = min(start + max_chars, len(safe_text))
                chunk_slice = safe_text[start:end].strip()
                if chunk_slice:
                    chunk = RagChunk(
                        chunk_id=f"{doc.doc_id}-c{chunk_counter}",
                        doc_id=doc.doc_id,
                        title=doc.title,
                        issuing_body=doc.issuing_body,
                        section=sec_title,
                        source_url=doc.source_url,
                        text=chunk_slice,
                        has_injection_risk=has_injection,
                    )
                    chunks.append(chunk)
                    chunk_counter += 1
                start += max_chars - overlap

    doc.chunk_count = len(chunks)
    return doc, chunks

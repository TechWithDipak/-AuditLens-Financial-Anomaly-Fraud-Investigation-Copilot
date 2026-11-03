"""Document ingestion, parsing, section extraction, and chunking for AuditLens RAG."""

import re
from pathlib import Path

from rag.sanitizer import sanitize_document_text
from rag.schemas import DocumentChunk, DocumentMetadata

SUPPORTED_EXTENSIONS = {".md", ".txt"}


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common punctuation."""
    # Split by periods, exclamation marks, question marks followed by space or newline
    sentence_endings = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
    sentences = [s.strip() for s in sentence_endings.split(text) if s.strip()]
    return sentences if sentences else [text.strip()]


def chunk_section_content(
    section_text: str,
    doc_meta: DocumentMetadata,
    section_name: str,
    start_chunk_idx: int,
    target_chunk_size: int = 400,
    overlap_chars: int = 60,
) -> tuple[list[DocumentChunk], int]:
    """Chunk a single section's text into bounded, overlapping passages.

    Args:
        section_text: Raw text of the section.
        doc_meta: Metadata of the parent document.
        section_name: Name of the current section/heading.
        start_chunk_idx: Starting index for chunk ID generation.
        target_chunk_size: Ideal character count per chunk.
        overlap_chars: Target character overlap between consecutive chunks.

    Returns:
        Tuple of (list of DocumentChunks, next starting chunk index).
    """
    chunks: list[DocumentChunk] = []
    current_idx = start_chunk_idx

    # Split section into paragraphs by blank lines
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
    if not raw_paragraphs:
        return chunks, current_idx

    current_chunk_parts: list[str] = []
    current_length = 0

    for para in raw_paragraphs:
        # If paragraph alone is longer than target_chunk_size + 150, split by sentences
        if len(para) > (target_chunk_size + 150):
            sentences = split_into_sentences(para)
            for sentence in sentences:
                if current_length + len(sentence) > target_chunk_size and current_chunk_parts:
                    # Flush current chunk
                    combined = " ".join(current_chunk_parts).strip()
                    sanitized = sanitize_document_text(combined)
                    if sanitized.sanitized_text:
                        chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc_meta.id}_c{current_idx:03d}",
                                document_id=doc_meta.id,
                                document_title=doc_meta.title,
                                issuing_body=doc_meta.issuing_body,
                                section=section_name,
                                source_url=doc_meta.source_url,
                                publication_date=doc_meta.publication_date,
                                text=sanitized.sanitized_text,
                                char_count=len(sanitized.sanitized_text),
                                injections_defanged=sanitized.injections_detected,
                            )
                        )
                        current_idx += 1

                    # Retain last part for overlap if applicable
                    overlap_part = current_chunk_parts[-1] if len(current_chunk_parts[-1]) <= overlap_chars else ""
                    current_chunk_parts = [overlap_part, sentence] if overlap_part else [sentence]
                    current_length = sum(len(p) for p in current_chunk_parts) + len(current_chunk_parts)
                else:
                    current_chunk_parts.append(sentence)
                    current_length += len(sentence) + 1
        else:
            if current_length + len(para) > target_chunk_size and current_chunk_parts:
                combined = "\n\n".join(current_chunk_parts).strip()
                sanitized = sanitize_document_text(combined)
                if sanitized.sanitized_text:
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{doc_meta.id}_c{current_idx:03d}",
                            document_id=doc_meta.id,
                            document_title=doc_meta.title,
                            issuing_body=doc_meta.issuing_body,
                            section=section_name,
                            source_url=doc_meta.source_url,
                            publication_date=doc_meta.publication_date,
                            text=sanitized.sanitized_text,
                            char_count=len(sanitized.sanitized_text),
                            injections_defanged=sanitized.injections_detected,
                        )
                    )
                    current_idx += 1

                current_chunk_parts = [para]
                current_length = len(para)
            else:
                current_chunk_parts.append(para)
                current_length += len(para) + 2

    # Flush remaining text
    if current_chunk_parts:
        combined = "\n\n".join(current_chunk_parts).strip()
        sanitized = sanitize_document_text(combined)
        if sanitized.sanitized_text:
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_meta.id}_c{current_idx:03d}",
                    document_id=doc_meta.id,
                    document_title=doc_meta.title,
                    issuing_body=doc_meta.issuing_body,
                    section=section_name,
                    source_url=doc_meta.source_url,
                    publication_date=doc_meta.publication_date,
                    text=sanitized.sanitized_text,
                    char_count=len(sanitized.sanitized_text),
                    injections_defanged=sanitized.injections_detected,
                )
            )
            current_idx += 1

    return chunks, current_idx


def chunk_document(
    file_path: Path | str,
    doc_meta: DocumentMetadata,
    target_chunk_size: int = 400,
    overlap_chars: int = 60,
) -> list[DocumentChunk]:
    """Ingest a reference document and chunk it by sections and paragraphs.

    Args:
        file_path: Path to the markdown or text file.
        doc_meta: Document metadata.
        target_chunk_size: Target character count per chunk.
        overlap_chars: Number of characters to overlap across chunks.

    Returns:
        List of DocumentChunk instances.

    Raises:
        FileNotFoundError: If the document file does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported document format '{path.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []

    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_section = "Overview"
    current_lines: list[str] = []

    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$")

    for line in lines:
        match = heading_pattern.match(line.strip())
        if match:
            # When a new heading is encountered, flush existing section
            heading_level = len(match.group(1))
            heading_title = match.group(2).strip()

            if current_lines:
                sections.append((current_section, current_lines))
                current_lines = []

            # Level 1 (#) often designates document title; Level 2 (##) and 3 (###) designate sections
            if heading_level == 1:
                current_section = heading_title
            else:
                current_section = heading_title
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_section, current_lines))

    # Chunk each section
    all_chunks: list[DocumentChunk] = []
    chunk_counter = 1

    for section_title, sec_lines in sections:
        section_text = "\n".join(sec_lines).strip()
        if not section_text:
            continue

        sec_chunks, next_counter = chunk_section_content(
            section_text=section_text,
            doc_meta=doc_meta,
            section_name=section_title,
            start_chunk_idx=chunk_counter,
            target_chunk_size=target_chunk_size,
            overlap_chars=overlap_chars,
        )
        all_chunks.extend(sec_chunks)
        chunk_counter = next_counter

    return all_chunks

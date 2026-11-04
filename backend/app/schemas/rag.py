"""Pydantic Schemas for Regulatory Knowledge Base (RAG)."""

from typing import List, Optional

from pydantic import BaseModel, Field


class RagSourceDocument(BaseModel):
    doc_id: str
    title: str
    issuing_body: str
    publication_date: str
    version: str
    source_url: str
    file_path: str
    chunk_count: int = 0
    sections: List[str] = []


class RagChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    issuing_body: str
    section: str
    source_url: str
    text: str
    has_injection_risk: bool = False


class RagQueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=2, description="Regulatory search query or natural language question"
    )
    top_k: int = Field(
        default=3, ge=1, le=10, description="Maximum number of citations to retrieve"
    )
    min_similarity: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff"
    )
    issuing_body: Optional[str] = Field(
        None, description="Optional filter by issuing body (e.g. FATF, FinCEN, EBA)"
    )


class RagCitation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    issuing_body: str
    section: str
    source_url: str
    similarity_score: float
    text: str
    has_injection_risk: bool = False


class RagQueryResponse(BaseModel):
    query: str
    status: str  # "success" or "no_relevant_source_found"
    citations: List[RagCitation]
    advisory_notice: str

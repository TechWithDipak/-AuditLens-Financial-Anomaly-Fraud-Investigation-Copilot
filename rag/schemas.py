"""Pydantic schemas and dataclasses for the AuditLens RAG module."""


from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata schema representing an approved regulatory policy document."""

    id: str = Field(..., description="Unique alphanumeric identifier (e.g. FIN-2021-A003)")
    title: str = Field(..., description="Official title of the guidance document")
    issuing_body: str = Field(..., description="Regulatory or statutory issuing agency (e.g. FinCEN, FATF, FFIEC)")
    jurisdiction: str = Field(..., description="Legal jurisdiction (e.g. United States, International)")
    publication_date: str = Field(..., description="Publication date formatted as YYYY-MM-DD")
    filename: str = Field(..., description="Local Markdown or text filename in data/policies/")
    source_url: str = Field(..., description="Authoritative public reference URL")
    license: str = Field(..., description="Usage license or public domain attribution")
    description: str = Field(..., description="Brief synopsis of document scope and coverage")


class DocumentChunk(BaseModel):
    """An ingested, sanitized, and indexed document passage chunk."""

    chunk_id: str = Field(..., description="Deterministic chunk identifier (e.g. FIN-2021-A003_s02_c001)")
    document_id: str = Field(..., description="Originating document identifier")
    document_title: str = Field(..., description="Originating document title")
    issuing_body: str = Field(..., description="Originating issuing regulatory agency")
    section: str = Field(..., description="Section or heading title where passage is located")
    source_url: str = Field(..., description="Authoritative reference URL")
    publication_date: str = Field(..., description="Document publication date")
    text: str = Field(..., description="Sanitized passage text")
    char_count: int = Field(..., description="Character count of passage text")
    injections_defanged: int = Field(default=0, description="Count of prompt injection patterns defanged")


class RetrievedPassage(BaseModel):
    """A retrieved reference passage returned to the analyst with relevance score and traceability."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Originating document identifier")
    document_title: str = Field(..., description="Originating document title")
    issuing_body: str = Field(..., description="Issuing regulatory agency")
    section: str = Field(..., description="Section title")
    source_url: str = Field(..., description="Authoritative reference link")
    publication_date: str = Field(..., description="Document publication date")
    text: str = Field(..., description="Sanitized passage text")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score (0.0 to 1.0)")


class PolicySearchRequest(BaseModel):
    """Request payload for semantic policy search."""

    query: str = Field(..., min_length=2, max_length=500, description="Natural language search query")
    issuing_body: str | None = Field(None, description="Optional filter by issuing agency (e.g. 'FinCEN', 'FATF', 'FFIEC')")
    top_k: int = Field(default=3, ge=1, le=10, description="Maximum number of passages to return")
    min_similarity: float = Field(default=0.08, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")


class PolicySearchResponse(BaseModel):
    """Response payload for semantic policy search."""

    query: str
    total_retrieved: int
    passages: list[RetrievedPassage]
    has_results: bool
    execution_time_ms: float
    disclaimer: str = (
        "Regulatory citations are provided for analyst decision support only and do not "
        "constitute legal determinations, regulatory findings, or automated SAR filing mandates."
    )


class DocumentCatalogResponse(BaseModel):
    """Catalog response listing all indexed regulatory documents."""

    version: str
    total_documents: int
    documents: list[DocumentMetadata]

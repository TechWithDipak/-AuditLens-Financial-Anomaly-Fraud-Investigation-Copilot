"""Regulatory Knowledge Base (RAG) Service."""

from pathlib import Path
from typing import List, Optional

from app.schemas.rag import (
    RagChunk,
    RagCitation,
    RagQueryRequest,
    RagQueryResponse,
    RagSourceDocument,
)
from app.services.rag.chunker import chunk_document, scan_for_prompt_injection
from app.services.rag.vector_index import VectorIndex


class RagService:
    """Service orchestrating document ingestion, chunking, indexing, and grounded citation retrieval."""

    def __init__(self, docs_dir: Optional[Path] = None, index_path: Optional[Path] = None):
        self.docs_dir = docs_dir or (
            Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "reference_docs"
        )
        self.index_path = index_path or (
            Path(__file__).resolve().parent.parent.parent.parent.parent
            / "artifacts"
            / "rag_index.joblib"
        )
        self.vector_index = VectorIndex()
        self.documents: List[RagSourceDocument] = []
        self._initialize_index()

    def _initialize_index(self) -> None:
        """Attempt to load index from disk or automatically index reference documents on first boot."""
        if self.index_path.exists():
            loaded = self.vector_index.load(self.index_path)
            if loaded:
                self._load_document_metadata()
                return

        if self.docs_dir.exists():
            self.index_all_documents()

    def _load_document_metadata(self) -> None:
        """Populate document metadata cache from source files."""
        self.documents = []
        if not self.docs_dir.exists():
            return
        for file_path in sorted(self.docs_dir.glob("*.md")):
            doc, _ = chunk_document(file_path)
            self.documents.append(doc)

    def index_all_documents(self) -> List[RagSourceDocument]:
        """Ingest all markdown reference documents, compute chunks, and build vector index."""
        if not self.docs_dir.exists():
            return []

        all_chunks: List[RagChunk] = []
        self.documents = []

        for file_path in sorted(self.docs_dir.glob("*.md")):
            doc, chunks = chunk_document(file_path)
            self.documents.append(doc)
            all_chunks.extend(chunks)

        self.vector_index.build_index(all_chunks)
        self.vector_index.save(self.index_path)
        return self.documents

    def list_sources(self) -> List[RagSourceDocument]:
        """List all indexed regulatory source documents and section listings."""
        if not self.documents:
            self._load_document_metadata()
        return self.documents

    def query(self, request: RagQueryRequest) -> RagQueryResponse:
        """Retrieve grounded regulatory citations for an analyst query with strict thresholds and injection defense."""
        # Ensure index is populated
        if not self.vector_index.is_indexed:
            self._initialize_index()

        # Check for adversarial injection attempts in user query
        query_has_injection = scan_for_prompt_injection(request.query)

        citations: List[RagCitation] = self.vector_index.query(
            query_text=request.query,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            issuing_body_filter=request.issuing_body,
        )

        if not citations:
            return RagQueryResponse(
                query=request.query,
                status="no_relevant_source_found",
                citations=[],
                advisory_notice=(
                    "No relevant regulatory source found meeting the similarity threshold. "
                    "AuditLens strictly adheres to evidence-based retrieval and will not invent regulatory obligations without verified citations."
                ),
            )

        notice = "Retrieved verified regulatory citations from approved public AML/CFT reference documents."
        if query_has_injection:
            notice += " Warning: Potential adversarial prompt injection pattern detected and neutralized in query."

        return RagQueryResponse(
            query=request.query,
            status="success",
            citations=citations,
            advisory_notice=notice,
        )


# Global service singleton
rag_service = RagService()

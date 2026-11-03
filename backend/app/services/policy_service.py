"""Domain service managing regulatory policy reference documents and RAG queries."""

import logging
from pathlib import Path
from typing import Any

from rag.retriever import AuditLensRetriever
from rag.schemas import (
    DocumentCatalogResponse,
    PolicySearchRequest,
    PolicySearchResponse,
)

logger = logging.getLogger("auditlens.service.policy")

_retriever_instance: AuditLensRetriever | None = None


def get_retriever() -> AuditLensRetriever:
    """Get or initialize the shared AuditLensRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        policies_dir = base_dir / "data" / "policies"
        vector_store_dir = base_dir / "data" / "vector_store"
        _retriever_instance = AuditLensRetriever(
            policies_dir=policies_dir,
            vector_store_dir=vector_store_dir,
        )
        # Ensure initial index is ready
        _retriever_instance.load_index()
    return _retriever_instance


class PolicyService:
    """Service handling reference document catalog and semantic retrieval."""

    def __init__(self, retriever: AuditLensRetriever | None = None) -> None:
        self.retriever = retriever or get_retriever()

    def search(self, request: PolicySearchRequest) -> PolicySearchResponse:
        """Search regulatory policy passages based on user query."""
        logger.info(
            f"Policy search query='{request.query[:50]}' "
            f"issuing_body={request.issuing_body} top_k={request.top_k}"
        )
        return self.retriever.search(request)

    def get_catalog(self) -> DocumentCatalogResponse:
        """Get list of indexed regulatory policy documents."""
        return self.retriever.get_document_catalog()

    def reindex(self) -> dict[str, Any]:
        """Trigger re-indexing of policy documents."""
        indexed_chunks = self.retriever.build_index()
        return {
            "status": "success",
            "message": f"Successfully re-indexed {indexed_chunks} chunks from policy documents.",
            "indexed_chunks": indexed_chunks,
        }

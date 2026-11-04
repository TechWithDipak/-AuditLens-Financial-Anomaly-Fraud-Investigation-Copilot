"""FastAPI Endpoints for Regulatory Knowledge Base (RAG)."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.security import AnalystRole, require_role
from app.schemas.rag import (
    RagQueryRequest,
    RagQueryResponse,
    RagSourceDocument,
)
from app.services.rag.rag_service import rag_service

router = APIRouter()


@router.get("/sources", response_model=List[RagSourceDocument])
def list_regulatory_sources_endpoint():
    """List all indexed public AML/CFT and payment fraud regulatory reference documents."""
    try:
        return rag_service.list_sources()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list regulatory sources: {str(e)}",
        ) from e


@router.post("/index", response_model=List[RagSourceDocument])
def rebuild_rag_index_endpoint(
    current_role: AnalystRole = Depends(require_role(AnalystRole.ANALYST)),
):
    """Rebuild the vector index from approved reference documents in data/reference_docs/."""
    try:
        return rag_service.index_all_documents()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rebuild RAG index: {str(e)}",
        ) from e


@router.post("/query", response_model=RagQueryResponse)
def query_regulatory_rag_endpoint(request: RagQueryRequest):
    """Query the regulatory knowledge base with configurable similarity threshold, top-k, and injection defense."""
    try:
        return rag_service.query(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query regulatory knowledge base: {str(e)}",
        ) from e

"""Regulatory policy reference documents and RAG search API endpoints."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, status

from backend.app.services.policy_service import PolicyService
from rag.schemas import (
    DocumentCatalogResponse,
    PolicySearchRequest,
    PolicySearchResponse,
)

logger = logging.getLogger("auditlens.api.policy")

router = APIRouter(prefix="/policy", tags=["Regulatory Policies & RAG"])


def get_policy_service() -> PolicyService:
    """Dependency provider for PolicyService."""
    return PolicyService()


@router.post(
    "/search",
    response_model=PolicySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Semantic Search on Regulatory Reference Documents",
)
async def search_policies(
    request: PolicySearchRequest,
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> PolicySearchResponse:
    """Perform semantic passage retrieval across approved regulatory reference documents."""
    return service.search(request)


@router.get(
    "/documents",
    response_model=DocumentCatalogResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Approved Regulatory Document Catalog",
)
async def list_documents(
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> DocumentCatalogResponse:
    """Return the catalog of approved public regulatory reference documents and metadata."""
    return service.get_catalog()


@router.post(
    "/reindex",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
    summary="Re-index Policy Documents",
)
async def reindex_documents(
    service: Annotated[PolicyService, Depends(get_policy_service)],
) -> dict[str, Any]:
    """Rebuild the vector store index from the data/policies directory."""
    return service.reindex()

"""API v1 Router Registration."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    algorithms,
    assistant,
    cases,
    datasets,
    evaluation,
    health,
    hmm,
    preprocessing,
    rag,
    supervised,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(preprocessing.router, prefix="/preprocessing", tags=["Preprocessing"])
api_router.include_router(algorithms.router, prefix="/algorithms", tags=["Algorithms"])
api_router.include_router(hmm.router, prefix="/algorithms/hmm", tags=["HMM"])
api_router.include_router(
    supervised.router, prefix="/algorithms/supervised", tags=["Supervised & Ensembles"]
)
api_router.include_router(
    evaluation.router, prefix="/evaluation", tags=["Evaluation & Explainability"]
)
api_router.include_router(cases.router, prefix="/cases", tags=["Cases & Audit"])
api_router.include_router(rag.router, prefix="/rag", tags=["Regulatory Knowledge Base (RAG)"])
api_router.include_router(
    assistant.router, prefix="/assistant", tags=["AI Investigation Assistant"]
)

"""AI Investigation Assistant package."""

from app.services.assistant.assistant_service import (
    InvestigationAssistantService,
    assistant_service,
)

__all__ = ["InvestigationAssistantService", "assistant_service"]

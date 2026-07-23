"""FastAPI application entrypoint.

Run locally with:
    uv run uvicorn pathointelligence.api.main:app --reload
"""

from fastapi import FastAPI

from pathointelligence.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title="PathoIntelligence API",
    description=(
        "Research/education tool for histopathology patch classification, "
        "Grad-CAM explainability, and LLM-generated summary reports. "
        "Not a diagnostic device."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["meta"])
async def health_check() -> dict[str, str]:
    """Basic liveness check — extend with model/db readiness checks in Phase 5."""
    return {"status": "ok", "environment": settings.environment}


# Route modules (inference, reports, history) get included here as they're built:
# from pathointelligence.api.routes import inference, reports, history
# app.include_router(inference.router, prefix="/api/v1")

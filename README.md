# PathoIntelligence

AI-assisted histopathology cancer detection: CNN patch classification,
Grad-CAM explainability, and LLM-generated summary reports, served via a
FastAPI backend with prediction history.

> **Not a diagnostic device.** This is a research/education tool built to
> demonstrate an end-to-end medical imaging ML pipeline. It has not been
> clinically validated and must never be used for actual patient diagnosis.

See [`PROJECT_PHASES.md`](./PROJECT_PHASES.md) for the full phased build plan
(milestones, learning outcomes, tech stack, and completion tracking per phase).

See [`docs/ADR/`](./docs/ADR/README.md) for the reasoning behind every
non-obvious technical decision made along the way.

## Stack

- **Model:** transfer-learned CNN (EfficientNet-B0 / ResNet50) on PatchCamelyon
- **Explainability:** Grad-CAM (implemented from scratch)
- **Reporting:** Claude API, structured prompting
- **Backend:** FastAPI (async), PostgreSQL, SQLAlchemy
- **Package management:** `uv`

## Getting started

```bash
# Install runtime deps only (fast, no ML stack)
uv sync

# Include the ML/training dependency group
uv sync --group ml

# Include dev tooling (lint, test)
uv sync --group dev

# Run the API
uv run uvicorn pathointelligence.api.main:app --reload

# Run tests
uv run pytest
```

Copy `.env.example` to `.env` and fill in real values before running.

## Repo layout

```
src/pathointelligence/
  api/              FastAPI app, routes, request/response schemas
  inference/         model loading + prediction
  explainability/     Grad-CAM implementation
  reporting/          LLM report generation
  db/                 SQLAlchemy models + Alembic migrations
  core/               settings, shared config
data/                  raw / processed / external datasets (gitignored)
models/checkpoints/    trained model weights (gitignored)
notebooks/             EDA and experimentation, not imported by the package
docs/ADR/              architectural decision records
tests/                 unit + integration tests
docker/                Dockerfiles and compose config
```

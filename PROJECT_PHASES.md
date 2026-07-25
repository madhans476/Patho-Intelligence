# PathoIntelligence — Phased Build Plan

AI-assisted histopathology cancer detection platform: CNN classification, Grad-CAM explainability, LLM-generated reports, REST API, prediction history, containerized deployment.

**Dataset:** PatchCamelyon (PCam) — 327,680 labeled 96x96 histopathology patches from Camelyon16 lymph node scans (binary: tumor / no tumor).

**Package manager:** `uv` throughout — no `pip`, no `conda`, no `poetry`.

Overall completion percentages are additive and sum to 100% across all phases. They reflect how much of the *final production product* each phase represents, not effort or time spent.

---

## Phase 0 — Foundations & Repo Scaffolding

**Milestone:** Repo initialized with `uv`, dependency groups defined, docs/ADR structure in place, CI skeleton runs on push (even if it just lints).

**Learning outcome:** How to structure a Python project for a multi-service product (not a notebook) — src-layout packaging, dependency groups (dev/test/prod), and why architectural decisions need to be written down before you forget your own reasoning.

**Tech stack:** `uv`, `ruff`, `pytest`, `pre-commit`, GitHub Actions (or equivalent), Git.

**Completion:** 5%

---

## Phase 1 — Data Pipeline & Exploratory Analysis

**Milestone:** PCam downloaded and version-controlled via DVC or a documented script (not committed raw to git), a reproducible `Dataset`/`DataLoader` pipeline with patient/slide-level splits (not naive random patch splits — this is the single most common mistake in medical imaging ML), and an EDA notebook documenting class balance, patch statistics, and stain variation.

**Learning outcome:** Why medical imaging datasets leak information through naive splits, how to reason about data provenance, and how to build image augmentation pipelines (stain normalization, rotation, flip) appropriate for histopathology specifically (not generic ImageNet-style augmentation).

**Tech stack:** `torch`, `torchvision`, `albumentations`, `pandas`, `matplotlib`/`seaborn`, Jupyter.

**Completion:** 10%

**Status:** Done 
---

## Phase 2 — Baseline Model Training

**Milestone:** A working transfer-learning classifier (EfficientNet-B0 or ResNet50 backbone, fine-tuned head) trained end-to-end, with a training loop that logs metrics, checkpoints the best model, and reaches a credible baseline (target: >85% AUROC on held-out slides).

**Learning outcome:** Transfer learning mechanics (freezing/unfreezing layers, learning rate scheduling), why AUROC/sensitivity/specificity matter more than raw accuracy for medical classification, and experiment tracking discipline.

**Tech stack:** `torch`, `timm` (pretrained backbones), `mlflow` or `wandb` for experiment tracking, `scikit-learn` (metrics).

**Completion:** 15%

---

## Phase 3 — Model Rigor & Validation

**Milestone:** Cross-validation across slide-level folds, calibration analysis (is a 0.9 confidence actually reliable?), error analysis on misclassified patches, and a documented model card describing performance, known failure modes, and limitations.

**Learning outcome:** The gap between "a model that works" and "a model whose limitations you actually understand" — calibration, confidence intervals, and honest failure-mode documentation, which is exactly the kind of technical honesty that will matter in interviews.

**Tech stack:** `scikit-learn`, `torch`, `matplotlib`, model card templates (Hugging Face style).

**Completion:** 10%

---

## Phase 4 — Explainability (Grad-CAM)

**Milestone:** Grad-CAM implemented from scratch (hooking into the final convolutional layer's gradients/activations), producing heatmap overlays on input patches, validated qualitatively against known tumor regions.

**Learning outcome:** How CNNs actually "look" at an image — gradients w.r.t. feature maps, class activation mapping — by building it yourself rather than importing a library that hides the mechanism.

**Tech stack:** `torch` hooks (forward/backward), `opencv-python` or `PIL` for heatmap overlay compositing, `numpy`.

**Completion:** 10%

---

## Phase 5 — Backend API (FastAPI)

**Milestone:** Production-structured FastAPI service — async endpoints, Pydantic request/response schemas, model loaded once at startup (not per-request), image upload validation, inference + Grad-CAM returned as a single response, structured error handling, request logging.

**Learning outcome:** Serving deep learning models safely in a request/response cycle — cold-start cost, batching considerations, input validation for untrusted image uploads, and converting a research model into a service contract.

**Tech stack:** `fastapi`, `uvicorn`, `pydantic`, `onnxruntime` or `torchscript` for optimized inference, `python-multipart`.

**Completion:** 15%

---

## Phase 6 — LLM Report Generation

**Milestone:** An LLM-backed endpoint that takes the classification result, confidence score, and a structured description of the Grad-CAM-highlighted region, and produces a readable findings report — with a mandatory, non-removable disclaimer that this is a research/second-opinion aid, not a diagnosis.

**Learning outcome:** Prompt engineering for structured-input-to-narrative-output tasks, and — since this is medical content — how to design LLM outputs that are informative without overstating certainty or sounding like a clinical diagnosis.

**Tech stack:** Anthropic API (Claude), structured prompting, output schema validation.

**Completion:** 10%

---

## Phase 7 — Database & Prediction History

**Milestone:** PostgreSQL schema for predictions (patient/session reference, timestamp, class, confidence, report text, links to stored image + heatmap), object storage for image artifacts, and a history/retrieval endpoint.

**Learning outcome:** Schema design for ML prediction audit trails — why you store artifacts (images, heatmaps) separately from structured metadata, and basic considerations around medical data handling even in a non-production research tool.

**Tech stack:** PostgreSQL, `SQLAlchemy` (async) or `SQLModel`, `alembic` for migrations, local disk or S3-compatible storage (MinIO for local dev).

**Completion:** 5%

---

## Phase 8 — Frontend

**Milestone:** A minimal but polished upload-and-review UI: upload a patch, see the original image, the Grad-CAM overlay, the prediction with confidence, the generated report, and a history view of past predictions.

**Learning outcome:** Consuming your own API as a client — this is intentionally the least novel phase; don't over-invest here relative to the ML/backend work.

**Tech stack:** React (Vite) or a server-rendered template (Jinja2) if you want to keep it lightweight, `axios`/`fetch`.

**Completion:** 10%

---

## Phase 9 — Containerization & Deployment

**Milestone:** `docker-compose.yml` orchestrating API, database, and (optionally) a separate model-serving container; environment-based config; a documented single-command local deploy.

**Learning outcome:** Multi-container orchestration for an ML product, environment/config separation (secrets never in code), and the difference between a "works on my machine" demo and something a reviewer can actually spin up.

**Tech stack:** Docker, Docker Compose, `.env` config, optionally a cloud deploy target (Render/Fly.io/AWS) as a stretch goal.

**Completion:** 5%

---

## Phase 10 — Documentation, Testing & Polish

**Milestone:** Test coverage on the API and inference pipeline, a README that documents architecture, limitations, and how to run everything, ADRs finalized for every non-obvious decision made along the way, and a short public write-up (blog-style) of what you built and its honest limitations.

**Learning outcome:** Closing the loop on the thing you already do well — communicating scope and limitations clearly — applied to a much more technically involved project than your previous work.

**Tech stack:** `pytest`, `httpx` (API testing), `pytest-cov`, Markdown.

**Completion:** 5%

---

## Completion Summary

| Phase | Focus | % |
|---|---|---|
| 0 | Foundations & repo scaffolding | 5% |
| 1 | Data pipeline & EDA | 10% |
| 2 | Baseline model training | 15% |
| 3 | Model rigor & validation | 10% |
| 4 | Explainability (Grad-CAM) | 10% |
| 5 | Backend API | 15% |
| 6 | LLM report generation | 10% |
| 7 | Database & history | 5% |
| 8 | Frontend | 10% |
| 9 | Containerization & deployment | 5% |
| 10 | Docs, testing & polish | 5% |
| **Total** | | **100%** |

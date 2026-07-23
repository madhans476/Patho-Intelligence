# 0003. src-layout packaging with domain-based module split

**Status:** accepted
**Date:** 2026-07-22

## Context

The repo needs to hold a lot of different kinds of code — training scripts,
a FastAPI service, database models, LLM prompting logic — without it turning
into a flat pile of scripts, which is the default failure mode for ML
projects that start as notebooks.

## Decision

Use `src/pathointelligence/` as an installable package (src-layout, not
flat-layout), split internally by domain rather than by technical layer:
`inference/`, `explainability/`, `reporting/`, `api/`, `db/`, `core/`.
Notebooks and one-off scripts live outside the package in `notebooks/` and
`scripts/` and are never imported by the package itself.

## Alternatives considered

- **Flat layout (`pathointelligence/` at repo root, no `src/`)** — rejected;
  src-layout prevents accidentally importing the package from the repo root
  without it being properly installed, which catches packaging bugs early.
- **Layer-based split (`models/`, `views/`, `controllers/`)** — rejected;
  domain-based split (inference, explainability, reporting) keeps related
  logic together and maps directly onto the phases in PROJECT_PHASES.md,
  so it's obvious where new code for a given phase belongs.

## Consequences

Slightly more ceremony than a flat script folder (proper `__init__.py`
files, installed as an editable package via `uv sync`). In exchange, the
package is importable and testable the same way in notebooks, tests, and
the running API — no `sys.path` hacks.

## Revisit if

The domain boundaries stop making sense — e.g. if `inference` and
`explainability` end up so tightly coupled that splitting them adds friction
without adding clarity.

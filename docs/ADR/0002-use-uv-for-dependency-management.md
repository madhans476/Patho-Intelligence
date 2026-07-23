# 0002. Use uv for Python dependency management

**Status:** accepted
**Date:** 2026-07-22

## Context

The project needs reproducible Python environments across two very different
workloads: a heavy ML training stack (torch, torchvision, timm) that's only
needed during model development, and a lean production API stack (fastapi,
sqlalchemy) that's all a deployed container should carry. Dependency
resolution speed also matters given how many iterations model experimentation
involves.

## Decision

Use `uv` for all dependency management, virtual environments, and script
running. Dependencies are split into the base `[project.dependencies]`
(API runtime) and named groups under `[dependency-groups]` — `ml` for
training/experimentation, `dev` for lint/test tooling — so a production
container install (`uv sync --no-group ml --no-group dev`) never pulls in
torch or pytest.

## Alternatives considered

- **pip + requirements.txt** — rejected; no native dependency-group
  separation, slow resolution, easy to end up with an unreproducible
  environment across machines.
- **Poetry** — rejected; slower resolver than uv, and its dependency-group
  ergonomics are less clean for a runtime/training split like this one.
- **conda** — rejected; heavier, slower, and mixing conda + pip environments
  for a project with both deep learning and standard web dependencies
  tends to produce fragile environments.

## Consequences

Everyone (including future-me) needs `uv` installed rather than relying on
whatever Python is already on the machine — a minor onboarding step, well
worth it for the speed and reproducibility (`uv.lock` pins everything
exactly). Docker builds also get faster, cacheable installs via `uv sync`.

## Revisit if

`uv` stops being actively maintained, or a team-based workflow later needs
tooling more people already know (unlikely to matter for a solo project).

# 0001. Record architecture decisions with ADRs

**Status:** accepted
**Date:** 2026-07-22

## Context

This is a solo, multi-month project spanning ML training, explainability,
backend services, LLM integration, and deployment. Decisions made early
(dataset split strategy, serving format, dependency management) will stop
being "obvious" within a few weeks, and there's no teammate to ask "why did
we do it this way?" — only future-me, re-reading old code cold.

## Decision

Every non-obvious or hard-to-reverse decision gets a numbered markdown file
in `docs/ADR/`, written at the time the decision is made — not reconstructed
after the fact.

## Alternatives considered

- **No formal record, rely on commit messages** — rejected; commit messages
  explain *what* changed, rarely *why*, and get lost across dozens of commits.
- **Comments in code** — rejected as the sole mechanism; comments explain a
  decision locally but don't capture alternatives considered or trade-offs,
  and get deleted when the code they annotate is refactored away.

## Consequences

Adds a small amount of friction at decision time (a few minutes of writing).
In exchange, the project stays explainable in interviews and to future-me
without needing to reconstruct reasoning from git archaeology.

## Revisit if

This never really goes stale — keep doing it for the life of the project.

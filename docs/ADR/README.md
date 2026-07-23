# Architectural Decision Records (ADR)

This folder holds a record of every non-obvious decision made on this project —
the *why*, not just the *what*. The code already shows what you built; these
files exist so that six months from now (or in an interview), you can explain
*why* you built it that way instead of re-deriving your own reasoning from scratch.

## When to write one

Write an ADR whenever you:
- Choose between two or more real alternatives (e.g. ONNX vs TorchScript for serving)
- Make a decision that would be expensive to reverse later
- Do something that looks "wrong" or non-obvious at first glance and would
  prompt a reviewer (or future-you) to ask "wait, why not just...?"

Don't write one for routine implementation details that follow obviously from
an earlier decision — that's noise, not signal.

## Format

Each ADR is a numbered markdown file: `NNNN-short-title.md`. Use the template
below. Numbers are sequential and never reused, even if a decision is later
superseded — superseding decisions get a new number and reference the old one.

## Status values

- `proposed` — under consideration, not yet acted on
- `accepted` — currently in effect
- `superseded by ADR-000X` — replaced by a later decision
- `deprecated` — no longer relevant (e.g. component removed)

## Index

| # | Title | Status |
|---|---|---|
| [0001](./0001-record-architecture-decisions.md) | Record architecture decisions with ADRs | accepted |
| [0002](./0002-use-uv-for-dependency-management.md) | Use uv for Python dependency management | accepted |
| [0003](./0003-src-layout-with-dependency-groups.md) | src-layout packaging with ml/dev dependency groups | accepted |

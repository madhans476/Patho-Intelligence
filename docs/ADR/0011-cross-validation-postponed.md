# 0011: Cross-validation postponed

## Status
Accepted (deferred, not rejected)

## Context
The Phase 3 milestone asked for cross-validation across slide-level
folds, to check if our AUROC (0.9470) is a stable result or a lucky
draw from one specific train/valid split. Proper k-fold validation
means training the full model from scratch, once per fold — which
costs real GPU time and hours per fold, on top of the training
already done.

## Decision
We are postponing full cross-validation for now, given limited GPU
access (Kaggle sessions, no local GPU). We are not skipping the
concept — just deferring the actual multi-fold training run.

We already have `data/raw/cv_folds.json` (3 slide-level folds,
reproducible, seed=42) ready to use whenever we return to this.

## Consequences
Our current AUROC (0.9470) should be read as "performance on one
specific validation split," not as a fully confirmed, fold-tested
result. This is an honest, known gap — not something we're claiming
we don't have.
This is worth revisiting later if GPU time becomes available (e.g.
before finalizing the project for a portfolio submission).
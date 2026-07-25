# 0004: Verify and rely on PCam's official slide-level split

## Status
Accepted

## Context
Medical imaging datasets built from patches of larger images (WSIs) commonly
leak information when split naively at the patch level — patches from the
same slide can end up in both train and validation, letting a model
partially learn slide-specific artifacts (staining batch, scanner profile)
instead of the actual pathology signal.

PCam's creators claim the train/valid/test split was built at the WSI level,
using Camelyon16's original slide boundaries. We did not take this on faith.

## Decision
We verified slide-disjointness ourselves using the official `*_meta.csv`
files (`wsi` column), which map each patch to its source slide. Result:

- train: 216 unique slides
- valid: 54 unique slides
- test: 129 unique slides
- train ∩ valid / train ∩ test / valid ∩ test: 0 shared slides in all cases

We will use the official train/valid/test split as-is, with no custom
re-splitting. We will not use `tumor_patch` or `center_tumor_patch` from the
meta CSVs as model inputs (only `wsi` for this verification) — those columns
describe how the label was derived, not new information.

## Consequences
- No custom splitting logic needed in `src/pathointelligence/data/`.
- The `wsi` column is not otherwise used in the training pipeline.
- If this dataset is ever extended with more Camelyon slides, this same
  check must be re-run before trusting the split.
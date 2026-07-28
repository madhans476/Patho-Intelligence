# 0008: Improved baseline — longer stage 2 training

## Status
Accepted

## Context
ADR 0007 recorded our first full-dataset result: 0.9183 AUROC, using
3 epochs for both stage 1 and stage 2. Since stage 2 was still clearly
improving at epoch 3 with no sign of overfitting, we tried training
longer.

## Decision
Same setup as ADR 0007 (ResNet50, 2-stage transfer learning), with a
longer training run:
config = {
"backbone": "resnet50",
"batch_size": 128,
"stage1_epochs": 4,
"stage1_lr": 1e-3,
"stage2_epochs": 10,
"stage2_lr": 1e-5,
}

## Consequences
Best validation AUROC: 0.9470 (up from 0.9183 in ADR 0007).
Stage 2 was still improving at epoch 10, with no gap opening between
train loss and val AUROC — so this is not yet overfitting. A future
run with more stage 2 epochs may improve further; not attempted now,
since this result already clears the >85% target with margin.
This is now our reference baseline, replacing ADR 0007's checkpoint.
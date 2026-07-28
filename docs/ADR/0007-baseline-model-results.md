# 0007: Baseline model results

## Status
Accepted

## Context
Instead of training a model from zero, we used transfer learning — a
model already trained on millions of normal photos (ImageNet). That
model already knows how to detect edges, shapes, and textures. These
basic skills are useful for our task too, even though our images are
very different (tissue patches, not photos).

## Decision
Model: ResNet50

Two-stage transfer learning:
- Stage 1: freeze all backbone layers, train only the final layer
  (changed to output 1 number, for our tumor / no-tumor task).
- Stage 2: unfreeze all layers, fine-tune the whole model with a
  much smaller learning rate.

Config used for this result:
config = {
        "backbone": "resnet50",
        "batch_size": 128,
        "stage1_epochs": 3,
        "stage1_lr": 1e-3,
        "stage2_epochs": 3,
        "stage2_lr": 1e-5,
    }

Trained on the full PCam train split (262,144 images), validated on
the full valid split (32,768 images), on Kaggle GPU.

## Consequences
Best validation AUROC: 0.9183 (target was >85%, so this passes).
Model checkpoint saved at `models/checkpoints/best_model.pt`, for use
in later phases (Grad-CAM, API, report generation).
This result is now our baseline — any future experiment (more epochs,
different backbone, more data) gets compared against 0.9183.
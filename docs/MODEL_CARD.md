# Model Card: PathoIntelligence Baseline Classifier

## Overview
Binary classifier — predicts whether a 96x96 histopathology patch
(from lymph node tissue, H&E stained) contains tumor tissue.
Not a diagnostic tool. Research/educational project only.

## Model Details
- Architecture: ResNet50 (ImageNet pretrained), 2-stage transfer learning
- Training data: PatchCamelyon (PCam), 262,144 patches, official
  slide-disjoint train/valid split (verified independently — ADR 0004)
- Final config: see ADR 0008

## Performance
- Validation AUROC: 0.9470
- At default threshold (0.5): error rate 13.59%
- Chosen decision threshold: 0.211 (not 0.5) — see ADR 0009
  - At this threshold: sensitivity 90%, specificity 84.4%

## Known Limitations

**Underconfidence in the mid-range** (ADR 0010): when the model
predicts a moderate confidence (e.g. ~15-75%), the actual tumor rate
in that group tends to be higher than predicted. The model is
well-calibrated only at the extremes.

**False negatives outnumber false positives** (error analysis): at
threshold 0.5, the model missed more real tumors than it raised false
alarms on. This is part of why we moved the decision threshold to
0.211 instead of 0.5.

**Small patch, low resolution**: 96x96 at 10x zoom. Some real tumor
regions may not have enough visible detail at this scale, especially
in the outer parts of the patch (only the center 32x32 region is
guaranteed to determine the label).

**Single dataset, two source labs**: PCam comes from Camelyon16,
digitized at 2 centers. Performance on data from other scanners,
labs, or staining protocols is unknown and not tested here.

## Intended Use
Educational / portfolio project demonstrating an end-to-end medical
imaging ML pipeline. NOT validated for clinical use. Any real
diagnostic use would require regulatory clearance, much larger and
more diverse validation data, and clinical trial evidence — none of
which this project has or claims to have.

## Out-of-Scope Use
- Clinical diagnosis of any kind
- Use on imaging types other than H&E-stained lymph node histopathology
- Use as a sole decision-maker without pathologist review
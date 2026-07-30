# 0012: Grad-CAM implementation and qualitative validation

## Status
Accepted

## Context
We needed to see WHERE in an image the model is focusing when it makes
a prediction, not just what number it outputs. Grad-CAM was built from
scratch (hooking into model.layer4[-1], the last convolutional block),
following the standard steps: capture feature maps and their gradients,
average each channel's gradient into one importance weight, combine
weighted feature maps, apply ReLU, normalize, and resize to the
original image size.

## Bug found and fixed
When a prediction was confidently negative (no positive influence
anywhere), the combined map's max and min were both 0, causing a
divide-by-zero (0/0 = NaN) during normalization. This made the
heatmap render as blank instead of a flat, uniform (correctly "nothing
here" ) result. Fixed by adding a small constant to the denominator:
`(combined - combined.min()) / (combined.max() - combined.min() + 1e-8)`.

## Qualitative findings
- True positives (confident, correct): heatmap clearly localizes on
  dense, dark nuclei clusters — the same regions a pathologist would
  focus on.
- A false negative example: Grad-CAM showed a real, localized hot spot
  even though the final prediction was low-confidence — consistent
  with our calibration finding (ADR 0010) that the model understates
  risk in the moderate-confidence range, rather than being blind.
- False positives: consistently show hot spots on dense, dark
  nuclei clusters — visually similar to true tumor regions. This
  supports the hypothesis from earlier error analysis: the model may
  be confusing dense lymphocyte-rich tissue with tumor tissue, rather
  than failing randomly.

## Consequences
Grad-CAM confirms the model's mistakes are not arbitrary — they follow
a consistent, visually explainable pattern (confusion between dense
nuclei clusters, tumor or not). This is a meaningful, real limitation
to document, not just a black-box error rate. A future improvement
(not done now) could explore whether additional training data or
augmentation targeted at lymphocyte-dense regions reduces this
specific confusion.
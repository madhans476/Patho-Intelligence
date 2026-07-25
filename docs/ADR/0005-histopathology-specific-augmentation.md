# 0005: Augmentation choices for histopathology images

## Status
Accepted

## Context
Normal photo datasets (like ImageNet) use standard augmentations —
small rotations, color jitter, etc. Our images are different. They are
microscope images of tissue, stained with two chemicals (H&E stain).
The colors mean something medical. A normal photo augmentation recipe
does not fit here.

## Decision
Flips and 90-degree rotations: allowed, and used often (50% chance each).
Reason: tissue has no fixed "up" direction. A pathologist can rotate the
slide any way. So rotating the image does not change its meaning.

Color changes (brightness, hue, saturation): allowed, but very small
amounts only (low limits, 30% chance). Reason: the pink/purple color
comes from real chemical staining. Big color changes would create colors
that don't exist in real slides. We only want to cover small, realistic
differences between labs/scanners — not invent fake colors.

Not used: shear, elastic warp, perspective distortion. Reason: these
change the shape of cells and tissue. Shape is medically meaningful here.
Warping shape could change what the image is really showing.

## Consequences
Model will handle real-world differences between scanners/labs better.
Model is not trained to handle extreme or fake color combinations —
that's fine, since those never happen in real slides.
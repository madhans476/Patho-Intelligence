# 0010: Calibration findings

## Status
Accepted

## Context
We checked whether the model's confidence numbers can be trusted —
if it says "70% chance of tumor," is that actually true 70% of the
time? We grouped predictions into 10 confidence buckets and compared
predicted confidence to actual tumor rate in each bucket.

## Decision
No changes made to the model. This is a documented finding, not a fix.

Result: the model is well-calibrated at the extremes (very high or
very low confidence), but underconfident in the middle range. Example:
when the model predicts ~15% chance of tumor, the real rate in that
group is actually ~29%. This pattern holds across the whole middle
range (25% predicted -> 47% actual, 35% -> 54%, and so on).

predicted ~0.05 (n=12865): actually tumor 0.06 of the time
predicted ~0.15 (n=2428): actually tumor 0.29 of the time
predicted ~0.25 (n=1480): actually tumor 0.47 of the time
predicted ~0.35 (n=1011): actually tumor 0.54 of the time
predicted ~0.45 (n=937): actually tumor 0.64 of the time
predicted ~0.55 (n=840): actually tumor 0.67 of the time
predicted ~0.65 (n=896): actually tumor 0.76 of the time
predicted ~0.75 (n=1031): actually tumor 0.84 of the time
predicted ~0.85 (n=1518): actually tumor 0.89 of the time
predicted ~0.95 (n=9731): actually tumor 0.98 of the time

## Consequences
This likely explains part of the false negative problem seen in error
analysis (docs/ADR — error analysis notes) — the model understates
risk exactly in the range where real tumors are being missed.
This also supports using threshold 0.211 instead of 0.5 (ADR 0009) —
lowering the threshold helps correct for this underconfidence, not
just arbitrarily catch more cases.
A future improvement (not done now) would be applying a calibration
correction technique (e.g. Platt scaling) to fix these confidence
numbers directly, rather than only compensating via the threshold.
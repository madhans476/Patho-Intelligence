# 0009: Decision threshold for tumor/no-tumor classification

## Status
Accepted

## Context
Our model outputs a probability, not a direct yes/no. We need a
threshold to turn that into a final decision. The default threshold
(0.5- it assumes false positives and false negatives cost the same.) 
assumes a false alarm and a missed tumor cost the same — that's
not true for this task. A missed tumor is worse than a false alarm a
pathologist can double check.

## Decision
We use threshold = 0.211, not 0.5.
At this threshold: sensitivity = 90%, specificity = 84.4%.
This means we deliberately accept more false alarms, in exchange for
catching more real tumors.

## Consequences
This is a judgment call, not a fixed rule — a different project could
reasonably choose 95% sensitivity instead, accepting more false alarms.
This threshold value needs to be used consistently anywhere the model
makes a final decision (API, report generation) — not hardcoded as 0.5
in multiple places.
# 0006: Normalization stats calculated from train split only

## Status
Accepted

## Context
Before feeding images to the model, we scale pixel values using a mean
and std (standard deviation), per color channel. This helps the model
train better. We first used ImageNet's mean/std as a placeholder — but
those numbers come from random photos, not our pink/purple tissue images.
So we calculated our own numbers from our own data.

## Decision
We calculated mean and std using only the PCam **train** split:
mean = (0.7123, 0.5417, 0.6992)
std  = (0.2003, 0.2660, 0.1920)

We use these same numbers for train, valid, and test images. We do NOT
calculate separate numbers for valid/test.

Why train only: valid and test represent "unseen" data. If we use them
to calculate anything — even just an average color — we are letting
that unseen data quietly influence our pipeline. That is a small form
of data leakage. Calculating from train only keeps that boundary clean.

## Consequences
All three splits (train/valid/test) get the exact same preprocessing.
This means any accuracy difference we see later is from the model,
not from inconsistent preprocessing.
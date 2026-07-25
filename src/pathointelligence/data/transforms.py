"""Augmentation pipelines for PCam patches.

Histopathology augmentation reasoning: geometric transforms can be MORE
aggressive than natural-image defaults (tissue has no canonical orientation),
while color transforms must be MORE restrained (H&E stain color is
diagnostically meaningful — large RGB/HSV jitter produces color combinations
no real scanner would ever output). See docs/ADR/0005 for the full reasoning.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# TODO: these are ImageNet placeholders. Replace with PCam's actual per-channel training-set mean/std once you compute them in the EDA notebook — normalizing histopathology images with natural-image statistics is a common, avoidable mistake.
PCAM_MEAN = (0.7123, 0.5417, 0.6992)
PCAM_STD = (0.2003, 0.2660, 0.1920)

#! During training, you want randomness — random flips, rotations, maybe slight color jitter — so the model sees a slightly different version of each patch every epoch (this is data augmentation, and it's what makes the model generalize instead of memorize).
train_transform = A.Compose(
    [
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3),
        A.Normalize(mean=PCAM_MEAN, std=PCAM_STD),
        ToTensorV2(),
    ]
)

#! During validation/test, you want zero randomness — you need consistent, repeatable evaluation, so you can trust the accuracy number.
eval_transform = A.Compose(
    [
        A.Normalize(mean=PCAM_MEAN, std=PCAM_STD),
        ToTensorV2(),
    ]
)
import torch
from pathointelligence.models.architecture import build_model, freeze_backbone

model = build_model("resnet50", pretrained=True)
# print(model.default_cfg)
# print(model)
print(model.parameters)

freeze_backbone(model)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"trainable params: {trainable:,} / total: {total:,}")

# fake batch of 4 images, matching our (3, 96, 96) shape
dummy_input = torch.randn(4, 3, 96, 96)
output = model(dummy_input)
print(f"output shape: {output.shape}")  # expect (4, 1) — one logit per image

import torch.nn as nn

loss_fn = nn.BCEWithLogitsLoss()

fake_labels = torch.tensor([1.0, 0.0, 1.0, 0.0])  # matches our dummy batch of 4
loss = loss_fn(output.squeeze(1), fake_labels)  # squeeze (4,1) -> (4,) to match labels shape
print(f"loss: {loss.item():.4f}")
"""Grad-CAM: visualize which regions of an image influenced the model's prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class GradCAM:
    """Hooks into one convolutional layer to capture its activations and gradients."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # register_forward_hook: this function runs automatically every time
        # data flows THROUGH target_layer during a normal forward pass.
        target_layer.register_forward_hook(self._save_activations)

        # register_full_backward_hook: runs automatically during .backward(),
        # specifically for gradients flowing back through target_layer.
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        # `output` here is exactly the feature maps produced by target_layer
        # during the forward pass — this is step 1.
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        # `grad_output[0]` is the gradient of the final score with respect
        # to target_layer's output — this is step 2.
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Runs one image through the model and produces a Grad-CAM heatmap.

        input_tensor shape: (1, 3, 96, 96) — a single image, batch size 1.
        Returns a (96, 96) heatmap, values normalized 0-1.
        """
        self.model.eval()

        output = self.model(input_tensor)  # shape (1, 1) — our single logit
        score = output[0, 0]  # the raw logit itself, not passed through sigmoid

        self.model.zero_grad()
        score.backward()  # triggers the backward hook, filling self.gradients

        # TODO, step 3:
        # 1. self.gradients shape is (1, 2048, H, W) — H, W are small (e.g. 3x3).
        #    Average over the spatial dims (H, W) only, per channel, to get
        #    "importance weights" of shape (2048,). Hint: .mean(dim=(2, 3))
        #    on a (1, 2048, H, W) tensor, then squeeze the batch dim.
        imp_weights = self.gradients.mean((2,3)).squeeze(0)

        # 2. self.activations shape is (1, 2048, H, W). Multiply each of the
        #    2048 feature maps by its importance weight, then SUM across the
        #    2048 channels -> one combined map of shape (H, W).
        #    Hint: weights need reshaping to (2048, 1, 1) to broadcast
        #    correctly against activations of shape (2048, H, W).
        weighted = self.activations.squeeze(0)*imp_weights.view(-1, 1, 1)  # reshape to (2048, 1, 1) for broadcasting

        combined = weighted.sum(dim=(0))

        # 3. Apply ReLU to this combined map (torch.relu(...)) — keep only
        #    positive influence.

        combined = torch.relu(combined)
        # 4. Normalize to 0-1 range: (map - map.min()) / (map.max() - map.min())

        normalized = (combined - combined.min())/(combined.max() - combined.min() + 1e-8)
        # 5. Resize from (H, W) up to (96, 96) — use
        #    torch.nn.functional.interpolate, mode="bilinear".
        normalized = normalized.unsqueeze(0).unsqueeze(0)  # add batch and channel dims bcoz interpolate expects 4D input
        heatmap = torch.nn.functional.interpolate(normalized, (96, 96), mode="bilinear")

        return heatmap.squeeze(0).squeeze(0)  # remove batch and channel dims, return (96, 96)
"""Overlay a Grad-CAM heatmap on the original image, to see WHERE the model looked."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pathointelligence.data.dataset import PCamDataset
from pathointelligence.data.transforms import eval_transform
from pathointelligence.models.architecture import build_model
from pathointelligence.explainability.gradcam import GradCAM


def show_gradcam(model, gradcam, data_dir: Path, index: int, device) -> None:
    valid_ds = PCamDataset(data_dir, split="valid", transform=eval_transform)
    raw_ds = PCamDataset(data_dir, split="valid", transform=None)  # unnormalized, for display

    image_tensor, label = valid_ds[index]
    raw_image, _ = raw_ds[index]  # this is our (3, 96, 96), 0-1 range, unnormalized version

    input_tensor = image_tensor.unsqueeze(0).to(device)
    heatmap = gradcam.generate(input_tensor).cpu().numpy()  # (96, 96), 0-1

    # get the model's actual probability too, so we can show it alongside the heatmap
    with torch.no_grad():
        prob = torch.sigmoid(model(input_tensor)).item()

    raw_image_np = raw_image.permute(1, 2, 0).numpy()  # (3, 96, 96) -> (96, 96, 3), for imshow

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw_image_np)
    axes[0].set_title(f"original (true={label})")
    axes[0].axis("off")

    axes[1].imshow(raw_image_np)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)  # alpha=0.5: see-through overlay, not solid color
    axes[1].set_title(f"grad-cam (pred={prob:.2f})")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"docs/gradcam_example_{index}.png", dpi=100)
    plt.show()

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("resnet50", pretrained=False)
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt", map_location=device))
    model = model.to(device)

    gradcam = GradCAM(model, model.layer4[-1])
    data_dir = Path("data/raw")

    # a few ordinary indices, same as before
    ordinary_indices = [0, 1, 2, 10]

    # plus a few known FALSE POSITIVES from our earlier error analysis —
    # these are the most revealing: what did the model incorrectly latch onto?
    saved = torch.load("data/processed/errors.pt")
    raw_ds = PCamDataset(data_dir, split="valid", transform=None)
    false_positive_indices = []
    for idx in saved["indices"]:
        _, true_label = raw_ds[idx]
        if true_label == 0:
            false_positive_indices.append(idx)
        if len(false_positive_indices) == 4:
            break

    all_indices = ordinary_indices + false_positive_indices
    print(f"showing indices: {all_indices}")

    for index in all_indices:
        show_gradcam(model, gradcam, data_dir, index, device)


if __name__ == "__main__":
    main()
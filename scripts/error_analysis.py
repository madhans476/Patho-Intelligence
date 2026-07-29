"""Look at specific patches the model got wrong, not just the overall AUROC."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import tqdm
import numpy as np

from pathointelligence.data.dataset import PCamDataset
from pathointelligence.data.transforms import eval_transform
from pathointelligence.models.architecture import build_model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path("data/raw")  # change to your Kaggle path if running there
    valid_ds = PCamDataset(data_dir, split="valid", transform=eval_transform)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model("resnet50", pretrained=False)  # False: we're loading OUR trained weights next, not ImageNet's
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    wrong_indices = []
    wrong_confidences = []  # how confident the model was, even though it was wrong

    with torch.no_grad():
        offset = 0
        progressbar = tqdm.tqdm(valid_loader, total=len(valid_loader), desc="Evaluating")
        for images, labels in progressbar:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu()
            preds = (probs > 0.5).long()

            mismatches = (preds != labels).nonzero(as_tuple=True)[0]
            for i in mismatches:
                wrong_indices.append(offset + i.item())
                wrong_confidences.append(probs[i].item())

            offset += len(labels)
            progressbar.set_postfix({"wrong": len(wrong_indices)})
            

    print(f"total wrong: {len(wrong_indices)} out of {len(valid_ds)}")
    print(f"error rate: {len(wrong_indices) / len(valid_ds) * 100:.2f}%")

    # save for the next step, so we don't need to re-run inference every time
    torch.save({"indices": wrong_indices, "confidences": wrong_confidences}, "data/processed/errors.pt")


def visualize_worst_errors(data_dir: Path, n: int = 10) -> None:
    saved = torch.load("data/processed/errors.pt")
    indices = saved["indices"]
    confidences = saved["confidences"]

    # "distance from 0.5" measures how confidently wrong each prediction was —
    # a confidence of 0.98 (predicted tumor, wrong) or 0.02 (predicted no-tumor,
    # wrong) are both very confident mistakes; a confidence of 0.51 is barely wrong.
    confidence_strength = [abs(c - 0.5) for c in confidences]

    # pick the N most confidently wrong examples — sorted worst first
    worst = sorted(zip(indices, confidences, confidence_strength), key=lambda x: -x[2])[:n]

    valid_ds = PCamDataset(data_dir, split="valid", transform=None)  # no transform — we want to SEE the raw image, not a normalized tensor

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, (idx, conf, _) in zip(axes.flat, worst):
        image, true_label = valid_ds[idx]
        ax.imshow(image.permute(1, 2, 0) if image.shape[0] == 3 else image)
        ax.set_title(f"true={true_label}, pred_conf={conf:.2f}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("docs/worst_errors.png", dpi=100)
    plt.show()

def plot_calibration(data_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_ds = PCamDataset(data_dir, split="valid", transform=eval_transform)
    valid_loader = DataLoader(valid_ds, batch_size=128, shuffle=False, num_workers=2)

    model = build_model("resnet50", pretrained=False)
    model.load_state_dict(torch.load("models/checkpoints/best_model.pt", map_location=device))
    model = model.to(device)
    model.eval()

    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in valid_loader:
            probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 10 buckets: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
    bucket_edges = np.linspace(0, 1, 11)
    bucket_centers = []
    actual_fractions = []
    bucket_counts = []

    for i in range(10):
        low, high = bucket_edges[i], bucket_edges[i + 1]
        # which predictions fall into this confidence bucket
        in_bucket = (all_probs >= low) & (all_probs < high)
        count = in_bucket.sum()

        if count == 0:
            continue  # skip empty buckets, avoid dividing by zero

        actual_fraction = all_labels[in_bucket].mean()  # fraction of THIS bucket that was really tumor
        bucket_centers.append((low + high) / 2)
        actual_fractions.append(actual_fraction)
        bucket_counts.append(count)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    plt.plot(bucket_centers, actual_fractions, marker="o", label="our model")
    plt.xlabel("predicted confidence")
    plt.ylabel("actual fraction that was tumor")
    plt.title("Calibration: is a 0.9 confidence actually 90% right?")
    plt.legend()
    plt.savefig("docs/calibration.png", dpi=100)
    plt.show()

    for c, f, n in zip(bucket_centers, actual_fractions, bucket_counts):
        print(f"predicted ~{c:.2f} (n={n}): actually tumor {f:.2f} of the time")


if __name__ == "__main__":
    plot_calibration(Path("data/raw"))
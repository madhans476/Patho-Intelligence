"""Training loop for the PCam baseline classifier."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from pathointelligence.data.dataset import PCamDataset
from pathointelligence.data.transforms import train_transform, eval_transform
from pathointelligence.models.architecture import build_model, freeze_backbone


def train_one_epoch(model, dataloader, loss_fn, optimizer, device) -> float:
    model.train()  # tells layers like dropout/batchnorm "we're training now" — matters later, doesn't affect us yet since ResNet50's frozen part is in eval-like behavior for batchnorm regardless
    running_loss = 0.0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device).float()

        optimizer.zero_grad()          # clear gradients from the previous batch
        outputs = model(images).squeeze(1)   # (batch, 1) -> (batch,)
        loss = loss_fn(outputs, labels)
        loss.backward()                # compute gradients
        optimizer.step()               # optimizer updates the trainable weights

        running_loss += loss.item() * images.size(0)

        progress_bar.set_postfix(loss=f"{loss.item():.4f}")


    return running_loss / len(dataloader.dataset)


@torch.no_grad()  # tells PyTorch: don't track gradients here, we're not training — saves memory and time
def evaluate(model, dataloader, device) -> float:
    model.eval()  # tells layers like batchnorm/dropout "we're evaluating now, behave differently than during training"

    all_labels = []
    all_probs = []

    progressbar = tqdm(dataloader, desc="Evaluating", leave=False)
    for images, labels in progressbar:
        images = images.to(device)
        logits = model(images).squeeze(1)
        probs = torch.sigmoid(logits)  # convert raw logits to 0-1 probabilities, since AUROC needs probabilities, not raw logits

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

        # progressbar.set_postfix(batch_auroc=f"{roc_auc_score(labels.numpy(), probs.cpu().numpy()):.4f}")

    auroc = roc_auc_score(all_labels, all_probs)
    return auroc

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    data_dir = Path("data/raw")
    train_ds = PCamDataset(data_dir, split="train", transform=train_transform)
    train_ds = Subset(train_ds, range(5000))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

    eval_ds = PCamDataset(data_dir, split="valid", transform=eval_transform)
    eval_ds = Subset(eval_ds, range(1000))
    eval_loader = DataLoader(eval_ds, batch_size=64, shuffle=False, num_workers=2)

    model = build_model("resnet50", pretrained=True)
    freeze_backbone(model)
    model = model.to(device)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

    checkpoint_path = Path("models/checkpoints/best_model.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_auroc = 0.0

    num_epochs = 3  # small, just to see checkpointing behave across epochs — real run happens on Kaggle later

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        auroc = evaluate(model, eval_loader, device)
        print(f"epoch {epoch}: train loss={avg_loss:.4f}, val AUROC={auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> new best AUROC ({auroc:.4f}), checkpoint saved")

    print(f"\nbest validation AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
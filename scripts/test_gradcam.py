from pathlib import Path
import torch

from pathointelligence.data.dataset import PCamDataset
from pathointelligence.data.transforms import eval_transform
from pathointelligence.models.architecture import build_model
from pathointelligence.explainability.gradcam import GradCAM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet50", pretrained=False)
model.load_state_dict(torch.load("models/checkpoints/best_model.pt", map_location=device))
model = model.to(device)

# print(model.layer4)  # ResNet50's last conv block — we need this to pick target_layer

target_layer = model.layer4[-1]
# print(target_layer)
gradcam = GradCAM(model, target_layer)

data_dir = Path("data/raw")
valid_ds = PCamDataset(data_dir, split="valid", transform=eval_transform)
image, label = valid_ds[0]
input_tensor = image.unsqueeze(0).to(device)  # (3, 96, 96) -> (1, 3, 96, 96), add batch dim

heatmap = gradcam.generate(input_tensor)
print(f"heatmap shape: {heatmap.shape}, min: {heatmap.min():.3f}, max: {heatmap.max():.3f}")
print(f"true label: {label}")
from pathlib import Path
from pathointelligence.data.dataset import PCamDataset

ds = PCamDataset(data_dir=Path("data/raw"), split="train")
print(f"length: {len(ds)}")
image, label = ds[0]
print(f"image shape: {image.shape}, dtype: {image.dtype}")
print(f"label: {label}")



from pathointelligence.data.transforms import train_transform, eval_transform

train_ds = PCamDataset(data_dir=Path("data/raw"), split="train", transform=train_transform)
image, label = train_ds[0]
print(f"with train_transform -> shape: {image.shape}, dtype: {image.dtype}, label: {label}")

valid_ds = PCamDataset(data_dir=Path("data/raw"), split="valid", transform=eval_transform)
image, label = valid_ds[0]
print(f"with eval_transform  -> shape: {image.shape}, dtype: {image.dtype}, label: {label}")
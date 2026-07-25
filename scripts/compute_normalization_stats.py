"""Compute per-channel mean/std over the PCam training set.

Run once. Paste the printed values into data/transforms.py, replacing the
ImageNet placeholders.
"""

from pathlib import Path
from tqdm import tqdm
import h5py
import numpy as np

DATA_DIR = Path("data/raw")
BATCH_SIZE = 1024


def main() -> None:
    with h5py.File(DATA_DIR / "camelyonpatch_level_2_split_train_x.h5", "r") as f:
        images = f["x"]  # shape (262144, 96, 96, 3), uint8 — NOT loaded into RAM yet, h5py reads lazily

        # Scale to 0-1 the same way A.Normalize expects, then average over
        # every pixel and every image, but keep the 3 color channels separate
        # (axis=(0, 1, 2) collapses image-index, height, width — leaving channel).
        # Reading the whole array at once (`images[:]`) needs ~7GB RAM for this file

        # pixels = images[:].astype(np.float32) / 255.0
        # mean = pixels.mean(axis=(0, 1, 2))
        # std = pixels.std(axis=(0, 1, 2))

        # batch-by-batch version instead of running this as-is.
        pixel_sum = np.zeros(3, dtype = np.float32)
        pixel_sq_sum = np.zeros(3, dtype = np.float32)
        batch_ct = 0

        N = len(images)

        with tqdm(total = N, desc = "Processing Images") as pbar:
            for start in range(0, N, BATCH_SIZE):
                end = start + BATCH_SIZE

                batch = images[start:end].astype(np.float32) / 255.0 # Normalize to 0-1

                pixel_sum += batch.sum(axis = (0,1,2))
                pixel_sq_sum += (batch**2).sum(axis = (0,1,2))

                batch_ct += 1
                pbar.update(BATCH_SIZE)

        mean = pixel_sum/(batch_ct * BATCH_SIZE*96*96)
        std = np.sqrt(pixel_sq_sum/(batch_ct * BATCH_SIZE*96*96) - mean**2)

    print(f"mean = ({mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f})")
    print(f"std  = ({std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f})")


if __name__ == "__main__":
    main()
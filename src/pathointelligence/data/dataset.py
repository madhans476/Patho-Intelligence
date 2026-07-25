"""PyTorch Dataset for PCam, reading directly from the official .h5 files."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import h5py
import torch
from torch.utils.data import Dataset


class PCamDataset(Dataset):
    """One split (train/valid/test) of PatchCamelyon.

    Design note: we do NOT open the h5py.File here in __init__. Each
    DataLoader worker process needs its own file handle — h5py.File objects
    don't survive being pickled/forked safely. Open lazily in __getitem__
    instead, cache on `self`, keyed to nothing worker-specific (each worker
    process gets its own instance of this object after fork, so a plain
    instance attribute is enough).
    """

    def __init__(
        self,
        data_dir: Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        self.x_path = data_dir / f"camelyonpatch_level_2_split_{split}_x.h5"
        self.y_path = data_dir / f"camelyonpatch_level_2_split_{split}_y.h5"
        self.transform = transform
        self._x_file: h5py.File | None = None
        self._y_file: h5py.File | None = None

        # TODO: open y_path here (this file is tiny, safe to read eagerly) and store the total patch count in self._length using its shape.
        # Approach 1
        # self._y_file = h5py.File(self.y_path, "r")
        # key = list(self._y_file.keys())[0]
        # self._length: int = self._y_file[key].shape[0]
        # # self._y_file.close()
        # self._y_file = None

        # Approach 2
        with h5py.File(self.y_path, 'r') as f:
            key = list(f.keys())[0]
            self._length: int = f[key].shape[0]

    def __len__(self) -> int:
        return self._length

    def _ensure_open(self) -> None:
        # TODO: if self._x_file is None, open both h5 files in read mode and assign to self._x_file / self._y_file.
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
        if self._y_file is None:
            self._y_file = h5py.File(self.y_path, "r")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()

        # TODO:
        # 1. Read the image at `index` from self._x_file — shape (96, 96, 3), uint8.
        # 2. Read the label at `index` from self._y_file — shape (1, 1, 1), uint8.
        #    Squeeze it down to a single int (0 or 1).
        # 3. If self.transform is set, apply it to the image (we'll wire up
        #    albumentations here in the next step) — otherwise convert the
        #    raw uint8 HWC array to a float32 CHW tensor yourself.
        # 4. Return (image_tensor, label_tensor).
        
        image = self._x_file['x'][index]
        # print("Available keys in y_file:", list(self._y_file.keys()))
        label = self._y_file['y'][index].reshape(-1)[0]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image_tensor = transformed["image"]
        else:
            # converting the raw uint8 HWC array to a float32 CHW tensor or else it throws error
            # the /255 is to normalize the pixel values to [0,1] range
            # permute(2, 0, 1) is to change the shape from HWC(Height, Width, Channels) to CHW
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        label_tensor = torch.tensor(label, dtype=torch.long)

        return (image_tensor, label_tensor)
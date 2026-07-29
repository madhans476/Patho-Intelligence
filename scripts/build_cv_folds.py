"""Build slide-level cross-validation folds from the combined train+valid slide pool.

Test split is NEVER touched here — it stays held out, same as always.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw")
N_FOLDS = 3
SEED = 42  # fixed, so this is reproducible every time we run it


def main() -> None:
    train_meta = pd.read_csv(DATA_DIR / "camelyonpatch_level_2_split_train_meta.csv")
    valid_meta = pd.read_csv(DATA_DIR / "camelyonpatch_level_2_split_valid_meta.csv")

    all_slides = sorted(set(train_meta["wsi"]) | set(valid_meta["wsi"]))
    print(f"combined slide pool: {len(all_slides)} slides")

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(all_slides)

    folds = np.array_split(shuffled, N_FOLDS)
    fold_assignment = {f"fold_{i}": fold.tolist() for i, fold in enumerate(folds)}

    for name, slides in fold_assignment.items():
        print(f"{name}: {len(slides)} slides")

    with open(DATA_DIR / "cv_folds.json", "w") as f:
        json.dump(fold_assignment, f, indent=2)

    print("saved to data/raw/cv_folds.json")


if __name__ == "__main__":
    main()
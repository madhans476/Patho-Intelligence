import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")


def load_wsi_ids(split: str) -> set[str]:
    path = DATA_DIR / f"camelyonpatch_level_2_split_{split}_meta.csv"
    return set(pd.read_csv(path)["wsi"].unique())


def main() -> None:
    train_ids = load_wsi_ids("train")
    valid_ids = load_wsi_ids("valid")
    test_ids = load_wsi_ids("test")

    print(f"train: {len(train_ids)} unique slides")
    print(f"valid: {len(valid_ids)} unique slides")
    print(f"test:  {len(test_ids)} unique slides")

    train_valid_overlap = train_ids & valid_ids
    train_test_overlap = train_ids & test_ids
    valid_test_overlap = valid_ids & test_ids

    print(f"\ntrain ∩ valid: {len(train_valid_overlap)} shared slides")
    print(f"train ∩ test:  {len(train_test_overlap)} shared slides")
    print(f"valid ∩ test:  {len(valid_test_overlap)} shared slides")

    if train_valid_overlap:
        print(f"\n  LEAK — shared slide IDs: {sorted(train_valid_overlap)[:10]}")
    else:
        print("\n  No train/valid slide overlap — split is slide-disjoint.")


if __name__ == "__main__":
    main()
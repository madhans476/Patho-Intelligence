import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

for split in ("train", "valid", "test"):
    path = DATA_DIR / f"camelyonpatch_level_2_split_{split}_meta.csv"
    df = pd.read_csv(path)
    print(f"--- {split} ---")
    print(f"rows: {len(df)}")
    print(f"columns: {list(df.columns)}")
    print(df.head(3))
    print()
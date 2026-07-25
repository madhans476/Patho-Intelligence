import h5py
from pathlib import Path

DATA_DIR = Path("data/raw")

for f in sorted(DATA_DIR.glob("*.h5")):
    with h5py.File(f, "r") as h5f:
        key = list(h5f.keys())[0]
        print(f"{f.name}: shape={h5f[key].shape} dtype={h5f[key].dtype}")

meta_files = sorted(DATA_DIR.glob("*meta.csv"))
print(f"\nmeta CSVs found: {[m.name for m in meta_files] if meta_files else 'none'}")
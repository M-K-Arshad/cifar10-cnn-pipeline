"""Stage 1 - prepare: download CIFAR-10 and save raw arrays to data/raw/."""
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR10

RAW = Path("data/raw")
CACHE = Path(".cache/cifar10")  # torchvision download cache (git-ignored, not tracked)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for split, is_train in [("train", True), ("test", False)]:
        ds = CIFAR10(root=CACHE, train=is_train, download=True)
        np.savez(RAW / f"{split}.npz",
                 images=ds.data,                                  # (N, 32, 32, 3) uint8
                 labels=np.array(ds.targets, dtype=np.int64))
        print(f"[prepare] saved {split}: {ds.data.shape}")


if __name__ == "__main__":
    main()

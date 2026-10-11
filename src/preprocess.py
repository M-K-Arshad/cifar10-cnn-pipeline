"""Stage 2 - preprocess: normalize, split train/val, save to data/processed/."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

RAW, OUT = Path("data/raw"), Path("data/processed")


def load(split, image_size):
    d = np.load(RAW / f"{split}.npz")
    x = torch.from_numpy(d["images"]).permute(0, 3, 1, 2).float() / 255.0  # NHWC uint8 -> NCHW [0,1]
    if image_size != 32:
        x = F.interpolate(x, size=image_size, mode="bilinear", align_corners=False)
    return x, torch.from_numpy(d["labels"])


def main():
    p = yaml.safe_load(open("params.yaml"))["preprocess"]
    OUT.mkdir(parents=True, exist_ok=True)

    x, y = load("train", p["image_size"])
    x_test, y_test = load("test", p["image_size"])

    # train / validation split
    g = torch.Generator().manual_seed(p["seed"])
    idx = torch.randperm(len(x), generator=g)
    n_val = int(len(x) * p["val_size"])
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    # --- NORMALIZATION: per-channel standardization using training-set statistics ---
    mean = x[tr_idx].mean(dim=(0, 2, 3), keepdim=True)
    std = x[tr_idx].std(dim=(0, 2, 3), keepdim=True)

    def normalize(t):
        return ((t - mean) / std).half()
    # --- END NORMALIZATION ---
    # Part A5 Comment: The normalization function is defined to standardize the input tensor `t` using the mean and standard deviation computed from the training set. The result is converted to half-precision (float16) to reduce storage requirements when saving the processed data. This is particularly useful for large datasets, as it helps save space on disk and can speed up data loading during training.
    splits = {"train": (x[tr_idx], y[tr_idx]),
              "val": (x[val_idx], y[val_idx]),
              "test": (x_test, y_test)}
    for name, (xs, ys) in splits.items():
        torch.save({"x": normalize(xs), "y": ys}, OUT / f"{name}.pt")
        print(f"[preprocess] {name}: {tuple(xs.shape)}")


if __name__ == "__main__":
    main()

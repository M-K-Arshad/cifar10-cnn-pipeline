"""Stage 3 - train: build the CNN, train on data/processed/, save models/."""
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

DATA, OUT = Path("data/processed"), Path("models")


def conv_block(cin, cout):
    # Conv2D -> BatchNorm -> ReLU (x2) -> MaxPool
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CNN(nn.Module):
    def __init__(self, num_filters=32, dropout_rate=0.4, num_classes=10):
        super().__init__()
        f = num_filters
        self.features = nn.Sequential(conv_block(3, f), conv_block(f, 2 * f), conv_block(2 * f, 4 * f),
                                      nn.AdaptiveAvgPool2d(4))
        self.classifier = nn.Sequential(          # Dense -> Dropout -> Output(10)
            nn.Flatten(), nn.Linear(4 * f * 16, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(name):
    d = torch.load(DATA / f"{name}.pt")
    return TensorDataset(d["x"], d["y"])


def augment(x):
    """Random horizontal flip + random 4px shift (crop from reflect-padded image)."""
    flip = torch.rand(x.size(0), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    h, w = x.shape[2:]
    padded = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2,)).tolist()
    return padded[:, :, i:i + h, j:j + w]


def run_epoch(model, loader, device, opt=None):
    training = opt is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device).float(), yb.to(device)
            if training:
                xb = augment(xb)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


def main():
    p = yaml.safe_load(open("params.yaml"))["train"]
    torch.manual_seed(p["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()

    train_dl = DataLoader(load_split("train"), batch_size=p["batch_size"], shuffle=True)
    val_dl = DataLoader(load_split("val"), batch_size=512)

    model = CNN(p["num_filters"], p["dropout_rate"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=p["epochs"])

    history, best_acc = [], 0.0
    for epoch in range(1, p["epochs"] + 1):
        tr_loss, tr_acc = run_epoch(model, train_dl, device, opt)
        va_loss, va_acc = run_epoch(model, val_dl, device)
        sched.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
                        "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4)})
        print(f"[train] epoch {epoch:02d}  train_acc={tr_acc:.4f}  val_acc={va_acc:.4f}")
        if va_acc > best_acc:  # keep the best checkpoint
            best_acc = va_acc
            torch.save(model.state_dict(), OUT / "model.pth")

    with open(OUT / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"[train] best val_acc={best_acc:.4f} -> models/model.pth")


if __name__ == "__main__":
    main()

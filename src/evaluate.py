"""Stage 4 - evaluate: test loss/accuracy -> metrics.json, confusion matrix -> PNG."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from train import CNN, get_device  # same model definition as training

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


def main():
    p = yaml.safe_load(open("params.yaml"))["train"]
    device = get_device()

    model = CNN(p["num_filters"], p["dropout_rate"]).to(device)
    model.load_state_dict(torch.load("models/model.pth", map_location=device))
    model.eval()

    test = torch.load("data/processed/test.pt")
    preds, total_loss = [], 0.0
    with torch.no_grad():
        for i in range(0, len(test["y"]), 512):
            xb = test["x"][i:i + 512].to(device).float()
            yb = test["y"][i:i + 512].to(device)
            out = model(xb)
            total_loss += F.cross_entropy(out, yb, reduction="sum").item()
            preds.append(out.argmax(1).cpu())
    preds = torch.cat(preds)

    metrics = {"test_loss": round(total_loss / len(test["y"]), 4),
               "test_accuracy": round((preds == test["y"]).float().mean().item(), 4)}
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] {metrics}")

    cm = confusion_matrix(test["y"].numpy(), preds.numpy())
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"CIFAR-10 confusion matrix (acc={metrics['test_accuracy']:.2%})")
    fig.tight_layout()
    fig.savefig("confusion_matrix.png", dpi=120)


if __name__ == "__main__":
    main()

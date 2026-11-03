"""Final model: train the chosen variant with several seeds, pick the best seed on the VALIDATION images,
export it as TorchScript (model.pt), and - only with --test - evaluate that ONE model on the official test split.

Rules this script follows:
  * training uses only the train images, the seed is chosen on val
  * the test split is loaded only if you pass --test, and only AFTER model.pt has been exported and checked
  * the test split is used for reporting only (never for choosing anything)

Usage (Colab GPU or laptop):
    python final_model.py --seeds 1                  # quick dry run: checks that training and export work (no test)
    python final_model.py --seeds 5 --test           # the real run: final model + one test evaluation
"""
import argparse
import copy
import json
import os

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from experiments import CLASSES, DEVICE, VARIANTS, load, metrics, predict, train_one


def save_confusion_png(y_true, y_pred, path):
    """Row-normalised confusion matrix: the diagonal is the sensitivity of each class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES)))).astype(float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test split: share of each true class predicted as each class")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="cw_resnet18_lr3e4", help="name from VARIANTS in experiments.py")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", default="model.pt")
    parser.add_argument("--test", action="store_true", help="evaluate the chosen model once on the TEST split")
    args = parser.parse_args()
    cfg = VARIANTS[args.variant]
    os.makedirs("results", exist_ok=True)

    X_train, y_train = load("train")
    X_val, y_val = load("val")
    print(f"device: {DEVICE}  variant: {args.variant}  seeds: {args.seeds}")

    # ---- 1. train every seed, keep the one with the best VALIDATION balanced accuracy
    best = None
    for seed in range(args.seeds):
        writer = SummaryWriter(f"runs/final/{args.variant}/seed{seed}")
        model = train_one(cfg, seed, X_train, y_train, X_val, y_val, writer)
        writer.close()
        m = metrics(y_val, predict(model, X_val))
        print(f"seed {seed}  val balanced acc {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_F1']:.3f}")
        if best is None or m["balanced_accuracy"] > best["val"]["balanced_accuracy"]:
            best = {"seed": seed, "model": model, "val": m}
    print(f"\nchosen on validation: seed {best['seed']}  (val balanced acc {best['val']['balanced_accuracy']:.3f})")

    # ---- 2. export as TorchScript and check it (before the test split is touched)
    export_model = copy.deepcopy(best["model"]).cpu().eval()
    scripted = torch.jit.script(export_model)
    scripted.save(args.out)
    loaded = torch.jit.load(args.out)
    with torch.no_grad():
        x = X_val[:16]
        out = loaded(x)
        diff = (out - export_model(x)).abs().max().item()
    assert tuple(out.shape) == (16, len(CLASSES)), f"unexpected output shape {tuple(out.shape)}"
    print(f"{args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB, loads OK, output {tuple(out.shape)}, "
          f"max difference to the original model {diff:.2e}")
    with open("results/final_val_metrics.json", "w") as f:
        json.dump({"variant": args.variant, "seed": best["seed"], **{k: float(v) for k, v in best["val"].items()}}, f, indent=2)

    # ---- 3. the test split: reporting only, one model, once
    if args.test:
        X_test, y_test = load("test")
        pred = predict(best["model"], X_test)
        t = metrics(y_test, pred)
        print("\n=== OFFICIAL TEST SPLIT (final model, reported once) ===")
        print(f"balanced accuracy {t['balanced_accuracy']:.3f}   macro-F1 {t['macro_F1']:.3f}")
        for c in CLASSES:
            print(f"  sensitivity {c:22s} {t['sens_' + c]:.2f}")
        with open("results/final_test_metrics.json", "w") as f:
            json.dump({"variant": args.variant, "seed": best["seed"], **{k: float(v) for k, v in t.items()}}, f, indent=2)
        save_confusion_png(y_test.numpy(), pred.numpy(), "results/final_confusion_test.png")
        print("saved results/final_test_metrics.json and results/final_confusion_test.png")
    else:
        print("\n(test split not used: run again with --test for the final evaluation)")


if __name__ == "__main__":
    main()

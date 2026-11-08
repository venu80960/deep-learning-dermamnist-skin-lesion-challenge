"""Error analysis for the report (validation split only; the test split is not touched).

1. Per-class sensitivity (recall) of four variants on the val images, mean +- std over the 5 seeds
   (from results/metrics_colab_gpu_5seeds.csv - no training needed).
2. Row-normalised confusion matrices on val: baseline (seed 0, cached predictions) vs. the final model.pt.
   Row = true class, column = predicted class; the diagonal is the sensitivity of the class.

Usage:  python error_analysis.py
Output: results/per_class_val_sensitivity.csv / .png, results/confusion_val_baseline_vs_final.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from experiments import CLASSES, PRED_DIR, load, predict

SHOW = ["baseline", "class_weights", "cw_bn", "cw_resnet18_lr3e4"]
LABEL = {"baseline": "baseline", "class_weights": "+ class weights", "cw_bn": "+ BatchNorm",
         "cw_resnet18_lr3e4": "pretrained ResNet-18"}

# ---------------------------------------------------------------- 1. per-class sensitivity
df = pd.read_csv("results/metrics_colab_gpu_5seeds.csv")
cols = [f"sens_{c}" for c in CLASSES]
mean = df[df.variant.isin(SHOW)].groupby("variant")[cols].mean().loc[SHOW]
std = df[df.variant.isin(SHOW)].groupby("variant")[cols].std(ddof=1).loc[SHOW]
mean.columns = std.columns = CLASSES
table = mean.round(2).astype(str) + " +- " + std.round(2).astype(str)
print("Per-class validation sensitivity, mean +- std over 5 seeds (Colab GPU):\n")
print(table.T.to_string())
table.T.to_csv("results/per_class_val_sensitivity.csv")

fig, ax = plt.subplots(figsize=(9, 3.6))
w = 0.2
for k, v in enumerate(SHOW):
    ax.bar(np.arange(7) + (k - 1.5) * w, mean.loc[v], w, yerr=std.loc[v], capsize=2, label=LABEL[v])
ax.set_xticks(range(7), [c.replace(" ", "\n", 1) for c in CLASSES], fontsize=8)
ax.set_ylabel("sensitivity (val)")
ax.set_ylim(0, 1.05)
ax.set_title("Per-class sensitivity: what each change does to each class (mean +- std, 5 seeds)")
ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
fig.tight_layout()
fig.savefig("results/per_class_val_sensitivity.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. confusion matrices on val
X_val, y_val = load("val")
y_true = y_val.numpy()
model = torch.jit.load("model.pt", map_location="cpu")
pred_final = predict(model, X_val).numpy()
cache = os.path.join(PRED_DIR, "baseline_seed0.pt")
pred_base = torch.load(cache).numpy() if os.path.exists(cache) else None


def norm_cm(pred):
    cm = confusion_matrix(y_true, pred, labels=list(range(7))).astype(float)
    return cm / cm.sum(axis=1, keepdims=True)


panels = [("Final model (model.pt), val", pred_final)]
if pred_base is not None:
    panels.insert(0, ("Baseline (seed 0), val", pred_base))
fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4))
axes = np.atleast_1d(axes)
for ax, (title, pred) in zip(axes, panels):
    cm = norm_cm(pred)
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(7), CLASSES, fontsize=7)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
fig.tight_layout()
fig.savefig("results/confusion_val_baseline_vs_final.png", dpi=200)
plt.close(fig)

print("\nLargest confusions of the final model on val (true class -> predicted class, share of the true class):")
cm = norm_cm(pred_final)
pairs = sorted(((cm[i, j], CLASSES[i], CLASSES[j]) for i in range(7) for j in range(7) if i != j), reverse=True)[:6]
for share, t, p in pairs:
    print(f"  {t:22s} -> {p:22s} {share:.2f}")
print("\nsaved: results/per_class_val_sensitivity.csv/.png and results/confusion_val_baseline_vs_final.png")

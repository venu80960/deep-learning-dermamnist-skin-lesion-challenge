"""Data overview for the report (Part A): class counts and one example image per class.

Usage:  python data_overview.py
Output: results/data_overview.png and a count table in the terminal.

Only the train and val splits are looked at (class counts and example images come from the train split).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from medmnist import DermaMNIST

CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)
train = DermaMNIST(split="train", download=True, size=28, root="data")
val = DermaMNIST(split="val", download=True, size=28, root="data")
y_train = train.labels.squeeze(1)
y_val = val.labels.squeeze(1)
n_train = np.bincount(y_train, minlength=7)
n_val = np.bincount(y_val, minlength=7)

print(f"train {len(y_train)} images, val {len(y_val)} images, image shape {train.imgs.shape[1:]}")
print(f"{'class':24s} {'train':>6s} {'share':>7s} {'val':>5s}")
for i, c in enumerate(CLASSES):
    print(f"{c:24s} {n_train[i]:6d} {n_train[i] / n_train.sum():7.1%} {n_val[i]:5d}")
print(f"imbalance: largest class / smallest class = {n_train.max() / n_train.min():.1f}")
print(f"always predicting 'melanocytic nevi' gives accuracy {n_val[5] / n_val.sum():.3f} "
      f"but balanced accuracy {1 / 7:.3f} on val")

fig = plt.figure(figsize=(11, 4.2))
gs = fig.add_gridspec(2, 7, height_ratios=[1.6, 1])

ax = fig.add_subplot(gs[0, :])
x = np.arange(7)
ax.bar(x - 0.2, n_train, width=0.4, label="train", color="#3b6ea5")
ax.bar(x + 0.2, n_val, width=0.4, label="val", color="#e07b39")
for i in range(7):
    ax.text(i - 0.2, n_train[i], str(n_train[i]), ha="center", va="bottom", fontsize=7)
    ax.text(i + 0.2, n_val[i], str(n_val[i]), ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, [""] * 7)
ax.set_ylabel("images")
ax.set_title("DermaMNIST: class counts (7 classes, 28x28 colour images)")
ax.legend(frameon=False)

for i, c in enumerate(CLASSES):
    a = fig.add_subplot(gs[1, i])
    idx = int(np.where(y_train == i)[0][0])
    a.imshow(train.imgs[idx])
    a.set_title(c.replace(" ", "\n", 1), fontsize=7)
    a.axis("off")

fig.tight_layout()
fig.savefig("results/data_overview.png", dpi=200)
print("saved results/data_overview.png")

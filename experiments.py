"""DermaMNIST experiments: train variants with several seeds and compare them to the baseline.

Usage (from the project folder, inside your virtual environment):
    python experiments.py                       # baseline + class_weights
    python experiments.py --variants baseline class_weights augment
    tensorboard --logdir runs                   # look at the curves in the browser

Rules built into this script:
  * only the TRAIN images are used for training
  * only the VAL images are used for evaluation / choosing
  * the TEST split is never loaded here
"""
import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
from medmnist import DermaMNIST
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import v2

# ---------------------------------------------------------------- data (same as the starter)
MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
normalize = v2.Normalize(MEAN, STD)
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]
RESULTS_DIR = "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # GPU on Colab, CPU on the laptop
PRED_DIR = os.path.join(RESULTS_DIR, "preds")   # predictions of every trained model are cached here


def load(split):
    """DermaMNIST split as tensors: images (N, 3, 28, 28) normalised, labels (N,)."""
    os.makedirs("data", exist_ok=True)
    d = DermaMNIST(split=split, download=True, size=28, root="data")
    X = torch.tensor(d.imgs).permute(0, 3, 1, 2).float() / 255
    y = torch.tensor(d.labels).squeeze(1).long()
    return normalize(X), y


# ---------------------------------------------------------------- models
def make_baseline():
    """A: the CNN of the starter notebook (16 071 parameters)."""
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, len(CLASSES)))


def make_bn():
    """B4: the baseline plus BatchNorm after every conv layer (nothing else changed)."""
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(32 * 7 * 7, len(CLASSES)))


def make_bn_deep():
    """B3: make_bn plus a third conv block (64 channels). 28 -> 14 -> 7 -> 3 pixels."""
    return nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(64 * 3 * 3, len(CLASSES)))


def make_resnet18(pretrained=True):
    """B5: ImageNet-pretrained ResNet-18 (He et al. 2016), fine-tuned on DermaMNIST.
    The 28x28 input is upsampled to 64x64 INSIDE the model, so the exported model still takes (N,3,28,28).
    pretrained=False gives the same network with random start weights (ablation)."""
    net = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASSES))
    return nn.Sequential(nn.Upsample(size=(64, 64), mode="bilinear", align_corners=False), net)


# ---------------------------------------------------------------- variants
# ONE change per variant, compared with the baseline. Add your next idea here.
VARIANTS = {
    "baseline":      dict(make_model=make_baseline, class_weights=False, augment=False, epochs=10, lr=1e-3),
    "class_weights": dict(make_model=make_baseline, class_weights=True,  augment=False, epochs=10, lr=1e-3),
    "augment":       dict(make_model=make_baseline, class_weights=False, augment=True,  epochs=10, lr=1e-3),
    "epochs30":      dict(make_model=make_baseline, class_weights=False, augment=False, epochs=30, lr=1e-3),
    "batchnorm":     dict(make_model=make_bn,       class_weights=False, augment=False, epochs=10, lr=1e-3),
    # ---- ladder on top of class weights: each row changes ONE thing compared with "class_weights"
    "cw_epochs30":   dict(make_model=make_baseline, class_weights=True,  augment=False, epochs=30, lr=1e-3),
    "cw_bn":         dict(make_model=make_bn,       class_weights=True,  augment=False, epochs=10, lr=1e-3),
    # ---- next rungs, each ONE change compared with "cw_bn"
    "cw_bn_deep":    dict(make_model=make_bn_deep,  class_weights=True,  augment=False, epochs=10, lr=1e-3),
    "cw_bn_cosine":  dict(make_model=make_bn,       class_weights=True,  augment=False, epochs=10, lr=1e-3, cosine=True),
    # ---- pretrained network, compared with "cw_bn": only the model changes
    "cw_resnet18":   dict(make_model=make_resnet18, class_weights=True,  augment=False, epochs=10, lr=1e-3),
    # ---- tuning the ResNet, each ONE change compared with "cw_resnet18"
    "cw_resnet18_lr3e4": dict(make_model=make_resnet18, class_weights=True, augment=False, epochs=10, lr=3e-4),
    "cw_resnet18_aug":   dict(make_model=make_resnet18, class_weights=True, augment=True,  epochs=10, lr=1e-3),
    # ---- ablation: is it the PRETRAINING? same network and settings as "cw_resnet18_lr3e4", random start weights
    "cw_resnet18_scratch_lr3e4": dict(make_model=lambda: make_resnet18(pretrained=False), class_weights=True,
                                      augment=False, epochs=10, lr=3e-4),
}


# ---------------------------------------------------------------- training / evaluation
def predict(model, X, batch_size=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch_size].to(DEVICE)).argmax(1).cpu() for i in range(0, len(X), batch_size)])


def flip_augment(x):
    """Random horizontal and vertical flips, a different one for every image."""
    flip = torch.rand(len(x), device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    flip = torch.rand(len(x), device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], x.flip(-2), x)


def train_one(cfg, seed, X_train, y_train, X_val, y_val, writer):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cfg["make_model"]().to(DEVICE)
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True, generator=gen)
    weights = None
    if cfg["class_weights"]:
        counts = torch.bincount(y_train, minlength=len(CLASSES)).float()
        weights = (len(y_train) / (len(CLASSES) * counts)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # optional cosine learning-rate schedule: lr falls smoothly from cfg["lr"] to ~0 over all epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"]) if cfg.get("cosine") else None

    for epoch in range(cfg["epochs"]):
        model.train()
        total, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if cfg["augment"]:
                x = flip_augment(x)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
            n += len(x)
        if scheduler is not None:
            scheduler.step()
        val_pred = predict(model, X_val)
        writer.add_scalar("train/loss", total / n, epoch + 1)
        writer.add_scalar("val/balanced_accuracy", balanced_accuracy_score(y_val, val_pred), epoch + 1)
    return model


def metrics(y_true, y_pred):
    sens = recall_score(y_true, y_pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    return {"balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "macro_F1": f1_score(y_true, y_pred, average="macro"),
            **{f"sens_{c}": s for c, s in zip(CLASSES, sens)}}


def fast_bacc(y_true, y_pred, k=len(CLASSES)):
    """Balanced accuracy (= mean recall over the classes present in y_true), same value as sklearn but ~100x faster."""
    total = torch.bincount(y_true, minlength=k).float()
    hit = torch.bincount(y_true[y_pred == y_true], minlength=k).float()
    present = total > 0
    return float((hit[present] / total[present]).mean())


def bootstrap_diff_mean_over_seeds(y_true, preds_a, preds_b, n_draws=2000, seed=0):
    """95 % interval of  mean_over_seeds(bacc(B)) - mean_over_seeds(bacc(A))  on the same val images.
    In every draw the SAME resampled images are used for A and B and for all seeds."""
    g = torch.Generator().manual_seed(seed)
    diffs = []
    for _ in range(n_draws):
        idx = torch.randint(0, len(y_true), (len(y_true),), generator=g)
        a = np.mean([fast_bacc(y_true[idx], p[idx]) for p in preds_a])
        b = np.mean([fast_bacc(y_true[idx], p[idx]) for p in preds_b])
        diffs.append(b - a)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return round(float(lo), 3), round(float(hi), 3)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["baseline", "class_weights"])
    parser.add_argument("--reference", default="baseline",
                        help="extra comparison: also print the interval vs. this variant (e.g. class_weights)")
    parser.add_argument("--seeds", type=int, default=3, help="number of seeds per variant (0, 1, 2, ...); handout needs >= 3")
    parser.add_argument("--retrain", action="store_true",
                        help="ignore saved predictions in results/preds and train every model again")
    args = parser.parse_args()
    for needed in ("baseline", args.reference):        # everything is compared with A (and the reference)
        if needed not in args.variants:
            args.variants = [needed] + args.variants

    os.makedirs(PRED_DIR, exist_ok=True)
    X_train, y_train = load("train")
    X_val, y_val = load("val")
    print("device:", DEVICE, " train:", tuple(X_train.shape), " val:", tuple(X_val.shape))

    all_preds, rows = {}, []
    for name in args.variants:
        cfg = VARIANTS[name]
        all_preds[name] = []
        per_seed = []
        for seed in range(args.seeds):
            cache = os.path.join(PRED_DIR, f"{name}_seed{seed}.pt")
            if os.path.exists(cache) and not args.retrain:
                pred = torch.load(cache)               # already trained earlier: reuse the saved val predictions
            else:
                writer = SummaryWriter(f"runs/{name}/seed{seed}")
                model = train_one(cfg, seed, X_train, y_train, X_val, y_val, writer)
                writer.close()
                pred = predict(model, X_val)
                torch.save(pred, cache)                # saved immediately, so a crash later loses nothing
            all_preds[name].append(pred)
            m = metrics(y_val, pred)
            per_seed.append(m)
            rows.append({"variant": name, "seed": seed, **m})
            print(f"{name:14s} seed {seed}  balanced acc {m['balanced_accuracy']:.3f}  macro-F1 {m['macro_F1']:.3f}")

    # ---- summary table: mean ± std over seeds, interval vs. baseline
    print(f"\n=== summary on VALIDATION images (mean ± std over {args.seeds} seeds) ===")
    for name in args.variants:
        rs = [r for r in rows if r["variant"] == name]
        bacc = np.array([r["balanced_accuracy"] for r in rs])
        f1 = np.array([r["macro_F1"] for r in rs])
        line = f"{name:14s} bacc {bacc.mean():.3f} ± {bacc.std(ddof=1):.3f}   macro-F1 {f1.mean():.3f} ± {f1.std(ddof=1):.3f}"
        if name != "baseline":
            lo, hi = bootstrap_diff_mean_over_seeds(y_val, all_preds["baseline"], all_preds[name])
            line += f"   95% interval vs baseline: [{lo}, {hi}]"
        if args.reference != "baseline" and name not in ("baseline", args.reference):
            lo, hi = bootstrap_diff_mean_over_seeds(y_val, all_preds[args.reference], all_preds[name])
            line += f"   vs {args.reference}: [{lo}, {hi}]"
        print(line)

    # ---- save everything so nothing gets lost
    path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nper-seed metrics saved to {path}")


if __name__ == "__main__":
    main()

"""Compare already-trained variants WITHOUT retraining, using the predictions saved in results/preds.

Usage:
    python compare.py cw_bn cw_resnet18 cw_resnet18_lr3e4 --seeds 5
means: for every variant after the first one, print mean +- std of val balanced accuracy / macro-F1 and the
95 % bootstrap interval of the difference  (variant - first variant)  on the validation images.
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from experiments import PRED_DIR, bootstrap_diff_mean_over_seeds, fast_bacc, load


def load_preds(name, n_seeds):
    return [torch.load(os.path.join(PRED_DIR, f"{name}_seed{s}.pt")) for s in range(n_seeds)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", help="first = reference, the others are compared with it")
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()

    _, y_val = load("val")
    preds = {v: load_preds(v, args.seeds) for v in args.variants}
    ref = args.variants[0]
    print(f"reference: {ref}   (interval = variant minus reference, 95 % bootstrap on val images, {args.seeds} seeds)")
    for v in args.variants:
        bacc = np.array([fast_bacc(y_val, p) for p in preds[v]])
        f1 = np.array([f1_score(y_val, p, average="macro") for p in preds[v]])
        line = f"{v:28s} bacc {bacc.mean():.3f} +- {bacc.std(ddof=1):.3f}   macro-F1 {f1.mean():.3f} +- {f1.std(ddof=1):.3f}"
        if v != ref:
            lo, hi = bootstrap_diff_mean_over_seeds(y_val, preds[ref], preds[v])
            line += f"   diff vs {ref}: [{lo}, {hi}]"
        print(line)


if __name__ == "__main__":
    main()

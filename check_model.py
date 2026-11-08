"""Safety check of model.pt - run this before handing the model in (uses only the VAL split, never test).

Checks what the lecturer's evaluation script needs:
  * loads with torch.jit.load on the CPU and is in eval mode
  * accepts normalised (N, 3, 28, 28) float tensors of any batch size
  * returns logits of shape (N, 7), all finite
  * gives the same output when called twice (no randomness)
  * reproduces the validation balanced accuracy we reported (0.618)

Usage:  python check_model.py
"""
import sys

import torch

from experiments import load, metrics, predict

PATH = sys.argv[1] if len(sys.argv) > 1 else "model.pt"
torch.set_grad_enabled(False)
model = torch.jit.load(PATH, map_location="cpu")
model.eval()
print(f"loaded {PATH}, training mode: {model.training}")

ok = True
for n in (1, 7, 64, 256):
    x = torch.randn(n, 3, 28, 28)
    a, b = model(x), model(x)
    good = tuple(a.shape) == (n, 7) and torch.isfinite(a).all().item() and torch.equal(a, b)
    ok &= good
    print(f"batch {n:4d}: output {tuple(a.shape)}, finite, repeatable -> {'OK' if good else 'FAILED'}")

x1 = torch.randn(5, 3, 28, 28)
same = torch.allclose(model(x1), torch.cat([model(x1[i:i + 1]) for i in range(5)]), atol=1e-4)
ok &= same
print(f"batch result equals one-by-one result -> {'OK' if same else 'FAILED'}")

X_val, y_val = load("val")
m = metrics(y_val, predict(model, X_val))
print(f"validation balanced accuracy {m['balanced_accuracy']:.3f} (reported: 0.618), macro-F1 {m['macro_F1']:.3f}")
ok &= abs(m["balanced_accuracy"] - 0.618) < 0.002
print("\nALL CHECKS PASSED" if ok else "\nSOMETHING FAILED - tell Claude")

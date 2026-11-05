"""Save one example image per class from the VALIDATION split into examples/ (for the app's example buttons).
The test split is not used here.   Usage:  python make_examples.py"""
import os

import numpy as np
from medmnist import DermaMNIST
from PIL import Image

CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]
os.makedirs("examples", exist_ok=True)
d = DermaMNIST(split="val", download=True, size=28, root="data")
labels = d.labels.squeeze(1)
for k, name in enumerate(CLASSES):
    img = d.imgs[int(np.where(labels == k)[0][0])]                      # first val image of class k, (28, 28, 3) uint8
    Image.fromarray(img).resize((224, 224), Image.Resampling.LANCZOS).save(f"examples/{k}_{name.replace(' ', '_')}.png")
    print("saved", name)

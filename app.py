"""Skin-lesion classifier app (SRH DermaMNIST challenge), built from the course template app_template.py.

Upload a skin-lesion image -> top-3 classes with probabilities, plus two extras:
  * an occlusion map: which image regions the model relies on for its top class
  * a confidence note with what the model achieved per class on the official test split

Local:                 python app.py            (model.pt in the same folder; opens http://127.0.0.1:7860)
Colab:                 !pip install -q gradio   then run this file; demo.launch(share=True) prints a public link
Hugging Face Spaces:   new Space (SDK: Gradio), upload app.py, model.pt, requirements.txt (= requirements_app.txt)
                       and the examples/ folder
Do not change MEAN / STD: they are the numbers of the starter notebook.
"""
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
MEAN = [0.7631, 0.5381, 0.5614]
STD = [0.1366, 0.1543, 0.1692]
CLASSES = ["actinic keratoses", "basal cell carcinoma", "benign keratosis", "dermatofibroma",
           "melanoma", "melanocytic nevi", "vascular lesions"]

# short, general descriptions of the classes (for the user of the demo)
ABOUT = {
    "actinic keratoses": "rough, scaly patches from sun damage; can be a precursor of skin cancer",
    "basal cell carcinoma": "the most common skin cancer; grows locally and rarely spreads",
    "benign keratosis": "harmless growths such as seborrheic keratoses or sun spots",
    "dermatofibroma": "a harmless firm nodule in the skin",
    "melanoma": "a malignant tumour of the pigment cells and the most dangerous skin cancer",
    "melanocytic nevi": "ordinary moles, harmless in the vast majority of cases",
    "vascular lesions": "benign blood-vessel lesions such as cherry angiomas",
}
# sensitivity of OUR final model per class on the official test split (results/final_test_metrics.json)
TEST_SENSITIVITY = {
    "actinic keratoses": 0.61, "basal cell carcinoma": 0.56, "benign keratosis": 0.42, "dermatofibroma": 0.57,
    "melanoma": 0.59, "melanocytic nevi": 0.76, "vascular lesions": 0.76,
}
LOW_CONFIDENCE = 0.50      # below this top-1 probability the app shows a warning

model = torch.jit.load(str(HERE / "model.pt"), map_location="cpu")   # your exported model
model.eval()
prepare = v2.Compose([
    v2.ToImage(),                                  # PIL image -> tensor (3, H, W)
    v2.Resize((28, 28), antialias=True),           # the size the model was trained on
    v2.ToDtype(torch.float32, scale=True),         # 0...255 -> 0...1
    v2.Normalize(MEAN, STD),                       # exactly like in the starter notebook
])


def occlusion_map(x, target, patch=8, stride=2):
    """How much does P(target) drop when a patch x patch square is hidden? Returns a (28, 28) array.
    Hidden = set to 0, i.e. the average colour after normalisation. High values = the region matters."""
    positions = list(range(0, 28 - patch + 1, stride))
    boxes, batch = [], []
    for r in positions:
        for c in positions:
            xo = x.clone()
            xo[:, :, r:r + patch, c:c + patch] = 0.0
            boxes.append((r, c))
            batch.append(xo)
    with torch.no_grad():
        base = F.softmax(model(x), dim=1)[0, target].item()
        probs = F.softmax(model(torch.cat(batch)), dim=1)[:, target]
    heat, count = torch.zeros(28, 28), torch.zeros(28, 28)
    for (r, c), p in zip(boxes, probs):
        heat[r:r + patch, c:c + patch] += base - p.item()
        count[r:r + patch, c:c + patch] += 1
    return (heat / count.clamp(min=1)).numpy()


def overlay(image, heat, size=336):
    """Red overlay on the uploaded image: the more the model relies on a region, the redder it is."""
    base = np.asarray(image.convert("RGB").resize((size, size)), dtype=np.float32) / 255
    h = np.clip(heat, 0, None)
    if h.max() > 0:
        h = h / h.max()
    h = Image.fromarray((h * 255).astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
    h = (np.asarray(h, dtype=np.float32) / 255)[..., None]
    red = np.zeros_like(base)
    red[..., 0] = 1.0
    out = base * (1 - 0.65 * h) + red * 0.65 * h
    return (out * 255).astype(np.uint8)


def describe(ranked):
    """Text under the prediction. `ranked` = list of (class name, probability), best first."""
    name, p = ranked[0]
    lines = [f"**Most likely: {name}** ({p:.0%}) - {ABOUT[name]}.",
             f"On our official test split the model recognised {TEST_SENSITIVITY[name]:.0%} of the images of this class."]
    if p < LOW_CONFIDENCE:
        lines.append(f"⚠️ **Low confidence** (below {LOW_CONFIDENCE:.0%}): the model is unsure, treat this result with extra care.")
    if name != "melanoma" and any(n == "melanoma" and q >= 0.15 for n, q in ranked):
        lines.append("⚠️ Melanoma is among the likely classes. Our model misses about 4 in 10 melanomas on the test split.")
    lines.append("The probabilities of this class-weighted model are not calibrated. "
                 "*Student project - not a medical device. A real skin lesion must be assessed by a doctor.*")
    return "\n\n".join(lines)


def classify(image):
    """PIL image -> (top-3 probabilities, occlusion overlay, explanation text)."""
    if image is None:
        return {}, None, "Please upload an image."
    x = prepare(image.convert("RGB")).unsqueeze(0)          # (1, 3, 28, 28)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0]
    top = probs.topk(3)
    ranked = [(CLASSES[i], float(p)) for p, i in zip(top.values, top.indices)]
    heat = occlusion_map(x, int(top.indices[0]))
    return dict(ranked), overlay(image, heat), describe(ranked)


examples = [[str(p)] for p in sorted((HERE / "examples").glob("*.png"))]

demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="skin lesion image"),
    outputs=[gr.Label(num_top_classes=3, label="top-3 prediction"),
             gr.Image(label="what the model looks at (red = important for the top class)"),
             gr.Markdown()],
    examples=examples or None,
    title="Skin lesion classifier (DermaMNIST challenge)",
    description="Upload an image of a skin lesion (or click an example). Student project - not a medical device!",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch(share=True)   # share=True: public link (needed in Colab); on Hugging Face Spaces it is ignored

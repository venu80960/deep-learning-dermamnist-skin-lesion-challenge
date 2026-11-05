# DermaMNIST skin-lesion classification (SRH Deep Learning challenge)

Course project "Challenge – Skin lesion classification" (Deep Learning, SRH, lecturer: Annika Stiehl).
Task: classify 28x28 colour dermatoscopy images into 7 classes (DermaMNIST, MedMNIST v2). Main metric:
**balanced accuracy** (mean sensitivity over the 7 classes) on the official test split.

**Team:** Group 1 - Ankita Phande, Venu Kamatam, Ye Min

## Result

| | Balanced accuracy | Macro-F1 |
|---|---|---|
| Validation, chosen seed | 0.618 | 0.574 |
| **Official test split (final model, evaluated once)** | **0.610** | **0.536** |

Final model: ImageNet-pretrained ResNet-18, fine-tuned end-to-end on 28x28 inputs upsampled to 64x64 inside the
model, class-weighted cross-entropy, Adam, lr 3e-4, 10 epochs, no augmentation. The best of 5 seeds was chosen on
the validation images only. Test sensitivity per class: nevus 0.76, vascular 0.76, actinic keratoses 0.61,
melanoma 0.59, dermatofibroma 0.57, basal cell carcinoma 0.56, benign keratosis 0.42
(`results/final_test_metrics.json`, confusion matrix in `results/final_confusion_test.png`).
This is a course project, not a diagnostic tool.

## Experiments (validation images, 5 seeds, mean ± std; Colab T4 GPU)

Interval = 95 % bootstrap interval of the difference to the baseline (val images resampled, averaged over seeds).
"Reference" is the variant a row changes ONE thing about.

| Variant | What changes | Reference | Balanced acc. | Macro-F1 | Interval vs baseline |
|---|---|---|---|---|---|
| baseline | starter CNN (16k parameters) | n/a | 0.444 ± 0.044 | 0.459 | n/a |
| class_weights | class-weighted loss | baseline | 0.533 ± 0.023 | 0.418 | [0.050, 0.128] |
| augment | random flips | baseline | 0.433 ± 0.041 | 0.461 | [-0.035, 0.014] |
| epochs30 | 30 instead of 10 epochs | baseline | 0.470 ± 0.023 | 0.492 | [-0.002, 0.056] |
| batchnorm | BatchNorm in the CNN | baseline | 0.457 ± 0.052 | 0.474 | [-0.010, 0.038] |
| cw_epochs30 | + 30 epochs | class_weights | 0.524 ± 0.033 | 0.449 | [0.042, 0.118] |
| cw_bn | + BatchNorm | class_weights | 0.551 ± 0.023 | 0.426 | [0.063, 0.150] |
| cw_bn_deep | + third conv block | cw_bn | 0.555 ± 0.023 | 0.457 | [0.068, 0.154] |
| cw_bn_cosine | + cosine LR schedule | cw_bn | 0.551 ± 0.004 | 0.440 | [0.062, 0.150] |
| cw_resnet18 | pretrained ResNet-18, lr 1e-3 | cw_bn | 0.579 ± 0.025 | 0.495 | [0.084, 0.182] |
| cw_resnet18_lr3e4 | + lr 3e-4 | cw_resnet18 | 0.580 ± 0.017 | 0.530 | [0.090, 0.181] |
| cw_resnet18_aug | + flips | cw_resnet18 | 0.534 ± 0.069 | 0.421 | [0.048, 0.133] |
| cw_resnet18_scratch_lr3e4 | same net, random init (ablation) | cw_resnet18_lr3e4 | 0.540 ± 0.035 | 0.494 | [0.053, 0.138] |

Findings: class weights are the one robust gain. BatchNorm, a deeper CNN and a cosine schedule give no clear
extra gain; the pretrained ResNet-18 is best on average, but against `cw_bn` its interval [-0.011, 0.068]
still contains 0. Pretraining helps compared with random initialisation (interval vs. the pretrained model
[-0.077, -0.006]); flips hurt the ResNet (interval vs. `cw_resnet18_lr3e4` [-0.086, -0.012]).
Raw per-seed numbers: `results/metrics_colab_gpu_5seeds.csv`.

## Rules followed
* Training uses the train split only; models, seeds and settings are chosen on the validation split.
* The test split is loaded only by `final_model.py --test`, after `model.pt` has been exported, to report one
  final model. It was evaluated once and never used for choosing anything.
* Input format of the starter: `(N, 3, 28, 28)` float tensor normalised with the given MEAN/STD; `model.pt` is
  TorchScript and returns logits `(N, 7)`.

## Repository
| File | Purpose |
|---|---|
| `experiments.py` | trains variants with several seeds, prints mean ± std and bootstrap intervals, logs to TensorBoard, caches val predictions in `results/preds/` |
| `compare.py` | pairwise bootstrap comparison from the cached predictions (no retraining) |
| `final_model.py` | trains the chosen variant, picks the best seed on val, exports and checks `model.pt`, evaluates once on test with `--test` |
| `model.pt` | final TorchScript model |
| `app.py` | Gradio app: upload an image, get the top-3 classes with probabilities, an occlusion map and a short note |
| `app_colab.ipynb` | one-click notebook that starts the app in Colab (public link) |
| `make_examples.py`, `examples/` | example images for the app (one validation image per class) |
| `requirements_app.txt` | the few packages the app needs |
| `results/` | metrics of the runs, final test metrics, confusion matrix |
| `day4_challenge_starter*.ipynb` | course starter notebook (baseline) |
| `requirements.txt`, `requirements-lock.txt` | direct dependencies / full pinned environment (Python 3.12) |

## Reproduce
```
py -3.12 -m venv .venv                      # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate                      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python experiments.py --seeds 5 --reference cw_bn --variants class_weights augment epochs30 batchnorm cw_epochs30 cw_bn cw_bn_deep cw_bn_cosine cw_resnet18 cw_resnet18_lr3e4 cw_resnet18_aug cw_resnet18_scratch_lr3e4
python compare.py cw_resnet18_lr3e4 cw_resnet18_scratch_lr3e4 cw_resnet18_aug cw_resnet18 --seeds 5
python final_model.py --seeds 5 --test      # trains, exports model.pt, evaluates once on test
tensorboard --logdir runs                   # training curves
```
DermaMNIST is downloaded automatically by the `medmnist` package into `data/` on the first run.

Hardware and determinism: the experiment table was produced on a Colab T4 GPU (about 25 minutes for all
variants with 5 seeds). On a CPU the same code is bit-for-bit repeatable (checked) but slow: one ResNet-18 seed
takes about 23 minutes on a Ryzen 5 5500U. GPU training is not bit-for-bit repeatable between runs: re-running
gives results of the same quality (validation balanced accuracy of the ResNet-18 variants roughly 0.52–0.62
across seeds and runs), not identical weights. The exported `model.pt` is the fixed artefact that was tested.

## Experiment tracking
TensorBoard (`runs/<variant>/seed<k>`: training loss and validation balanced accuracy per epoch).

## Sources
* Yang et al., MedMNIST v2, Scientific Data 2023.
* He et al., Deep Residual Learning for Image Recognition, CVPR 2016.
* Ioffe & Szegedy, Batch Normalization, ICML 2015.
* Loshchilov & Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts, ICLR 2017.
* Doerrich, Di Salvo, Brockmann, Ledig, Rethinking model prototyping through the MedMNIST+ dataset collection,
  Scientific Reports 15, 7669 (2025), doi:10.1038/s41598-025-92156-9.

## AI use statement
Claude (Anthropic) was used as a tutor and coding assistant: it explained the task and the handout, proposed the
experimental design, drafted `experiments.py`, `compare.py` and `final_model.py`, and helped interpret results.


## App
The app (`app.py`, Gradio) takes an uploaded skin lesion image and shows the top-3 classes with probabilities,
a map of the image regions that matter most for the top class (occlusion sensitivity), and a short note with the
model's test sensitivity for that class. It is a student project, not a medical device.

**Run it in Colab (one click):** open
[`app_colab.ipynb`](https://colab.research.google.com/github/venu80960/deep-learning-dermamnist-skin-lesion-challenge/blob/main/app_colab.ipynb)
and choose *Runtime > Run all*. After about a minute a public `gradio.live` link appears. A CPU runtime is enough.
If that link redirects to GitHub, open Colab, choose *File > Open notebook > GitHub*, paste the repository URL and
select `app_colab.ipynb`.

**Run it locally:**
```
pip install -r requirements.txt
python app.py        # then open http://127.0.0.1:7860
```
(Hugging Face Spaces was not used: Gradio Spaces need a paid plan there; the handout allows Colab.)

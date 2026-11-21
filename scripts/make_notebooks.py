"""Generate the Google Colab notebooks in notebooks/ (valid .ipynb JSON, one per guide step).

    python scripts/make_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict]) -> dict:
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}, "colab": {"provenance": []}}}


SETUP = [
    md("""
## 0 · Put the project on Google Drive (once)

1. Zip the `smart Attendance` folder on your laptop **without** `data/`, `models/` and `reports/`
   (they are rebuilt here). Rename the zip to `smart_attendance.zip`.
2. Upload it to `MyDrive/` in Google Drive.

Keeping data and results on Drive means they survive when Colab disconnects (Colab's own disk is wiped).
"""),
    code("""
from google.colab import drive
drive.mount('/content/drive')
"""),
    code("""
import os, zipfile
PROJECT = '/content/drive/MyDrive/smart_attendance'
if not os.path.exists(PROJECT):
    zipfile.ZipFile('/content/drive/MyDrive/smart_attendance.zip').extractall(PROJECT)
# if the zip contained a top-level folder, step into it
inner = [d for d in os.listdir(PROJECT) if os.path.isfile(os.path.join(PROJECT, d, 'pyproject.toml'))]
if inner:
    PROJECT = os.path.join(PROJECT, inner[0])
%cd "$PROJECT"
!ls
"""),
    code("""
# Colab already has torch, torchvision and OpenCV; this adds the rest and installs our package
!pip install -q -r requirements.txt ultralytics kagglehub
!pip install -q -e .
"""),
]

NB02 = notebook([
    md("""
# Requirement 2 · Datasets in Google Colab

This notebook downloads the Kaggle classroom dataset, maps its labels to *attentive / distracted / inactive*,
makes a **leakage-safe** train / validation / test split, and shows the label statistics.

**No GPU is needed for this step** (Runtime → Change runtime type → CPU is fine and saves your GPU quota).
"""),
    *SETUP,
    md("""
## 1 · Download from Kaggle

`kagglehub` is Kaggle's official download library. This dataset is public, so **no Kaggle account or API key
is needed**. The download is 129 MB.

Dataset page: https://www.kaggle.com/datasets/mustafaxgm/classroom-student-engagement-dataset
(licence CC BY-NC 4.0: free for non-commercial academic work, with attribution).
"""),
    code("""
import kagglehub
kaggle_path = kagglehub.dataset_download('mustafaxgm/classroom-student-engagement-dataset')
print(kaggle_path)
!ls "$kaggle_path/dataset"
"""),
    md("""
## 2 · Labels, de-duplication and the leakage-safe split

`prepare()` does everything in one call:

* reads the YOLO labels (one line per student: class, centre x, centre y, width, height, all 0-1)
* maps the 8 actions to 3 states (`look_forward/read/write/handrise/stand` → attentive,
  `turn_head/using_device` → distracted, `sleep` → inactive)
* finds near-duplicate frames with a perceptual hash and removes any that would cross splits
* splits by **recording group** (same students, same room) so no student appears in two splits
* writes three versions: people-only YOLO labels (counting), 3-state YOLO labels (detector training),
  and per-student crops (ResNet / GoogLeNet classification)
"""),
    code("""
from smartattend.data.kaggle_classroom import prepare, make_figures
report = prepare('data/raw', 'data/processed/kaggle_classroom', source=kaggle_path)
print('duplicates:', report['duplicates'])
for split, s in report['splits'].items():
    print(f"{split:5s} {s['images']:4d} images {s['people']:5d} people  groups={s['groups']}  states={s['states']}")
"""),
    md("""
**Expected output**

```
duplicates: {'near_duplicate_pairs_within_group': 50, 'removed_across_groups': 0}
train  217 images  2277 people  groups=['misc', 'owais', 'seminar']  states={'attentive': 1759, 'distracted': 448, 'inactive': 70}
val    179 images  1167 people  groups=['taimoor-class']  states={'attentive': 954, 'distracted': 143, 'inactive': 70}
test    85 images  1159 people  groups=['taimoor-zaeem']  states={'distracted': 646, 'attentive': 421, 'inactive': 92}
```

The 50 near-duplicates are consecutive video frames *inside* one group, so they stay in one split and cannot leak.
"""),
    code("""
from IPython.display import Image, display
for p in make_figures('data/processed/kaggle_classroom', 'reports/figures/data'):
    display(Image(str(p)))
"""),
    md("""
## 3 · The other datasets (optional here)

| purpose | dataset | how |
|---|---|---|
| face recognition, fairness | LFW: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset | `kagglehub.dataset_download('jessicali9530/lfw-dataset')` (118 MB), or `python -m smartattend stage data` (Hugging Face mirror with identity labels) |
| person detection on classroom-like photos | COCO 2017 val (official site, not Kaggle: the Kaggle copy is 27 GB) | `python -m smartattend stage data` downloads only the 250 MB of annotations and about 600 selected images |
| alertness frames with timestamps | drowsiness faces (Hugging Face, MIT) | `python -m smartattend stage data` |

Storage: everything together is about 1 GB, well within Colab's limits.
"""),
    code("""
!python -m smartattend stage data      # COCO-classroom + LFW + drowsiness (about 10 minutes)
!du -sh data/raw/* data/processed/*
"""),
])


NB03 = notebook([
    md("""
# Requirement 3 · Mathematical sandbox

Tensors, broadcasting, inner products, reshaping, SVD and numerical gradient checking, written in plain NumPy.
**CPU runtime is enough.** See `docs/guide/03_math_sandbox.md` for the explanation of every step.
"""),
    *SETUP,
    md("## 1 · Printed worked examples + figures"),
    code("!python -m smartattend math"),
    code("""
from IPython.display import Image, display
for name in ['broadcasting', 'inner_product', 'reshape_order', 'svd_low_rank', 'gradient_check']:
    display(Image(f'reports/figures/nnfund/{name}.png'))
"""),
    md("## 2 · Try it yourself"),
    code("""
import numpy as np
from smartattend.nnfund.tensor_sandbox import broadcast_shape, low_rank_approx, gradient_check

print(broadcast_shape((5, 1, 3), (4, 1)))            # -> (5, 4, 3)
A = np.random.default_rng(0).normal(size=(6, 4))
print(np.linalg.norm(A - low_rank_approx(A, 4)))     # full rank -> ~0
f = lambda x: float(np.sum(x ** 3))
print(gradient_check(f, lambda x: 3 * x ** 2, np.array([1.0, -2.0])))   # tiny -> formula is right
"""),
    md("## 3 · Tests"),
    code("!python -m pytest -q tests/test_tensor_sandbox.py"),
])


NB04 = notebook([
    md("""
# Requirement 4 · Neurons, multilayer network and backpropagation

McCulloch-Pitts neurons → a learning sigmoid neuron → a multilayer network trained with backpropagation on real
classroom crops, then the same network in **TensorFlow/Keras** and a check that NumPy, PyTorch and TensorFlow
compute identical gradients. See `docs/guide/04_neurons_mlp.md`.

**CPU runtime is enough** (about 10 minutes, most of it extracting ResNet18 features once).
This notebook needs the Kaggle crops from notebook 02 (`data/processed/kaggle_classroom/crops`).
"""),
    *SETUP,
    code("""
# Colab already has TensorFlow and PyTorch. Recreate the Kaggle crops if this is a fresh runtime:
import os, kagglehub
if not os.path.exists('data/processed/kaggle_classroom/crops'):
    from smartattend.data.kaggle_classroom import prepare
    prepare('data/raw', 'data/processed/kaggle_classroom',
            source=kagglehub.dataset_download('mustafaxgm/classroom-student-engagement-dataset'))
"""),
    md("## 1 · Everything in one command (printed walkthrough + training + figures)"),
    code("!python -m smartattend neurons"),
    code("""
from IPython.display import Image, display
for name in ['activations_step_sigmoid_tanh', 'neuron_decision_boundaries', 'mlp_crops_training']:
    display(Image(f'reports/figures/nnfund/{name}.png'))
"""),
    md("## 2 · Step by step: McCulloch-Pitts and the sigmoid neuron"),
    code("""
from smartattend.nnfund import neural_networks as nn
print('\\n'.join(nn.demo_mcculloch_pitts() + nn.demo_sigmoid_neuron()))
"""),
    md("## 3 · Forward pass, loss and backward pass by hand"),
    code("print('\\n'.join(nn.demo_worked_example()))"),
    md("## 4 · The same network in Keras (TensorFlow)"),
    code("""
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.Input(shape=(512,)),
    tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3),                      # logits for attentive / distracted / inactive
])
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.summary()                                    # 33,027 parameters, same as the NumPy MLP
"""),
    code("print('\\n'.join(nn.demo_frameworks()))"),
    md("## 5 · Tests"),
    code("!python -m pytest -q tests/test_neural_networks.py tests/test_mlp_gradients.py tests/test_neurons_optimizers.py"),
])


NB05 = notebook([
    md("""
# Requirement 5 · Optimizers and ablations

Batch GD, SGD (batch 1), mini-batch SGD, Momentum and Nesterov, then one-change-at-a-time ablations of
initialisation, normalisation, regularisation and augmentation, on real face frames (engaged / neutral / disengaged).
See `docs/guide/05_optimizers_ablations.md`.

**CPU runtime is enough** (about 25 minutes; most of it is SGD with batch size 1, which makes 1,293 updates per epoch).
"""),
    *SETUP,
    code("""
# the drowsiness face frames (Hugging Face, MIT licence), downloaded by the data stage
import os
if not os.path.exists('data/raw/drowsiness'):
    !python -m smartattend stage data
"""),
    md("## 1 · Run everything"),
    code("!python -m smartattend optimize"),
    code("""
from IPython.display import Image, display
for name in ['optimizer_paths', 'optimizer_comparison_real', 'ablations_real', 'overfitting_real']:
    display(Image(f'reports/figures/nnfund/{name}.png'))
"""),
    md("## 2 · Try one change yourself"),
    code("""
from smartattend.nnfund import optimization as opt
data = opt.load_faces('data/raw/drowsiness')
for change in [{}, {'init': 'zeros'}, {'aug': 'shift'}, {'dropout': 0.3, 'l2': 1e-3}]:
    h = opt.train(data, epochs=30, seed=0, **{**opt.BASE, **change})
    print(f"{str(change):35s} train F1 {h['train_f1'][-1]:.3f}  val F1 {h['val_f1_last']:.3f}  test F1 {h['test_f1_last']:.3f}")
"""),
    md("## 3 · Tests"),
    code("!python -m pytest -q tests/test_optimization.py"),
])


NB06 = notebook([
    md("""
# Requirement 6 · Hebbian, modified Hebbian, competitive, error-correction learning and the Boltzmann machine

Each classic rule is applied to the project's own data and checked against a standard method:
eigenfaces (Hebb / Oja / Sanger vs SVD), grouping unknown faces (competitive learning vs k-means),
engagement from face pixels (perceptron and delta rule vs the MLP), and a Restricted Boltzmann Machine on faces.
See `docs/guide/06_classic_learning_rbm.md`. **CPU runtime is enough** (about 2 minutes).
"""),
    *SETUP,
    code("""
import os
if not os.path.exists('data/raw/drowsiness') or not os.path.exists('data/processed/lfw/emb_clean.npz'):
    !python -m smartattend stage data      # drowsiness frames + LFW
    !python -m smartattend stage face      # LFW face embeddings (about 20 minutes on CPU)
"""),
    md("## 1 · Run everything"),
    code("!python -m smartattend classic"),
    code("""
from IPython.display import Image, display
for name in ['hebbian_eigenfaces', 'competitive_errorcorr_rbm', 'rbm_face_reconstructions']:
    display(Image(f'reports/figures/nnfund/{name}.png'))
"""),
    md("## 2 · The rules in a few lines each"),
    code("""
import numpy as np
from smartattend.nnfund import classic_real as cr
X = np.load('data/processed/drowsiness_faces32.npz')['train_X'].reshape(-1, 1024)
Xc = X - X.mean(0); Xc /= np.sqrt(np.mean(np.sum(Xc**2, 1)))
w, norms = cr.oja_run(Xc, lr=0.01, epochs=5)
pc1 = np.linalg.svd(Xc, full_matrices=False)[2][0]
print('Oja |w| per epoch:', np.round(norms, 3), ' |cos| with SVD PC1:', round(abs(w @ pc1) / np.linalg.norm(w), 4))
_, hebb = cr.hebbian_run(Xc, lr=0.01, epochs=5)
print('Hebb |w| per epoch:', [f'{v:.1e}' for v in hebb])
"""),
    md("## 3 · Tests"),
    code("!python -m pytest -q tests/test_classic_real.py tests/test_classic_and_rbm.py"),
])


NB07 = notebook([
    md("""
# Requirement 7 · The classroom app: detection, counting, per-student states, trends, convolution

Everything is measured on the Kaggle **test classroom** (taimoor-zaeem), which no model saw during training:
YOLOv8 / Faster R-CNN counting, YOLOv8n fine-tuned on the training classrooms, ResNet18 and GoogLeNet fine-tuned for
attentive / distracted / inactive, the trend over the video, and 2D/3D convolution figures.
See `docs/guide/07_classroom_app.md`.

**Use a GPU runtime** (Runtime → Change runtime type → T4 GPU): fine-tuning takes minutes instead of an hour.
"""),
    *SETUP,
    code("""
import os, kagglehub
if not os.path.exists('data/processed/kaggle_classroom/crops'):
    from smartattend.data.kaggle_classroom import prepare
    prepare('data/raw', 'data/processed/kaggle_classroom',
            source=kagglehub.dataset_download('mustafaxgm/classroom-student-engagement-dataset'))
"""),
    md("## 1 · Fine-tune YOLOv8n on the training classrooms (people only)"),
    code("""
import torch
from smartattend.vision.finetune_yolo import finetune
device_note = 'GPU' if torch.cuda.is_available() else 'CPU (slow)'
print('training on', device_note)
s = finetune('data/processed/kaggle_classroom/yolo_person', 'models', 'reports/experiments/detection',
             epochs=15, run_name='yolov8n_kaggle', target_name='yolov8n-kaggle.pt')
print(s['metrics_val'])
"""),
    md("## 2 · Evaluate everything on the test classroom"),
    code("!python -m smartattend classroom"),
    code("""
from IPython.display import Image, display
for name in ['classroom_test_eval.png', 'classroom_test_annotated.jpg', 'conv2d_explained.png', 'conv3d_motion.png',
             'resnet_filters_featuremaps.png']:
    display(Image(f'reports/figures/vision/{name}'))
"""),
    md("## 3 · The live app"),
    code("""
# Streamlit needs a public tunnel on Colab; on your laptop just run:
#   python -m streamlit run app/streamlit_app.py
!python -m smartattend analyze --input data/processed/kaggle_classroom/yolo_person/images/test --out reports/session_test --no-video
!head -5 reports/session_test/engagement_timeline.csv
"""),
    md("## 4 · Tests"),
    code("!python -m pytest -q tests/test_classroom_eval.py tests/test_vision_sequence.py"),
])


NB08 = notebook([
    md("""
# Requirement 8 · Benchmark: R-CNN, Fast R-CNN, Faster R-CNN, RetinaNet, YOLO, ResNet, GoogLeNet

All detector families on the same Kaggle **test classroom**, same metric code, thresholds from validation.
R-CNN and Fast R-CNN are implemented in `src/smartattend/vision/rcnn.py`; see `docs/guide/08_benchmark.md`.

**Use a GPU runtime.** On a laptop CPU the whole benchmark takes about 2 hours (R-CNN alone needs about 11 s per frame).
"""),
    *SETUP,
    code("""
import os, kagglehub
if not os.path.exists('data/processed/kaggle_classroom/crops'):
    from smartattend.data.kaggle_classroom import prepare
    prepare('data/raw', 'data/processed/kaggle_classroom',
            source=kagglehub.dataset_download('mustafaxgm/classroom-student-engagement-dataset'))
"""),
    md("""
## 1 · Region proposals (selective search)

Selective search is in OpenCV's *contrib* build. On Colab it is safe to install it directly
(on the laptop we load it from a separate folder so the project's OpenCV is untouched).
"""),
    code("""
!pip install -q opencv-contrib-python
!python scripts/selective_search.py
"""),
    md("""
## 2 · Requirement 7 first (it fine-tunes YOLOv8n and the ResNet18 / GoogLeNet classifiers that are compared here)
"""),
    code("""
from smartattend.vision.finetune_yolo import finetune
import os
if not os.path.exists('models/yolo/yolov8n-kaggle.pt'):
    finetune('data/processed/kaggle_classroom/yolo_person', 'models', 'reports/experiments/detection',
             epochs=15, run_name='yolov8n_kaggle', target_name='yolov8n-kaggle.pt')
!python -m smartattend classroom
"""),
    md("## 3 · The benchmark"),
    code("!python -m smartattend benchmark --tv-epochs 2"),
    code("""
from IPython.display import Image, display
display(Image('reports/figures/vision/benchmark_detectors.png'))
"""),
    md("## 4 · Tests"),
    code("!python -m pytest -q tests/test_rcnn.py"),
])


NB09 = notebook([
    md("""
# Requirement 9 · LSTM and Transformer, latent space, attention, step / sigmoid / tanh

Sequence models on **real classroom video**: estimate the class's attentive share from the last 8 frames of the app's
own outputs (LSTM vs Transformer vs simple baselines), a Transformer that attends across the students of one frame,
the latent spaces of the ResNet and the LSTM, and the three LSTM cell non-linearities.
See `docs/guide/09_sequence_models.md`. **CPU runtime is enough** (about 5 minutes) once Requirement 7 has run.
"""),
    *SETUP,
    code("""
# needs the fine-tuned YOLOv8n and the ResNet18 state classifier from Requirement 7
import os
assert os.path.exists('models/yolo/yolov8n-kaggle.pt') and os.path.exists('models/classroom_states_resnet18.pt'), \
    'run notebook 07 first'
"""),
    md("## 1 · Run everything"),
    code("!python -m smartattend sequence"),
    code("""
from IPython.display import Image, display
for name in ['classroom_lstm_transformer', 'classroom_attention', 'classroom_latent_space', 'lstm_activations']:
    display(Image(f'reports/figures/sequence/{name}.png'))
"""),
    md("## 2 · An LSTM cell and a Transformer block, step by step"),
    code("""
import torch
from smartattend.sequence.lstm_trend import FlexLSTM
from smartattend.sequence.classroom_seq import TemporalTransformer, gradient_through_time

x = torch.randn(1, 8, 7)                       # 1 video window, 8 frames, 7 numbers per frame
for act in ['tanh', 'sigmoid', 'step']:
    h = FlexLSTM(7, 16, act)(x)                # hidden state after every frame
    g = gradient_through_time(act, T=20, trials=10)
    print(f'{act:8s} hidden states {tuple(h.shape)}  gradient reaching 19 frames back: {g[0] / g[-1]:.1e}')

y, attention = TemporalTransformer()(x, return_attn=True)
print('Transformer output', y.shape, ' attention over the 8 frames:', attention[0].detach().numpy().round(3), ' sum =', float(attention.sum()))
"""),
    md("## 3 · Tests"),
    code("!python -m pytest -q tests/test_classroom_seq.py tests/test_vision_sequence.py"),
])


NB10 = notebook([
    md("""
# Requirement 10 · Integration and evaluation

The whole classroom pipeline (fine-tuned YOLOv8n → ResNet18 states) on the Kaggle **test classroom**:
robustness, fairness, error analysis, latency, one scorecard, the acceptance check and the results report.
See `docs/guide/10_integration_evaluation.md`. **CPU runtime is enough** (about 10 minutes) once notebooks 07–09 have run.
"""),
    *SETUP,
    md("## 1 · Final evaluation"),
    code("!python -m smartattend evaluate"),
    code("""
from IPython.display import Image, display
for name in ['final_robustness_fairness_errors', 'final_error_gallery']:
    display(Image(f'reports/figures/eval/{name}.png'))
"""),
    md("## 2 · Acceptance check: every requirement against its target"),
    code("""
!python -m smartattend stage pipeline_latency
!python -m smartattend acceptance
"""),
    md("## 3 · One report with every number"),
    code("""
!python scripts/build_report.py
from IPython.display import Markdown, display
display(Markdown(open('reports/RESULTS.md', encoding='utf-8').read()))
"""),
    md("## 4 · All tests"),
    code("!python -m pytest -q"),
])


def main() -> None:
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, nb in [("02_datasets_colab.ipynb", NB02), ("03_math_sandbox_colab.ipynb", NB03),
                     ("04_neurons_mlp_colab.ipynb", NB04), ("05_optimizers_ablations_colab.ipynb", NB05),
                     ("06_classic_learning_rbm_colab.ipynb", NB06), ("07_classroom_app_colab.ipynb", NB07),
                     ("08_benchmark_colab.ipynb", NB08),
                     ("09_sequence_models_colab.ipynb", NB09),
                     ("10_integration_evaluation_colab.ipynb", NB10)]:
        (out / name).write_text(json.dumps(nb, indent=1), encoding="utf-8")
        print("wrote", out / name)


if __name__ == "__main__":
    main()

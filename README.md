# Smart Attendance and Classroom Engagement Analyzer

Face-recognition attendance, people counting, seating analysis and class-level alertness trends, plus a
from-scratch NumPy neural-network fundamentals module. Python implementation of Project 22 (EdTech).
Runs entirely on CPU (developed on an Intel i3 laptop with 8 GB RAM).

**Recommendation: GO for a supervised attendance pilot, NO-GO for unsupervised rollout or individual
engagement scoring.** See [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) §10.

## Headline results (all reproducible, collected in [reports/RESULTS.md](reports/RESULTS.md))

| | result |
|---|---|
| Attendance (LFW, 200 simulated 30-student sessions) | recall **99.4 %**, false-present **0.1 %**, fairness gaps < 1 pp |
| Person detection (COCO-Classroom test, YOLOv8-s) | mAP@0.5 **0.816**, count error **0.99** people/image |
| Person segmentation (YOLOv8-n-seg) | IoU **0.738** |
| Engagement / alertness (ResNet-18 fine-tuned) | macro-F1 **0.862** on one split; **0.68–0.71** in 5-fold CV (probes) |
| Transformer class aggregator | 36 % lower error than naive averaging (simulated) |
| End-to-end demo | **12/12** register entries correct (1 look-alike flagged for review) |
| Real classroom video (Kaggle test classroom, 1,159 students) | fine-tuned YOLOv8n mAP@0.5 **0.608**, count off by **1.5** students per frame |
| Attentive / distracted / inactive in a new classroom | macro-F1 **0.47** (target 0.60: not met) |
| Acceptance check | **10 of 13** requirements pass (`python -m smartattend acceptance`) |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                                    # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt ultralytics streamlit
pip install -e .
pip install tensorflow-cpu                                # optional: Keras comparisons (steps 4 and 9)
python -m pytest                                          # 125 tests
```

## Use

```bash
streamlit run app/streamlit_app.py                        # GUI -> http://localhost:8501
python -m smartattend enroll                              # roster from data/roster/<id>/*.jpg
python -m smartattend analyze --input lecture.mp4         # register + engagement timeline + annotated video
python -m smartattend attend --input 0                    # attendance only, webcam
python scripts/make_demo.py                               # build + run the demo classroom
```

Full instructions: [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## Step-by-step guide (one page, one command and one Colab notebook per requirement)

| # | requirement | guide | command | Colab notebook |
|---|---|---|---|---|
| 1 | use case, measurable requirements, acceptance tests | [01](docs/guide/01_use_case_and_requirements.md) | `python -m smartattend acceptance` | – |
| 2 | datasets, licences, de-duplication, leakage-safe splits | [02](docs/guide/02_datasets.md) | `python -m smartattend stage kaggle_data` | `notebooks/02_datasets_colab.ipynb` |
| 3 | math sandbox: tensors, broadcasting, SVD, gradient checking | [03](docs/guide/03_math_sandbox.md) | `python -m smartattend math` | `03_math_sandbox_colab.ipynb` |
| 4 | McCulloch-Pitts, sigmoid neuron, MLP with backpropagation | [04](docs/guide/04_neurons_mlp.md) | `python -m smartattend neurons` | `04_neurons_mlp_colab.ipynb` |
| 5 | Batch GD, SGD, Momentum, Nesterov and ablations | [05](docs/guide/05_optimizers_ablations.md) | `python -m smartattend optimize` | `05_optimizers_ablations_colab.ipynb` |
| 6 | Hebbian, Oja, competitive, error-correction learning, RBM | [06](docs/guide/06_classic_learning_rbm.md) | `python -m smartattend classic` | `06_classic_learning_rbm_colab.ipynb` |
| 7 | the classroom app: detection, states, trends, convolution | [07](docs/guide/07_classroom_app.md) | `python -m smartattend classroom` | `07_classroom_app_colab.ipynb` |
| 8 | benchmark: R-CNN, Fast R-CNN, Faster R-CNN, YOLO, ResNet, GoogLeNet, RetinaNet | [08](docs/guide/08_benchmark.md) | `python -m smartattend benchmark` | `08_benchmark_colab.ipynb` |
| 9 | LSTM, Transformer, latent space, attention, step / sigmoid / tanh | [09](docs/guide/09_sequence_models.md) | `python -m smartattend sequence` | `09_sequence_models_colab.ipynb` |
| 10 | integration and evaluation | [10](docs/guide/10_integration_evaluation.md) | `python -m smartattend evaluate` | `10_integration_evaluation_colab.ipynb` |

Steps 7 to 10 use the Kaggle classroom dataset (https://www.kaggle.com/datasets/mustafaxgm/classroom-student-engagement-dataset,
CC BY-NC 4.0) and need, in order: `stage kaggle_data`, the YOLOv8n fine-tune in guide 07, `classroom`,
`python -I scripts/selective_search.py --cv2-path models/tools/opencv_contrib`, `benchmark`, `sequence`, `evaluate`.

## Reproduce every experiment

```bash
python -m smartattend stage data          # download + curate COCO-Classroom, LFW, drowsiness
python -m smartattend sandbox             # NumPy fundamentals figures
python -m smartattend experiments         # optimisers, ablations, overfitting
python -m smartattend stage engagement    # ResNet-18 vs GoogLeNet, fine-tuning
python -m smartattend stage engagement_cv # 5-fold grouped CV
python -m smartattend stage sequence      # LSTM trends, transformer aggregator, VAE, latent space
python -m smartattend stage detection     # YOLOv8-n/s, fine-tuned, Faster R-CNN, RetinaNet
python -m smartattend stage seating       # semantic + instance segmentation, seat maps
python -m smartattend stage face          # LFW verification, sessions, fairness, robustness
python -m smartattend stage conv_viz      # 2-D / 3-D convolution figures
python -m smartattend stage pipeline_latency
python scripts/build_report.py            # -> reports/RESULTS.md
NODE_PATH=<folder with pptxgenjs> node scripts/build_deck.js <apply_theme.js>  # -> docs/presentation.pptx (21 slides)
```

## Deliverables

| deliverable | location |
|---|---|
| application (CLI + GUI) | `src/smartattend/cli.py`, `src/smartattend/pipeline.py`, `app/streamlit_app.py` |
| NN fundamentals module + gradient/unit tests | `src/smartattend/nnfund/`, `tests/` |
| vision pipeline (detection, segmentation, engagement, conv visualisation) | `src/smartattend/vision/`, weights in `models/` |
| advanced models (LSTM, transformer, VAE, latent space) | `src/smartattend/sequence/` |
| evaluation package (mAP, count error, macro-F1, fairness, latency, robustness) | `src/smartattend/eval/`, `reports/` |
| requirements & acceptance tests | [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) |
| step-by-step guides and Colab notebooks | `docs/guide/`, `notebooks/` |
| dataset card | [docs/DATASET_CARD.md](docs/DATASET_CARD.md) |
| model card | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |
| technical report (go/no-go) | [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) |
| user guide | [docs/USER_GUIDE.md](docs/USER_GUIDE.md) |
| presentation | [docs/presentation.pptx](docs/presentation.pptx) |
| demonstration video | `reports/demo/annotated.mp4` (pipeline output); a narrated screen recording must be made by a person |

## Responsible use
Opt-in consent only; embeddings are stored, not photos; the teacher confirms every register; engagement
is class-level alertness and must never be used to grade or discipline individuals. Details in
[docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) and [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

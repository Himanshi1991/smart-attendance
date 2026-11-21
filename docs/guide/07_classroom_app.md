# Requirement 7 — The classroom app: detection (YOLO / Faster R-CNN), ResNet, seating, engagement, trends, convolution

## What this module is

This is the product itself: from a classroom photo or video, **count the students, map occupied/empty seats, label
each student attentive / distracted / inactive, track it over time and take attendance by face.**
The app (`app/streamlit_app.py`, `python -m smartattend analyze`) was built earlier. This step **measures it
honestly on a classroom it has never seen** and improves the parts that were weak.

```
camera frame ─► YOLOv8 people + chairs ─► seat map (occupied / empty)
            └─► YuNet faces ─► SFace embedding ─► roster match ─► attendance (3-of-5 frames rule)
                          └─► head turn + alertness ResNet18 ─► attentive / distracted / inactive / unclear
per frame ─► timeline CSV ─► trend charts (Streamlit) ─► session report
```

Code: `src/smartattend/vision/classroom_eval.py` (this evaluation), `pipeline.py`, `vision/detectors.py`,
`vision/seating.py`, `vision/conv_viz.py`, `vision/finetune_yolo.py` · tests: `tests/test_classroom_eval.py` (6) and
`tests/test_vision_sequence.py` · Colab: `notebooks/07_classroom_app_colab.ipynb` (**GPU recommended**).

## Run it

```bash
# fine-tune YOLOv8n on the Kaggle training classrooms (about 1 hour on a laptop CPU, a few minutes on a Colab GPU)
python -c "from smartattend.vision.finetune_yolo import finetune; finetune('data/processed/kaggle_classroom/yolo_person', 'models', 'reports/experiments/detection', epochs=15, run_name='yolov8n_kaggle', target_name='yolov8n-kaggle.pt')"
python -m smartattend classroom        # evaluation (about 25 min the first time; results are cached afterwards)
python -m streamlit run app/streamlit_app.py      # the app: live attendance, enrolment, photo/video analysis
```

**Test set:** the *taimoor-zaeem* recordings of the Kaggle classroom dataset (85 frames, **1,159 labelled students**,
11–15 per frame). Its students and room never appear in training or validation (Requirement 2 split by recording group).
Every threshold and model choice below was made on the **validation** classroom.

## 7a. Counting students: YOLO vs Faster R-CNN, and fine-tuning

```
detector                    mAP50  mAP50-95  count MAE   bias  ms/img
(ms/img = median after one warm-up image, laptop on battery at 1.2 GHz: worst case)
yolov8n                     0.328     0.111       2.60  -2.44      67
yolov8s                     0.321     0.112       1.61  -1.33     151
fasterrcnn_resnet50_fpn     0.322     0.098       1.73  +1.35    2037
yolov8n-kaggle              0.608     0.230       1.46  -1.01      70
```

- **COCO-pretrained detectors find the students but draw different boxes.** Kaggle's boxes cover the visible upper
  body behind the desk; COCO's "person" boxes are drawn differently, so many correct detections miss IoU 0.5
  (mAP50 about 0.32 for all three).
- **Fine-tuning YOLOv8n on 217 training frames** (15 epochs, first 10 layers frozen, the 80-class head kept) nearly
  **doubles mAP50 (0.33 → 0.61)** and gives the **best count: off by 1.5 students per frame** (out of about 14),
  at the same speed as YOLOv8n.
- **Faster R-CNN** (two-stage: region proposals, then classification) has the best recall of the pretrained models
  (0.54 vs 0.47 for YOLOv8s) but is **30× slower** (2 s per frame on this CPU) and over-counts (+1.35).
  **YOLO** (one-stage: a single pass predicts all boxes) is the right choice for a live classroom camera.
- Everything **undercounts slightly** (bias −1.0): students hidden behind others are missed.
- Remaining errors (`classroom_test_annotated.jpg`): an occasional second, nested box on one student, and partly
  hidden students in the back row.

`classroom_test_eval.png`, left: labelled vs predicted count for every frame.

## 7b. Attentive / distracted / inactive for each student

Each labelled student box is classified, so detection errors do not mix in. Labels come from the Kaggle actions
(look forward / read / write / hand raise / stand → attentive; turn head / using device → distracted;
sleep → inactive). The test classroom is unusual: **56 % of its students are labelled distracted**
(training classrooms: 20 %).

```
always 'attentive' baseline: macro-F1 0.178
app's face rule: a face was found and judged for 59% of students (no face 374, too small 105)
   on judged students: accuracy 0.524  macro-F1 0.447;  all students (unjudged = attentive): macro-F1 0.292
resnet18  fine-tuned: val 0.423 (epoch 3)  TEST accuracy 0.617  macro-F1 0.459 [95% CI 0.430-0.491]  F1 0.66/0.63/0.09   16 ms/crop
googlenet fine-tuned: val 0.416 (epoch 3)  TEST accuracy 0.557  macro-F1 0.444 [95% CI 0.415-0.474]  F1 0.65/0.54/0.14   33 ms/crop
```

**Transfer learning setup** (`finetune_cnn`): ImageNet weights; only the last block (`layer4` for ResNet18,
`inception5a/5b` for GoogLeNet) and a new 3-class layer are trained; class-weighted loss (inactive has only 70 training
crops); flip and brightness augmentation; AdamW with a cosine schedule; 6 epochs; the epoch with the best
**validation** macro-F1 is kept.

| | ResNet18 | GoogLeNet |
|---|---|---|
| parameters (trainable) | 11.2 M (8.4 M) | 5.6 M (2.3 M) |
| idea | residual connections: `out = F(x) + x` lets very deep nets train | inception blocks: 1×1, 3×3, 5×5 convolutions side by side |
| test macro-F1 | 0.459 | 0.444 |
| speed on CPU | 16 ms/crop | 33 ms/crop |

**What this shows (honestly):**
- **Both CNNs beat the baseline clearly, but this task is hard:** macro-F1 about 0.45 on a new classroom. ResNet18 and
  GoogLeNet are statistically tied (overlapping confidence intervals).
- **"Inactive" (sleeping) is almost never found** (F1 0.09–0.14): 70 training examples is far too few.
- **Validation peaks at epoch 3 and then drops** while training loss keeps falling: overfitting to the three
  training classrooms, the same pattern as Requirements 4 and 5.
- **The app's face rule only sees 59 % of students** (backs of heads, masks, faces under 30 px). Where it sees a face it
  is about as good as the CNNs (0.447), **but it badly over-calls "inactive"** (241 predicted vs 33 labelled among
  judged students). Its alertness model was trained on **one person's** drowsiness frames (Requirement 2), and
  that does not transfer to a real classroom.
- **For the product:** the head-turn rule for *distracted* is useful; the *inactive* signal should not be trusted
  until it is retrained on classroom data. Both are reported as *estimates* in the app, never used for grading
  (`docs/MODEL_CARD.md`).

## 7c. Trends over time

The whole test video, frame by frame, end to end: fine-tuned YOLOv8n finds the students, ResNet18 labels each one
(both chosen on validation). `classroom_test_eval.png`, middle:

```
attentive share per frame: labelled mean 0.36, predicted mean 0.69, mean abs. error 0.33, correlation 0.61
```

- The **shape of the trend is right** (correlation 0.61): when the class becomes more attentive, the prediction rises.
- The **level is wrong** (0.69 vs 0.36): the model learned from classrooms where 77 % of students were attentive, and
  it carries that expectation into a class that was mostly turning heads.
- So the app's trend chart is good for **"is attention going up or down in this lesson?"** and not for
  **"what percentage of the class was attentive?"** That is how the app presents it.

In the app, the timeline is saved per session as `engagement_timeline.csv` and drawn as a chart (Streamlit
"Analyze photo/video" mode, or `python -m smartattend analyze --input <video>`).

## 7d. Seating

YOLOv8 detects people **and chairs/benches in the same pass** (`detect_with_seats`). A seat counts as occupied when a
person box overlaps it; rows are found by clustering seat heights. The Kaggle data has **no seat labels**, so occupancy
cannot be scored here. In dense rooms the row clustering can merge rows (documented in `docs/TECHNICAL_REPORT.md`, seating section;
the fix is a one-time per-room seat calibration).

## 7e. 2D and 3D convolution, visualised

`conv2d_explained.png` works through one Sobel-x convolution on a real 7×7 patch of the test classroom:

```
out[0,0] = (110 + 2·115 + 107) − (127 + 2·144 + 140) = 447 − 555 = −108      (a dark-to-bright vertical edge)
```

…then shows the whole frame through vertical-edge, horizontal-edge, Laplacian and blur kernels, and what stride 2 with
padding 1 does to the output size (`out = (in + 2·pad − k) / stride + 1`).

`resnet_filters_featuremaps.png`: ResNet18's 64 learned first-layer filters (edges and colour blobs, like the
hand-made kernels) and its feature maps getting smaller and more abstract layer by layer
(112 → 56 → 28 → 14 → 7 pixels).

`conv3d_motion.png`: a 3D kernel spans **time** as well as space. A 2×3×3 "frame difference" kernel cancels everything
that stays still (the body) and lights up what moves (a raised hand). Video models use this to see actions such as
hand-raising.

```python
def conv2d_naive(img, k, stride=1, pad=0):
    img = np.pad(img, pad)
    kh, kw = k.shape
    H = (img.shape[0] - kh) // stride + 1
    W = (img.shape[1] - kw) // stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(img[i*stride:i*stride+kh, j*stride:j*stride+kw] * k)   # an inner product (Req. 3)
    return out
```

## Checklist
- [x] people detection and counting with YOLOv8n/s and Faster R-CNN, plus YOLOv8n fine-tuned on classroom data
- [x] ResNet18 and GoogLeNet transfer learning for attentive / distracted / inactive, with confidence intervals
- [x] the app's face-based rule evaluated on the same students; its weak point found and documented
- [x] trend over a whole test video, predicted vs labelled
- [x] seating from people + chairs in one pass (unscored: no seat labels exist)
- [x] 2D convolution worked by hand on a real frame, learned ResNet filters and feature maps, 3D convolution on motion
- [x] fair timing (warm-up, median) after finding that the first-run timings were wrong
- [x] 6 new tests

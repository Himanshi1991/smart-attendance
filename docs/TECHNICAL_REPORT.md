# Smart Attendance and Classroom Engagement Analyzer — Technical Report

Project 22 · Education Technology · Python implementation (the brief asked for C/C++/Java; the
project owner chose Python, and the "from-scratch" parts are written in pure NumPy to keep the maths visible).

Every number below is produced by a command in `docs/USER_GUIDE.md` §6 and collected automatically in
`reports/RESULTS.md` (`python scripts/build_report.py`). Seed 42, Intel i3-1215U laptop, 8 GB RAM, CPU only.

---

## 1. Use case and requirements (step 1)

A school wants to (a) replace roll-call with face-recognition attendance from a classroom camera and
(b) give teachers an **aggregate** view of class alertness over a lecture. Stakeholders, measurable
targets, acceptance tests and responsible-use limits are in `docs/REQUIREMENTS.md`. The targets are
summarised against results in §9.

## 2. Data (step 2) — see `docs/DATASET_CARD.md`

| dataset | purpose | size | split |
|---|---|---|---|
| COCO-Classroom (COCO val2017, curated by captions + scene objects) | detection, counting, segmentation, seating | 598 images, 4,995 people | 450 / 44 / 104 images, disjoint, dHash-deduplicated, unseen by all pretrained detectors |
| LFW + Columbia attributes | face verification, simulated class sessions, fairness, robustness | 13,233 faces, 5,749 people | enrolment and probe photos always different |
| Drowsiness faces (MIT) | engagement (alertness) classifier, real temporal trends | 2,376 frames, 1 person | 5-s time blocks, 0.5-s buffers, stratified (prevents near-duplicate leakage) |

Provenance and per-image licences are recorded in the processed data. Synthetic data is used only
where stated and is never reported as real-world accuracy.

## 3. Neural-network fundamentals in NumPy (steps 3–6)

Module `src/smartattend/nnfund/` (no deep-learning framework):

* **Tensor sandbox**: hand-written broadcasting rule (verified against NumPy), explicit inner products and
  matmul, reshape order, SVD low-rank approximation (Eckart–Young), central-difference gradient checking,
  with explanatory figures (`reports/figures/nnfund/`).
* **Neurons**: McCulloch–Pitts gates (AND/OR/NOT/NOR), a sigmoid neuron (learns AND/OR, provably fails
  XOR), and an MLP that solves XOR.
* **MLP**: Dense, activation, dropout and batch-norm layers with explicit backward passes. **Every
  parameter gradient is checked against finite differences in the test suite (relative error < 1e-6)**.
  `MLP.summary()` separates learned parameters from hand-chosen hyperparameters, and both are logged with every run.
* **Optimisers** (synthetic engagement features, 60 epochs): Momentum and Nesterov reach about 30 % lower
  training loss than SGD (0.276 / 0.269 vs 0.359). On an ill-conditioned quadratic, both beat SGD (unit-tested).
* **Ablations** (3 seeds, mean val accuracy): baseline 0.781; small-normal init 0.819; sigmoid 0.826;
  dropout 0.815; L2 0.796; batch-norm 0.787; augmentation 0.785. Seed standard deviation (0.03–0.05) is
  as large as most differences, which is itself a finding: on this small tabular problem no single trick dominates.
* **Overfitting study** (60 training samples, 128-128 MLP): without regularisation, train acc 1.00 and
  val 0.69 (gap 0.31, validation loss rising from epoch 10). L2 = 5e-2 gives val 0.77 (gap 0.11).
  Dropout alone did not help at this data size.
* **Classic learning rules**: Hebbian (weight norm diverges), Oja (converges to the first principal
  component, cos > 0.99), competitive winner-take-all (finds cluster centres), delta rule (recovers
  linear weights to 1e-3).
* **Energy-based model**: a binary RBM with CD-1 on 4×4 bars-and-stripes. Reconstruction MSE falls from
  0.25 to about 0. Training patterns sit in clearly lower free-energy valleys than random images. The free-energy
  formula is unit-tested against brute-force summation.

## 4. Vision pipeline (steps 7–8)

### 4.1 Candidate selection

| family | candidates in brief | benchmarked | reason |
|---|---|---|---|
| two-stage detection | R-CNN, Fast R-CNN, Faster R-CNN | **Faster R-CNN ResNet-50-FPN** | R-CNN / Fast R-CNN use selective-search proposals (tens of seconds per image on CPU) and are strictly superseded by Faster R-CNN's learned RPN. They are covered conceptually, not benchmarked. |
| one-stage detection | YOLO, RetinaNet | **YOLOv8-n, YOLOv8-s, YOLOv8-n fine-tuned, RetinaNet ResNet-50-FPN** | speed/accuracy frontier |
| classification | ResNet, GoogLeNet | **ResNet-18, GoogLeNet** | engagement classifier backbones |
| semantic segmentation | — | **LR-ASPP MobileNetV3** | person/chair pixels on CPU |
| instance segmentation | — | **YOLOv8-n-seg** | per-person masks, chairs for empty-seat detection |

### 4.2 Detection benchmark (COCO-Classroom test, 104 images, 767 people)

| model | mAP@0.5 | mAP@[.5:.95] | count MAE (all / complete-annotation images) | recall small / medium / large | CPU ms |
|---|---|---|---|---|---|
| YOLOv8-n | 0.766 | 0.537 | 2.28 / 1.54 | 0.31 / 0.72 / 0.94 | 150 |
| **YOLOv8-s** | **0.816** | **0.599** | 1.70 / **0.99** | 0.31 / 0.74 / 0.95 | 361 |
| YOLOv8-n fine-tuned (6 ep.) | 0.766 | 0.523 | 1.88 / 1.25 | 0.26 / 0.67 / 0.91 | 138 |
| Faster R-CNN R50-FPN | 0.797 | 0.473 | **1.47** / 1.04 | **0.35** / 0.72 / 0.93 | 3,772 |
| RetinaNet R50-FPN | 0.765 | 0.446 | 1.85 / 1.40 | 0.17 / 0.56 / 0.88 | 3,823 |

Findings:

* **YOLOv8-s is selected for the product**: best mAP at both IoU regimes, within 0.1 people of the best
  count error on fully annotated images, and about 10× faster than the two-stage models.
* **Faster R-CNN is the most robust to motion blur** (mAP 0.63 vs 0.50) and best on small people, but
  3.8 s per image rules it out on CPU. It is the recommended choice if a GPU becomes available.
* **Fine-tuning YOLOv8-n did not help** (val mAP@[.5:.95] 0.475 → 0.481, test 0.537 → 0.523). The
  450 multi-person training images are a similar distribution to what COCO pretraining already
  covers, and a frozen backbone limits adaptation. Real classroom CCTV is needed for fine-tuning to pay off.
* **Back-row problem**: recall on small people (< 32 px) is only 0.17–0.35 for every model, against
  0.88–0.95 for large ones. This size-recall gap is the vision analogue of a fairness gap (students
  at the back are under-counted) and drives the camera-placement recommendation in §10.
* **COCO crowd regions**: the six largest count "errors" are dense class photos where COCO labels about 13
  people and marks the rest as one crowd region. The model's higher counts there are mostly correct,
  so the complete-annotation figure is the fair one. Both are reported.

### 4.3 Seating (segmentation + seat map)

Person-pixel IoU on the test set: **instance YOLOv8-n-seg 0.738** vs **semantic LR-ASPP 0.642** (113 vs
130 ms). Instance masks separate neighbouring students; semantic masks merge the whole group (see
`reports/experiments/seating/seating_examples.png`).

The seat map groups people into rows by vertical position, orders them left to right (R1C1, R1C2, …),
and reports chairs not overlapped by any person as empty seats. **Limitation:** there is no row-level
ground truth, so this heuristic is unvalidated. In dense tiered group photos it merges rows (2 found
where 4 exist). For deployment, a one-time per-room seat-layout calibration should replace it.

### 4.4 Convolution visualisations
`reports/figures/vision/`: a step-by-step 2-D convolution (`conv2d_naive` is unit-tested against OpenCV),
stride and padding, ResNet-18 conv1 filters and feature maps by depth, and a **3-D spatio-temporal
convolution** whose temporal-difference kernel cancels the static body and responds only to a raised arm.

### 4.5 Engagement (alertness) classifier

Face crops → grayscale + CLAHE (to match the training domain) → ImageNet backbone → 3 classes. Class
imbalance is handled with inverse-frequency weights, and model selection uses validation macro-F1.

| model | test macro-F1 (single split) | 5-fold grouped CV macro-F1 |
|---|---|---|
| ResNet-18 linear probe (frozen) | 0.708 | **0.677 ± 0.090** (folds 0.57–0.81) |
| GoogLeNet linear probe (frozen) | 0.539 | **0.711 ± 0.099** (folds 0.57–0.85) |
| ResNet-18 fine-tuned (layer4 + head, augmentation, 6 epochs) — product model | **0.862**, 95 % CI [0.82, 0.91] | not run (6 × 11 min on CPU) |

Findings:

* **Single splits are unreliable here.** On an earlier split (whose validation set had only 4 disengaged
  frames) GoogLeNet scored 0.814. On the corrected stratified split it scored 0.539. Grouped CV over time
  blocks is the trustworthy comparison, and it shows the two backbones are statistically indistinguishable
  (difference 0.03, std about 0.09).
* Fine-tuning layer4 helps (0.862 vs 0.708 on the same split). Its CI comes from bootstrapping frames
  *within* one split, so it understates block-to-block variance. A realistic expectation is about 0.7–0.85.
* Errors are almost all between adjacent classes (engaged ↔ neutral). Engaged → disengaged confusions are rare.
* **Fundamental limit**: one person, one webcam, heuristic labels. This proves the pipeline works, not
  that it generalises across students, and fairness cannot be assessed on this data.

### 4.6 Attentive / distracted / inactive, and empty seats

The product reports each visible student as one of three states:

| state | rule | evidence |
|---|---|---|
| **inactive** | the alertness classifier (§4.5) says *disengaged*: eyes closing, drowsy, yawning | macro-F1 0.68–0.86 on held-out time blocks |
| **distracted** | alert, but the head is turned away: nose offset ≥ 0.30 × eye distance (about 30° yaw) | **not validated**: no openly licensed yaw ground truth was usable on this laptop (BIWI is about 5 GB with depth maps) |
| **attentive** | everything else | follows from the two rules above |
| *unclear* | face narrower than 30 px, not judged | see below |

* **Why "unclear" exists** (`python -m smartattend stage states`): on the classroom test set, faces under 30 px were
  called inactive **41 %** of the time, against 10 % at 30–40 px and 0 % above 60 px. Tiny, grainy faces look
  "drowsy" to a classifier trained on large webcam crops. They are now excluded from the percentages and counted separately.
* **Why "looking down" is not used**: a pitch proxy from the 5 YuNet landmarks has Spearman **−0.13** with the
  drowsiness dataset's per-frame head pitch, so it cannot measure pitch. Looking down is also often note-taking in class.
* **Empty seats**: YOLOv8-s now returns chairs, couches and benches from the same forward pass as people (no extra
  cost). A seat that no person overlaps is reported as empty. The timeline records occupied and empty seats per frame.

YuNet face detector + SFace 128-d embeddings (OpenCV ONNX models, pretrained, not trained here). Each
student is enrolled from 3–5 photos (mean embedding). Matching uses cosine similarity with a
one-to-one greedy assignment, so a student cannot be "present twice". A student is marked present after
being recognised in ≥ 2 sampled frames.

LFW, threshold 0.363:

* **Verification** (38,137 genuine / 60,000 impostor pairs): FAR 0.11 %, FRR 0.89 %. TAR is 99.6 / 99.1 /
  98.1 % at FAR 1e-2 / 1e-3 / 1e-4.
* **Simulated class sessions** (200 × 30 students, 5 absent, 3 non-enrolled strangers, 2 frames):
  **attendance recall 99.40 %, false-present rate 0.10 %, count error 0.16 students per session**.
* **Fairness** (groups from *predicted* LFW attributes):

| group | attendance recall | verification FRR | verification FAR (within-group impostors) |
|---|---|---|---|
| female / male | 99.55 / 99.36 % | 1.52 / 0.75 % | 0.20 / 0.13 % |
| asian / black / white | 100 / 100 / 99.30 % | 0.51 / 0.40 / 0.96 % | 0.17 / **0.56** / 0.16 % |
| **max gap** | **0.19 pp (gender), 0.70 pp (ethnicity)** | 0.77 pp / 0.56 pp | 0.07 pp / 0.40 pp |

  All gaps are within the 5-percentage-point target. But the female FRR is double the male FRR, and the black
  group's FAR is 3.5× the others'. Group sizes are small (217 black-identity probes) and the attributes are
  model predictions, so these gaps must be re-measured on the deploying school's own consented data.
* **Robustness** (clean enrolment, degraded probes, 2,426 images): identified correctly 99.8 % clean,
  99.7 % low light, 99.6 % small face (back row), 99.1 % JPEG-15, 98.7 % mask occlusion, 98.3 % blur, 97.8 % sensor noise.
* **Caveat**: LFW faces are frontal press photos of adults. Classroom CCTV (downward angle, motion,
  partial occlusion by other students) will be harder. The robustness tests narrow but do not close this gap.

## 6. Temporal and advanced models (step 9)

**Student-level LSTM trend (real frames).** Forecast the mean engagement score of the next 12 frames
(0.6 s) from the last 24 frames of eye/mouth aspect ratio, head roll/pitch and past score. Windows
never cross a split block (293 / 49 / 65 windows).

| model | test MAE |
|---|---|
| **last value (persistence)** | **0.063** |
| ridge on flattened window | 0.113 |
| window mean | 0.123 |
| LSTM, sigmoid cell activation | 0.111 ± 0.013 |
| LSTM, tanh cell activation (standard) | 0.155 ± 0.008 |
| LSTM, step activation (straight-through) | 0.174 ± 0.058 |

**Negative result:** at a 0.6-second horizon, engagement is almost constant and persistence wins.
293 overlapping windows from one person are far too few for an LSTM to learn more. The activation
comparison is still informative. The step cell trains (straight-through gradient) but is the least
accurate and by far the least stable across seeds (std 0.058 vs 0.008). Sigmoid beats tanh here because
its bounded, non-negative cell output suits a [0, 1] target with this little data.

**Class-level LSTM trend (simulated lectures).** 10-minute context, 5-minute horizon, observation noise
from the real classifier's confusion matrix: LSTM MAE 0.086 vs persistence 0.088 vs 3-min moving
average 0.094. That's a marginal gain. The value of the LSTM is the multi-step forecast shape, not accuracy.

**Transformer feature aggregator (simulated frames, real classifier outputs).** Per-student tokens
(class probabilities, face size, detection confidence, head turn) → 2-layer encoder with a CLS token,
permutation-invariant and padding-masked (both unit-tested). It predicts the class's engaged fraction:

| aggregator | MAE |
|---|---|
| mean of hard labels | 0.174 |
| mean of probabilities | 0.169 |
| DeepSets | 0.113 |
| **Transformer** | **0.111** |

Learned aggregation cuts error by about 36 %. CLS attention correlates with face size (r = 0.32): the model
learns to trust large, front-row faces whose classifier outputs are reliable. DeepSets is nearly as
good, so attention's main added value here is interpretability.

**Latent space and generative model.** Silhouette by engagement class: supervised ResNet-18 embedding
0.34 vs unsupervised convolutional VAE latent 0.01. The VAE (the generative component, alongside the RBM)
reconstructs and samples faces and supports an engaged → disengaged latent traversal. But without labels
its latent space organises by pose and lighting, not alertness (`reports/experiments/sequence/latent_space.png`).

## 7. Integrated product (step 10)

`python -m smartattend analyze --input lecture.mp4` or the Streamlit GUI (`streamlit run app/streamlit_app.py`).
Per frame: YOLOv8-s people → YuNet faces → SFace identity → ResNet-18 engagement on face crops →
face-to-person association → seat map → transformer class-engagement estimate. Per session:
attendance register (CSV), engagement timeline (CSV), annotated video, summary JSON.

**End-to-end demo** (`python scripts/make_demo.py`): 12 LFW identities enrolled from 3 photos each,
20-frame synthetic classroom video built from *different* photos, 2 students absent, 1 non-enrolled stranger.

* First run: 11/12 register entries correct. **Failure found:** the stranger was matched to the absent
  student Dennis_Kucinich in 7 of 20 frames (best similarity 0.451, above the 0.363 threshold). With the
  original "≥ 2 frames" rule, those occasional matches accumulated into a false "present".
* Fix: a student is **present** only if recognised in ≥ 2 frames **and** ≥ 50 % of processed frames.
  Between those bounds the status is **review** (late arrival or occasional look-alike), and the
  teacher decides. This rule is unit-tested.
* Second run: **12/12 correct**. Dennis_Kucinich is flagged for review, and every present student was recognised in 20/20 frames.

**Latency and memory** (40 classroom test images; laptop on battery, Windows capping the CPU at 1.2 GHz):
mean 1.8 s per frame (detection 585 ms, faces 719 ms, engagement 497 ms, seating 2 ms), p95 6.3 s, peak
RAM 735 MB. The p95 is driven by frames with many faces, because identification and engagement cost
grows linearly with the number of faces. On mains power the detector alone ran at 361 ms (benchmark, §4.2).
Attendance only needs one frame every few seconds, so this is workable for the register. It **misses**
the ≤ 500 ms per-frame target. Engineering options: batch all face crops into one engagement forward pass,
run engagement every N-th frame, and use YOLOv8-n (2.4× faster, −0.05 mAP).

## 8. Reproducibility and engineering

* `pyproject.toml` package with console script `smartattend`. Pinned seeds, YAML config, and every run logs its
  config and seed (`logs/`).
* Stages: `python -m smartattend stage {data,engagement,sequence,detection,seating,face,conv_viz,pipeline_latency}`.
* Tests: `python -m pytest` (gradient checks for every layer, behaviour tests for all learning rules,
  metric implementations cross-checked against `pycocotools`, leakage tests for splits,
  permutation-invariance of the aggregator, roster matching and session logic).

## 9. Acceptance against requirements

| requirement | target | result | status |
|---|---|---|---|
| attendance recall | ≥ 95 % | 99.40 % (LFW sessions); 10/10 present students in the demo | PASS (proxy data) |
| false-present rate | ≤ 2 % | 0.10 % (LFW); 0 in the demo after the review rule | PASS (proxy data) |
| stranger false-accept | ≤ 1 % | 0.11 % FAR | PASS (proxy data) |
| fairness gap, attendance recall | ≤ 5 pp | 0.19 pp gender, 0.70 pp ethnicity | PARTIAL: met, but predicted groups, small n, unequal FRR/FAR |
| person detection mAP@0.5 | ≥ 0.80 | 0.816 (YOLOv8-s) | PASS |
| count error | ≤ 1 person | 0.99 on fully annotated images (1.70 incl. crowd-annotated images) | PASS (narrow) |
| engagement macro-F1 | ≥ 0.65 | 0.68–0.71 in 5-fold CV (probes); 0.862 on one split (fine-tuned) | PARTIAL: one subject, weak labels |
| CPU latency p95 | ≤ 500 ms | 6.3 s (battery, 1.2 GHz cap, many faces) | FAIL |
| memory | ≤ 2 GB | 735 MB peak | PASS |
| reproducible build and tests | one command | `pip install -e .`, `python -m pytest` (53 tests), `stage all` | PASS |

## 10. Deployment recommendation (go / no-go)

**GO for a supervised one-term pilot of face-recognition attendance. NO-GO for unsupervised rollout and
for any individual-level use of engagement.**

| component | decision | evidence | conditions |
|---|---|---|---|
| face attendance | **GO (pilot)** | 99.4 % recall, 0.1 % false-present, gaps < 1 pp on LFW; 12/12 in the demo | opt-in consent; teacher confirms each register; "review" list checked; measure recall and per-group FRR/FAR on the pilot class |
| head count | **GO (advisory)** | count error 0.99 on fully annotated images | camera high at the front, back row ≤ 8 m, so people are > 32 px; report as approximate |
| seat map | **CONDITIONAL** | heuristic unvalidated; merges rows in dense photos | replace with per-room seat calibration before relying on seat IDs |
| class engagement trend | **CONDITIONAL (aggregate only)** | macro-F1 0.68–0.71 CV on one subject | class-level only, labelled "alertness estimate"; never shown per student |
| individual engagement | **NO-GO** | one-subject training data; fairness untestable | needs a consented, multi-student, human-labelled dataset |
| real-time video | **NO-GO on this laptop** | p95 6.3 s/frame on battery | sample 1 frame per 2–5 s for attendance; batch crops; or add a GPU (then Faster R-CNN becomes viable) |

Pilot exit criteria (to move to GO for rollout): recall ≥ 95 % and false-present ≤ 2 % against teacher-confirmed
registers over ≥ 20 lectures; per-group FRR gap ≤ 2 pp on the class's own data; zero complaints unresolved
through the opt-out process.

## 11. Responsible-use limits
* **Consent and opt-out**: only enrolled, consenting students are ever identified. Store embeddings, not photos.
* **Human in the loop**: the register is a suggestion the teacher confirms.
* **Engagement is aggregate and about alertness**: it must not be used to grade, rank or discipline individuals.
* **Monitor fairness in production**: per-group recall and false-match rates on the school's own data, each term.
* **Not for**: exam proctoring, behaviour scoring, sharing with third parties, or any use without notice to students.

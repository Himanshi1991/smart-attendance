# Requirement 8 — Benchmark: R-CNN, Fast R-CNN, Faster R-CNN, YOLO, ResNet, GoogLeNet, RetinaNet

## Why this module exists

Requirement 7 picked a detector for the app. This module puts **every detector family side by side** on the same task
(find the students in the Kaggle **test classroom**: 85 frames, 1,159 students) with the same metric code, so the
choice is justified by numbers. It also shows **why the field moved** from R-CNN (2014) to Fast R-CNN (2015), Faster
R-CNN (2015), and one-stage detectors (YOLO, RetinaNet).

R-CNN and Fast R-CNN did not exist in the project; they are **implemented here from their papers**
(`src/smartattend/vision/rcnn.py`).

Code: `vision/rcnn.py`, `vision/benchmark8.py`, `scripts/selective_search.py` · tests: `tests/test_rcnn.py` (9) ·
Colab: `notebooks/08_benchmark_colab.ipynb` (**GPU recommended**).

## Run it

```bash
python -I scripts/selective_search.py --cv2-path models/tools/opencv_contrib   # region proposals, once (about 13 min)
python -m smartattend benchmark --tv-epochs 2                                  # about 3 hours on a laptop CPU the first time
python -m pytest tests/test_rcnn.py
```

Everything is cached (features, weights, predictions), so an interrupted run resumes, and a second run takes about a
minute. Results: `reports/eval/benchmark.json`, figure: `reports/figures/vision/benchmark_detectors.png`.

## The five families in one picture

| family | how it finds objects | who proposes regions | CNN passes per image |
|---|---|---|---|
| **R-CNN** (2014) | crop each proposed region, warp it to a square, run a CNN on **each crop**, classify with a linear model | selective search (hand-made, no learning) | about 1,000 |
| **Fast R-CNN** (2015) | run the CNN **once** on the image; cut each region out of the shared feature map (RoI pooling); one head predicts class **and** box together | selective search | 1 |
| **Faster R-CNN** (2015) | as Fast R-CNN, but a small network (RPN) **learns** to propose regions from the same features | learned (RPN) | 1 |
| **RetinaNet** (2017) | one stage: predicts a class and a box at every position of a feature pyramid; *focal loss* stops the many easy background positions from swamping training | none | 1 |
| **YOLO** (2016–) | one stage: predicts boxes directly from a grid in a single fast pass | none | 1 |

**ResNet** and **GoogLeNet** are not detectors; they are *backbones* (the CNN inside). Here they are compared twice:
as the CNN inside R-CNN, and as attentive / distracted / inactive classifiers (Requirement 7).

## How R-CNN and Fast R-CNN were built

**Region proposals.** Selective search groups similar-looking pixels into about 1,000 candidate boxes per frame
(`scripts/selective_search.py`, 1.56 s per frame). It lives in OpenCV's *contrib* build, which is loaded from a
separate folder so the project's own OpenCV is untouched.

```
selective-search proposals cover 62% of students at IoU 0.5 and 12% at IoU 0.7
```

This is a **hard limit**: R-CNN and Fast R-CNN can never find a student the proposals missed.

**R-CNN** (`RCNN` class):
```python
crops    = warp_crops(image, proposals, size=112)        # every region stretched to 112x112
features = frozen_imagenet_cnn(crops)                    # ResNet18 (512 numbers) or GoogLeNet (1024), one pass PER region
score    = logistic_regression(features)                 # person vs background  (the paper used linear SVMs)
box      = decode(proposal, ridge_regression(features))  # box regression: shift and resize the proposal
keep     = nms(boxes, scores)                            # drop overlapping duplicates
```
Training regions: the labelled boxes (plus slightly jittered copies) and proposals with IoU ≥ 0.6 are positives;
proposals with IoU < 0.3 are background (13,888 regions, 3,285 positive).

**Fast R-CNN** (`FastRCNN` class):
```python
fmap  = resnet18_up_to_layer3(image)                     # ONE pass; stride 16, 256 channels
rois  = roi_align(fmap, proposals, output_size=7)        # cut every proposal out of the shared feature map
h     = fc(1024) -> fc(1024)                             # small head, shared by both outputs
loss  = cross_entropy(cls(h), label) + smooth_l1(box(h), target)    # multi-task loss: class and box learned together
```

**Simplifications (stated honestly):** the ImageNet backbones are frozen (the papers fine-tune them); R-CNN uses
logistic regression where the paper uses SVMs (both are linear classifiers on the same features); 1,000 proposals
per frame where the papers use 2,000.

**Faster R-CNN and RetinaNet** are torchvision's implementations, tested twice: as downloaded (trained on COCO) and
**fine-tuned for 2 epochs** on the Kaggle training classrooms (ResNet50 body frozen; FPN, RPN and heads trained).
YOLOv8n was fine-tuned in Requirement 7.

## Results (test classroom; thresholds and settings chosen on validation)

```
model                       Kaggle  mAP50  mAP50-95  recall  count MAE   ms/img  params M
R-CNN (ResNet18)               yes  0.163     0.036   0.233       3.81    11734      11.2
R-CNN (GoogLeNet)              yes  0.114     0.024   0.080       9.69    17744       5.6
Fast R-CNN (ResNet18)          yes  0.198     0.045   0.321       2.46      833      16.7
Faster R-CNN (COCO)             no  0.322     0.098   0.538       1.73     2129      41.8
Faster R-CNN (fine-tuned)      yes  0.616     0.221   0.694       1.25     2641      41.8
RetinaNet (COCO)                no  0.258     0.077   0.374       2.02     1836      34.0
RetinaNet (fine-tuned)         yes  0.620     0.222   0.707       1.47     2380      34.0
YOLOv8n (COCO)                  no  0.328     0.111   0.453       1.92       79       3.2
YOLOv8s (COCO)                  no  0.321     0.112   0.489       1.33      179      11.2
YOLOv8n (fine-tuned)           yes  0.608     0.230   0.619       1.46       76       3.2
```

- *Kaggle* = trained or fine-tuned on the Kaggle training classrooms. *recall* = share of students found at the
  counting threshold. *count MAE* = average error in the number of students per frame (about 14 students per frame).
- *ms/img* = median on 5 frames with all models run in turn on the same frame (laptop CPU on battery at 1.2 GHz,
  so a worst case). **R-CNN and Fast R-CNN also need 1,560 ms of selective search per frame, not included.**

### What the numbers say

1. **The history of detection is visible in one column.** R-CNN 11.7 s → Fast R-CNN 0.83 s per frame: **14× faster with
   the same proposals**, just by running the CNN once and sharing the features. That is exactly Fast R-CNN's contribution.
2. **Hand-made proposals are the bottleneck.** R-CNN and Fast R-CNN reach only mAP50 0.16–0.20. Selective search
   finds 62 % of students at all and places only 12 % of them accurately (IoU 0.7); a frozen classifier cannot repair
   that. **Faster R-CNN, which *learns* its proposals, reaches 0.616** once fine-tuned. That is its contribution.
3. **Fine-tuning matters more than the choice of modern detector.** As downloaded, Faster R-CNN, RetinaNet and YOLO
   score 0.26–0.33. Fine-tuned on 217 classroom frames, all three reach **0.61–0.62**: a statistical tie.
4. **Speed then decides.** At equal accuracy, fine-tuned YOLOv8n needs **76 ms** per frame, Faster R-CNN 2,641 ms and
   RetinaNet 2,380 ms (**about 30× slower**), with 13 and 11 times more parameters.
5. **Counting:** fine-tuned Faster R-CNN is best (off by 1.25 students per frame), then YOLOv8s (1.33), fine-tuned
   YOLOv8n (1.46) and fine-tuned RetinaNet (1.47). R-CNN is far worse (3.8 and 9.7).
6. **ResNet18 vs GoogLeNet.** Inside R-CNN, ResNet18 features are better (0.163 vs 0.114) and faster (11.7 s vs
   17.7 s). As state classifiers (Requirement 7) they tie: macro-F1 0.459 [0.430–0.491] vs 0.444 [0.415–0.474], with
   ResNet18 twice as fast (16 vs 33 ms per crop) and GoogLeNet half the size (5.6 M vs 11.2 M parameters).

### Decision for the app

**Fine-tuned YOLOv8n** (`models/yolo/yolov8n-kaggle.pt`): tied for the best accuracy, 30× faster than the
alternatives, smallest model. If a classroom needs the most exact head count and can wait 2–3 s per frame,
fine-tuned Faster R-CNN is the better counter.

## Honest notes on the method

- **R-CNN was tuned before being judged.** Its first scores used settings picked by hand. Its classifier strength (C),
  box-regression penalty (alpha) and NMS overlap are now chosen on validation frames from a 60-setting grid
  (validation mAP50 from 0.13 for the worst setting to 0.22 for the best). The grid had to be widened once because
  the best setting sat on its edge.
- **R-CNN's counting first looked absurd (20 students too many per frame).** Its scores bunch up near 1.0, and the
  threshold search stopped at 0.85. The search now goes to 0.999 for every model (R-CNN picks 0.98).
  R-CNN (GoogLeNet) still under-counts badly (recall 0.08): a threshold of 0.95 that suited the validation classroom
  is too strict for the test classroom. A score that needs such a fragile threshold is a weakness in itself.
- **A real bug found:** Fast R-CNN's loss became `nan` in the full run. Three training frames contain no students;
  with "background = IoU between 0.1 and 0.5" they produced an **empty batch**, whose loss is NaN, and one NaN update
  ruins every weight. Fixed (frames with no students supply plain background regions; empty batches are skipped;
  a zero-size label is ignored) with three regression tests.
- **Timing was measured three times before it was trusted.** Long runs on a battery-powered laptop drift (the same
  model measured 16.6 s and 33.9 s on different nights). The reported numbers come from running all ten models in
  turn on the same five frames.
- **The simplifications favour neither side unfairly, but they matter:** with fine-tuned backbones (as in the papers)
  R-CNN and Fast R-CNN would score higher. They would still be capped by the proposals (62 %) and still be the slowest.
- **Thresholds were chosen on every 4th validation frame (45 frames)** to bound R-CNN's CPU time, so YOLO's counting
  errors differ slightly from Requirement 7 (which used all 179).
- **One test classroom.** These rankings are for this room and camera; the earlier COCO benchmark
  (`reports/RESULTS.md`) gives the complementary view on varied photos.

## Checklist
- [x] R-CNN implemented (two backbones: ResNet18, GoogLeNet), tuned on validation
- [x] Fast R-CNN implemented (shared features, RoIAlign, multi-task loss)
- [x] Faster R-CNN and RetinaNet: COCO-pretrained and fine-tuned; YOLOv8n/s pretrained and YOLOv8n fine-tuned
- [x] ResNet vs GoogLeNet as detection backbones and as classifiers
- [x] same test set, same metric code (mAP50, mAP50-95, recall, count error), interleaved timing, parameter counts
- [x] proposal-recall ceiling measured; accuracy-vs-speed figure
- [x] 9 tests in `tests/test_rcnn.py`, including 3 regression tests for the bug found

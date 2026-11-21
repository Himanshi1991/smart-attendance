# Model card — Smart Attendance and Classroom Engagement Analyzer

All numbers come from `reports/RESULTS.md` (auto-generated). Hardware: Intel i3-1215U, 8 GB RAM, CPU only.

## Components

| component | model | weights | trained here? |
|---|---|---|---|
| person detection | YOLOv8-s (Ultralytics) | `models/yolo/yolov8s.pt` (COCO) | no, selected by benchmark |
| face detection | YuNet 2023-03 (OpenCV zoo) | `models/face_detection_yunet_2023mar.onnx` | no |
| face identity | SFace 2021-12 (OpenCV zoo), 128-d embedding | `models/face_recognition_sface_2021dec.onnx` | no; roster built from enrolment photos |
| engagement (alertness) | ResNet-18 (ImageNet), layer4 + head fine-tuned, 6 epochs | `models/engagement_resnet18.pt` | **yes**, drowsiness dataset |
| instance segmentation | YOLOv8-n-seg (COCO) | `models/yolo/yolov8n-seg.pt` | no |
| class aggregation | Transformer encoder (2 layers, 4 heads, d=32, CLS token) | `reports/experiments/sequence/transformer_aggregator.pt` | **yes**, simulated frames with real classifier outputs |
| class trend | LSTM (32 hidden) | `reports/experiments/sequence/lstm_trend_class.pt` | **yes**, simulated lectures |
| experimental only | YOLOv8-n fine-tuned on COCO-Classroom (6 epochs, frozen backbone) | `models/yolo/yolov8n-classroom.pt` | yes; did **not** beat the stock model, so not used |

## Intended use
* Suggesting an attendance register for **enrolled, consenting** students, for a teacher to confirm.
* Head count and approximate seat occupancy for a fixed classroom camera.
* **Class-level** alertness trends over a lecture, as a teaching aid.

## Out-of-scope uses
Grading, ranking or disciplining individual students; exam proctoring; identifying people who have not
consented; behaviour scoring; sharing identities or engagement data with third parties; any deployment
without notice to students. Engagement is **alertness from eyes, mouth and head pose**, not attention to
content or learning.

## Performance

| task | data | metric | result |
|---|---|---|---|
| attendance | LFW, 200 simulated sessions × 30 students | recall / false-present | 99.40 % / 0.10 % |
| face verification | LFW, 38k genuine / 60k impostor pairs | TAR @ FAR 0.1 % | 99.1 % |
| person detection | COCO-Classroom test (104 images) | mAP@0.5 / mAP@[.5:.95] | 0.816 / 0.599 |
| counting | 85 fully annotated test images | MAE (people per image) | 0.99 |
| person segmentation | COCO-Classroom test | IoU (instance masks) | 0.738 |
| engagement | drowsiness test split (5-s blocks) | macro-F1 (95 % CI) | 0.862 [0.82, 0.91] |
| engagement backbones | 5-fold grouped CV over time blocks | macro-F1 mean ± std | see `reports/RESULTS.md` §3 |
| class aggregation | simulated frames | engaged-fraction MAE | 0.111 (naive mean 0.174) |

## Fairness
* Face attendance recall by predicted gender: female 99.55 %, male 99.36 % (gap 0.19 pp). By predicted
  ethnicity: asian 100 %, black 100 %, white 99.30 % (gap 0.70 pp). All within the 5 pp target.
* Verification error rates are **not** equal. Female FRR is 1.52 % vs 0.75 % for male. The black group's FAR (within-group
  impostors) is 0.56 % vs 0.16–0.17 % for the other groups. Group labels are predicted, not self-identified, and the
  black group has only 217 probes.
* **Back-row students**: person-detection recall is 0.31 for people under 32 px vs 0.95 for large people.
  Small faces are also the main source of engagement errors (the transformer learns to down-weight them).
* Engagement fairness **cannot be measured**: the training data contains one person.

## Robustness (face identification, clean enrolment, degraded probes)
clean 99.8 %, low light 99.7 %, small face 99.6 %, JPEG-15 99.1 %, mask occlusion 98.7 %, blur 98.3 %, noise 97.8 %.
Detection under motion blur drops to mAP@0.5 0.50 for YOLOv8-s (0.83 clean); Faster R-CNN keeps 0.63.

## Limitations
1. **Proxy data**: no footage from a real lecture hall was used. LFW faces are frontal press photos, and COCO
   scenes are eye-level photos of 2–13 people, not 40-seat rooms from a ceiling camera.
2. **Engagement model**: one subject, one webcam, heuristic labels from eye/mouth aspect ratios. Results
   do not transfer to other students without new consented, human-labelled data.
3. **Temporal models**: the student-level LSTM does **not** beat a last-value baseline at a 0.6-second horizon.
   The class-level LSTM and the transformer are trained on simulated data.
4. **Seat map**: the row clustering heuristic has no ground truth and merges rows in dense tiered photos.
5. **Latency**: see `reports/eval/pipeline_latency.json`. Many faces in one frame push p95 latency above the 500 ms target.

## Ethical considerations
Facial recognition of minors is high-risk under most data-protection regimes. Deployment requires opt-in
consent (guardian consent for minors), a data-protection impact assessment, embeddings-only storage,
deletion on request, human confirmation of every register, and a per-term fairness audit on the school's own data.

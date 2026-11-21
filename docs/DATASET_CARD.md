# Dataset card

All data is public and downloaded reproducibly with `python -m smartattend stage data`
(code: `src/smartattend/data/`). No data about real students at the deploying school was collected.

## 0. Kaggle Classroom Student Engagement (real classrooms: counting + attentive/distracted/inactive)

| | |
|---|---|
| Source | https://www.kaggle.com/datasets/mustafaxgm/classroom-student-engagement-dataset (Ghulam Mustafa, Lahore Garrison University final-year project; annotated in Roboflow) |
| Licence | **CC BY-NC 4.0** on Kaggle: non-commercial use with attribution. The bundled Roboflow readme says "Private"; the Kaggle release licence governs the published copy. |
| Content | 481 frames from 11 classroom videos; 4,603 student boxes with 8 action labels (look_forward 2,384, turn_head 715, using_device 522, read 344, write 320, sleep 232, stand 60, handrise 26). |
| Labels → states | look_forward, read, write, handrise, stand → **attentive**; turn_head, using_device → **distracted**; sleep → **inactive**. |
| De-duplication | 64-bit difference hash, Hamming ≤ 3: 50 near-duplicate pairs, all consecutive frames *within* one recording group (kept, since they share a split); 0 across groups. |
| Leakage-safe split | By **recording group**: videos showing the same students in the same room are one group (checked visually: e.g. `taimoor-class-2/3/5` share students). Train = owais + seminars + misc (217 images, 2,277 students); val = taimoor-class (179 / 1,167); test = taimoor-zaeem, a different room (85 / 1,159). |
| Shift | Test is 56 % distracted vs 20 % in train: a deliberately hard, honest test on an unseen classroom. |
| Exports | `data/processed/kaggle_classroom/`: `yolo_person/` (counting), `yolo_states/` (3-class detector), `crops/{split}/{state}/` (ResNet/GoogLeNet), `provenance.json`. Reproduce: `python -m smartattend stage kaggle_data`. Colab: `notebooks/02_datasets_colab.ipynb`. |
| Known gaps | 11 videos from one university, mostly young men: no demographic balance, so fairness on this data cannot be assessed. Labels come from one annotator team. |

## 1. COCO-Classroom (person detection, counting, seating, segmentation)

| | |
|---|---|
| Source | COCO 2017 **val** images + instance and caption annotations (`images.cocodataset.org`) |
| Licence | Annotations CC BY 4.0. Images: individual Flickr licences, recorded per image in `data/processed/coco_classroom/{split}.json` (BY-NC-ND 170, BY-NC-SA 174, BY 97, BY-NC 75, BY-SA 47, BY-ND 34, no known restrictions 1). Research/educational use only; images are not redistributed in this repo. |
| Why val2017 only | All pretrained detectors (YOLOv8, torchvision Faster R-CNN / RetinaNet) were trained on COCO **train**2017, so val2017 is unseen by them: no train/test leakage in the benchmark. It also avoids loading the 450 MB train annotation file on an 8 GB laptop. |
| Selection | **Classroom-like** (val/test): a human caption mentions a classroom scene (students, class, lecture, audience, seminar, meeting, desk, …) with ≥ 2 people, **or** a seated indoor group (≥ 3 people + chair + desk/laptop/book/screen). **Train** (YOLO fine-tune only): other COCO val images with ≥ 4 people. |
| Labels | COCO human-annotated person boxes and polygon masks (`iscrowd=0`). Crowd regions are ignored, as in COCO. |
| De-duplication | 64-bit difference hash; images within Hamming distance 4 of an already-kept image are removed (test > val > train priority). 0 removed. |
| Splits | Disjoint by image, seed 0: **train 450 images / 3,916 people**, **val 44 / 312**, **test 104 / 767** (7.4 people per image, max 13). Size mix on test: 13.6 % small (< 32² px), 42.5 % medium, 43.9 % large. |
| Known gaps | Not real lecture-hall CCTV: views are mostly eye-level photos of 2–13 people, not 40-seat rooms from a ceiling camera. Results are evidence for the method, not a guarantee for a specific room. |

## 2. LFW + attributes (face recognition, attendance simulation, fairness)

| | |
|---|---|
| Source | Labeled Faces in the Wild (13,233 images, 5,749 people) via the Hugging Face mirror `marcelohaps/lfw`; attribute scores from Columbia `lfw_attributes.txt` (Kumar et al., ICCV 2009). |
| Licence | Research use (LFW has no formal licence; images are news photos of public figures). |
| Use | Verification (genuine/impostor pairs), simulated class sessions (30 enrolled students, 5 absent, 3 strangers; 200 sessions), robustness corruptions, fairness by group. |
| Groups | gender = sign of the *Male* attribute score; ethnicity = argmax of the *Asian/White/Black* scores. **These are classifier predictions, not self-identified identity**, and are imbalanced (male 10,181 / female 2,962; white 10,956 / asian 1,425 / black 762). |
| Leakage | Enrolment and probe images of a person are always different photos. The face models (YuNet, SFace) are pretrained by OpenCV and not trained here. |
| Known gaps | Celebrities, frontal press photos, adults: easier than classroom CCTV of students. The robustness tests (blur, low light, small faces, JPEG, mask occlusion) partly address this. |

## 3. Drowsiness faces → engagement (alertness) classification and temporal trends

| | |
|---|---|
| Source | `notgoodkeeper/cnn-based-drowsiness-detection-data` (Hugging Face), 2,376 webcam face crops (grayscale, CLAHE, white background) with eye/mouth aspect ratio and head roll/pitch per frame. |
| Licence | MIT. |
| Labels | Upstream 0–4 drowsiness class from a hand-tuned heuristic (weak labels, not human-annotated). Mapped to **engaged** (0) / **neutral** (1–2) / **disengaged** (3–4). |
| Temporal structure | About 3 minutes of 20 fps video in 2 recording sessions. The timeline comes from the capture timestamps in the file names. |
| Leakage-safe split | The upstream random split would put neighbouring near-identical frames into train and test. We re-split by **5-second time blocks**, assign whole blocks to splits (stratified by majority label), and drop a **0.5 s buffer** at every block edge. Blocks are stratified per class so every split contains every class. Result (seed 42): train 1,293 (658/491/144), val 213 (49/115/49), test 272 (92/140/40). Because a few blocks dominate any single split, backbones are also compared with **5-fold grouped cross-validation over blocks**. |
| Known gaps | **One person, one camera, weak labels.** It measures *alertness* (eyes and mouth), not cognitive engagement. Results do not transfer to other students without new data, and fairness cannot be measured on it. This is the main limitation of the engagement component. |

## 4. Synthetic data (clearly marked wherever used)

* `data/synthetic.py`: engagement features for the NumPy fundamentals module (steps 4–5).
* Class-level lecture sessions for the class-level LSTM forecast. Observation noise comes from the **real** classifier's confusion matrix.
* Per-frame student sets for the transformer aggregator. Per-student probabilities are **bootstrapped from the real classifier's test predictions**.

Synthetic results show that a method works. They are never reported as real-world accuracy.

## Ethics and consent
Deployment requires opt-in consent from students (and guardians for minors), a data-protection impact
assessment, and storing only embeddings. See `docs/REQUIREMENTS.md` → *Responsible-use limits*.

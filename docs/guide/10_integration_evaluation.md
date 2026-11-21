# Requirement 10 — Integration and evaluation: mAP, count error, macro-F1, fairness, latency, robustness, error analysis

## What this module is

The last step puts the pieces together and asks the questions a school would ask before using the system:
**How accurate is it? Who does it serve worse? Does it survive a dim or noisy camera? How fast is it? Where exactly
does it fail?** Every answer is measured, and every requirement from step 1 gets a pass or a fail.

**Pipeline under test** (both parts chosen on validation in Requirements 7–8):

```
frame ─► fine-tuned YOLOv8n ─► student boxes ─► count
                              └─► ResNet18 ─► attentive / distracted / inactive ─► class share ─► trend
(full app adds: YuNet + SFace face attendance, seat map with YOLOv8s chairs)
```

**Test data:** the Kaggle test classroom, 85 frames with 1,159 labelled students, never used for training or for any
choice. Attendance and its fairness come from the LFW evaluation done earlier.

Code: `src/smartattend/eval/final_eval.py`, `requirements.py`, `scripts/build_report.py`, `scripts/build_deck.js` ·
tests: `tests/test_final_eval.py` (11) · Colab: `notebooks/10_integration_evaluation_colab.ipynb`.

## Run it

```bash
python -m smartattend evaluate                  # robustness, fairness, errors, latency, scorecard (about 10 min)
python -m smartattend stage pipeline_latency    # app latency for the acceptance check
python -m smartattend acceptance                # every requirement against its target
python scripts/build_report.py                  # reports/RESULTS.md: every table, generated from the result files
python -m pytest -q                             # 125 tests
```

Results: `reports/eval/final_evaluation.json`, `reports/eval/acceptance.json`, `reports/RESULTS.md`,
`docs/presentation.pptx` (21 slides). Figures: `reports/figures/eval/`.

## The metrics, in plain words

| metric | question it answers | how to read it |
|---|---|---|
| **mAP@0.5** | are the boxes on the students? | 0 to 1; a box counts if it overlaps the label by at least half |
| **count error (MAE)** | is the head count right? | average number of students the count is off by, per frame |
| **macro-F1** | are the three states told apart? | average of the F1 of each class, so the rare class counts as much as the common one |
| **fairness gap** | is one group served worse? | best group minus worst group; 0 is perfectly equal |
| **latency (p95)** | is it fast enough? | 95 % of frames are faster than this |
| **robustness** | does a bad camera break it? | the same metrics on deliberately degraded frames |

## Results

### Robustness: the same 85 frames, degraded

```
condition                       mAP50  count MAE  state macro-F1
clean                           0.608       1.46           0.467
low light (30% brightness)      0.608       1.55           0.291
motion blur (15 px)             0.589       1.92           0.327
sensor noise (sigma 25)         0.519       4.26           0.183
heavy JPEG (quality 15)         0.606       2.01           0.304
half resolution                 0.606       1.48           0.489
```

- **The detector is robust**: low light, heavy compression and half resolution cost almost nothing. **Sensor noise is
  its weak spot** (mAP 0.52, count off by 4.3): the grain creates extra boxes.
- **The state classifier is fragile**: macro-F1 falls from 0.47 to 0.29 in low light and 0.18 with noise. It was
  fine-tuned on clean frames with only flip and brightness augmentation.
- **Half resolution does not hurt** (the classifier works on 160-pixel crops anyway), so a cheaper camera is not the
  problem; a dim, noisy one is.

(The clean state score here is 0.467; Requirement 7 reported 0.459 on the same students. The crops there had been
saved as JPEG files first; here they are cut straight from the frame.)

### Fairness: who is served worse?

The classroom data has no demographic labels, so fairness is measured on what it does have: **where the student sits**
(size in the image) and **what the student is doing**.

```
students found by the detector, by size:   small (back rows) 0.46, medium 0.60, large (front rows) 0.80   (gap 0.35)
students found, by their true state:       attentive 0.57, distracted 0.67, inactive 0.52                 (gap 0.15)
state classified correctly, by size:       small 0.61, medium 0.63, large 0.58                            (gap 0.05)
```

- **Back-row students are the under-served group**: found 46 % of the time, against 80 % in the front rows.
- **Sleeping (inactive) students are found least often** (52 %): a head on the desk looks less like a person.
- Once a student *is* found, the state classifier treats sizes almost equally (gap 0.05).
- **Attendance by face** (LFW, earlier): recall gap between gender and ethnicity groups 0.7 percentage points.

### Error analysis

```
missed students: 441 of 1159 (38%); by size: small 210, medium 154, large 77
false boxes: duplicate of a found student 8, poorly placed box on a student 308, background 39
largest state confusions: distracted -> attentive 300 (46% of distracted); attentive -> distracted 54 (13%); inactive -> distracted 54 (59% of inactive)
```

- **Why the count is right (off by 1.5) while 38 % of students are "missed":** 308 of the 355 false boxes sit **on a
  student**, just not well enough aligned to count (overlap between 10 % and 50 %). `final_error_gallery.png` shows
  why: in many frames the label is a small **head** box and the detector draws an **upper-body** box on the same
  student. The dataset mixes both labelling styles, so part of the "error" is in the labels.
- **Only 39 boxes are on empty background** (less than half a box per frame), and duplicates are rare (8).
- **The state classifier's main mistake is calling distracted students attentive** (46 % of them), which is why the
  predicted attentive share is too high (Requirement 7). **Inactive students are mostly called distracted** (59 %).

### Latency (laptop CPU, i3-1215U, on battery at 1.2 GHz: worst case)

```
classroom pipeline: detect 72 ms + states 210 ms = 283 ms median, 366 ms p95 (3.5 frames/s)
full app (yolov8s):         1168 ms median, 1294 ms p95 (detect 153, faces 698, engagement 318); peak memory 762 MB
full app (yolov8n-kaggle):  1125 ms median, 1303 ms p95; chairs/benches detected per frame 0.0 (yolov8s: 2.4)
```

- The **classroom pipeline is within the 500 ms target** even on battery (366 ms at p95).
- The **full app is not** (1.3 s here; 2.0 s at p95 on the varied photos used by the acceptance check). **Face
  detection and recognition take about 700 ms**, more than half the time, because the faces step runs on the full
  1920×1080 frame and then once per face.
- **How to meet the target:** attendance does not need every frame. Run faces once every few seconds and the counting
  and state steps on the frames in between; plug the laptop in (the CPU is capped at 1.2 GHz on battery).
- **Integration finding:** the fine-tuned YOLOv8n **no longer detects chairs** (0 per frame), because it was
  fine-tuned on people only. So the app keeps **YOLOv8s as its default** (it needs chairs for the seat map), and the
  fine-tuned model is used with `--detector yolov8n-kaggle` when counting and states matter most.

## Acceptance: every requirement against its target

```
ID  Stakeholder  Metric                                        Target         Measured  Status
R1  Teacher      attendance recall                             >= 0.95        0.9940    PASS
R2  Teacher      false-present rate                            <= 0.02        0.0010    PASS
R3  Admin        stranger false-accept rate (FAR)              <= 0.01        0.0011    PASS
R4  Students     max attendance-recall gap (gender, ethnicity) <= 0.05        0.0070    PASS
R5  Teacher      person detection mAP@0.5                      >= 0.8         0.8164    PASS
R6  Teacher      count error (MAE, fully annotated images)     <= 1 people    0.9882    PASS
R7  Teacher      engagement macro-F1 (5-fold CV)               >= 0.65        0.7109    PASS
R8  IT           CPU latency per frame, p95                    <= 500 ms      2015      FAIL
R9  IT           peak memory                                   <= 2000 MB     793       PASS
R10 Teacher      count error, classroom video (MAE)            <= 2 students  1.4588    PASS
R11 Teacher      state macro-F1, classroom video               >= 0.6         0.4671    FAIL
R12 Students     detection recall gap, small vs large          <= 0.15        0.3451    FAIL
R13 IT           mAP50 lost in low light                       <= 0.05        0.0000    PASS
10 pass, 3 fail
```

R10–R13 were added at this step, when real classroom video with per-student labels was available; their targets were
written down before the final evaluation ran.

**The three fails, and what each one means:**
- **R8, latency:** the full app with face recognition on every frame is too slow on this laptop on battery. Fixable by
  running faces less often (see above); not yet done.
- **R11, per-student state in a new classroom (0.47 against 0.60):** not good enough. R7 passes on one person's frames
  (0.71), which shows how misleading a single-person dataset is. Needs many more labelled classroom videos.
- **R12, back rows:** small students are found far less often. A camera mounted high at the front, or a higher input
  resolution, is the practical fix.

## Decision

| part | decision | evidence |
|---|---|---|
| Face attendance | **GO for a supervised pilot** (teacher confirms every register) | R1–R4 pass; recall 99.4 %, false-present 0.1 % |
| Counting students | **GO with conditions** | off by 1.5 students per frame; robust to light and compression; back rows under-counted; needs fine-tuning per site |
| Class attention trend | **Direction only** | follows ups and downs (correlation 0.61–0.64); the percentage is not reliable |
| Per-student attentive / distracted / inactive | **NO-GO** | macro-F1 0.47 on a new classroom; fragile in low light; never to be used on individuals |
| Seat map | **Unvalidated** | no seat labels exist; needs per-room calibration |

Limits that do not change: opt-in consent, embeddings instead of photos, no grading or discipline from engagement,
class-level reporting only (`docs/MODEL_CARD.md`).

## What was integrated in this step

- one command for the final evaluation, and one scorecard pulling numbers from every earlier stage
- acceptance check extended from 9 to 13 requirements
- `reports/RESULTS.md` regenerated with sections for the real-classroom benchmark, the state models, the sequence
  models, the final evaluation and the acceptance scorecard
- `docs/presentation.pptx` rebuilt from the result files (21 slides; three new slides on real classroom video, the
  scorecard with all 13 requirements, the recommendation updated)
- the stale app latency (6.3 s, measured while another job was running) re-measured: 2.0 s at p95

## Checklist
- [x] mAP, count error, macro-F1 on real classroom video and on the earlier public datasets, in one scorecard
- [x] fairness: by seat position, by student state, and (attendance) by gender and ethnicity
- [x] latency: median, p95, per stage, memory; honest about battery
- [x] robustness: five degradations of the same test frames
- [x] error analysis: missed students, kinds of false boxes, state confusions, a gallery of the worst frames
- [x] acceptance check (10 of 13 pass), results report and presentation regenerated from the result files
- [x] 11 new tests; 125 in total

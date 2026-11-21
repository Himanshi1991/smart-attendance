# User guide

For teachers and school IT staff. Everything runs on an ordinary Windows/Linux PC without a GPU.

## 1. Install (once, by IT)

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt ultralytics streamlit
pip install -e .
python -m pytest                       # all tests should pass
```

The face models (about 39 MB) download automatically on first use. The trained engagement model and
fine-tuned YOLO weights are in `models/`. To rebuild them from scratch, see section 6.

## 2. Enrol the class (once per term)

1. Collect **written consent**. Students who do not consent are simply not enrolled; they appear as
   "unknown" and are never identified.
2. For every consenting student, make a folder named with their student ID and put 3–5 clear,
   front-facing photos in it:
   ```
   data/roster/22CS001/1.jpg  2.jpg  3.jpg
   data/roster/22CS002/1.jpg  ...
   ```
3. Build the roster. Only a 128-number embedding per student is saved; you may delete the photos afterwards.
   ```bash
   python -m smartattend enroll
   ```
   The log lists every enrolled student and any photo in which no face was found.

### 2b. Enroll and take attendance with the camera (easiest)

```bash
streamlit run app/streamlit_app.py      # opens http://localhost:8501
```
1. Sidebar → **Enroll students**. Type the student ID and name, take 3–5 photos with **Add this photo**
   (looking straight, slightly left, slightly right), tick the consent box and click **Save student**.
   Each photo is checked: exactly one face, large enough.
2. Sidebar → **Live attendance** → switch **Camera on**. A recognised student gets a yellow
   "confirming..." box, then turns green **PRESENT** after being recognised in 3 of 5 consecutive frames.
   A notification shows the time. Unknown faces get a red box and are never stored.
3. The register (student, present/absent, time marked, similarity) is saved automatically to
   `reports/attendance/attendance_YYYY-MM-DD.csv` every time someone is marked present. Use
   **New session (reset)** for the next lecture.
4. To delete a student's data, use **Remove a student** in the Enroll page.

Tip: if a student is not recognised, add 1–2 more photos taken in the classroom's own lighting. If someone
is confused with another student, raise the **Match threshold** slider.

## 3. Take attendance during a lecture

**Graphical interface:**
```bash
streamlit run app/streamlit_app.py
```
Upload a photo or short video of the class. You get a head count, a seat map, the faces with their
engagement colour, the class engagement curve, and a **suggested** attendance register you can download
as CSV. Check it before submitting.

**Command line** (photo, folder of photos, video file, or webcam `0`):
```bash
python -m smartattend analyze --input lecture.mp4 --out reports/session_2026-10-31
python -m smartattend attend  --input 0            # attendance only, live webcam
```
Outputs in the `--out` folder:

| file | content |
|---|---|
| `attendance.csv` | student_id, present/absent, frames seen, best similarity, timestamp |
| `engagement_timeline.csv` | per processed frame: people, faces, recognised, class engagement |
| `annotated.mp4` | video with seat map, names and engagement colours |
| `session_summary.json` | totals, mean engagement, latency |

## 4. Reading the results

* **Present** means the student was recognised in at least `min_frames_present` sampled frames
  (default 2). A student who is present but always looking down or hidden may be missed, so confirm absences.
* **Unknown** faces are people who are not enrolled (visitors, non-consenting students) or faces that
  are too small or blurred.
* **Engagement** is an *alertness* estimate (eyes, mouth, head pose) for the class as a whole. It is
  a teaching signal ("energy dropped after minute 25"), **not** a measure of learning, and must not be
  used to grade or discipline individual students.

## 5. Settings (`configs/default.yaml`)

| setting | default | effect |
|---|---|---|
| `face.match_threshold` | 0.363 | higher = fewer wrong identities, more missed students |
| `attendance.min_frames_present` | 2 | frames a student must be recognised in |
| `attendance.sample_every_n_frames` | 10 | process 1 of every N video frames (speed vs coverage) |
| `product.detector` | yolov8s | person detector (chosen by the benchmark) |
| `product.person_threshold` | 0.30 | detection confidence for counting (selected on validation split) |

## 6. Reproduce every experiment

```bash
python -m smartattend stage data          # download + curate datasets
python -m smartattend sandbox             # NumPy fundamentals figures
python -m smartattend experiments         # optimisers / ablations / overfitting
python -m smartattend stage engagement    # ResNet-18 vs GoogLeNet, fine-tuning
python -m smartattend stage sequence      # LSTM trends, transformer aggregator, VAE, latent space
python -m smartattend stage detection     # YOLO / Faster R-CNN / RetinaNet benchmark
python -m smartattend stage seating       # semantic + instance segmentation, seat maps
python -m smartattend stage face          # LFW verification, sessions, fairness, robustness
python -m smartattend stage conv_viz      # 2-D / 3-D convolution figures
python -m smartattend stage pipeline_latency
```
All runs use seed 42 (`configs/default.yaml`). Results go to `reports/` and logs to `logs/`.

## 7. Troubleshooting

| problem | fix |
|---|---|
| "no face found" during enrolment | use a brighter, front-facing photo with the face at least 80 px wide |
| a student is never recognised | add more enrolment photos in classroom lighting; check the camera angle |
| a stranger is recognised as a student | raise `face.match_threshold` (e.g. 0.42) |
| slow processing | raise `sample_every_n_frames`; switch `product.detector` to `yolov8n` (2.4x faster, slightly less accurate) |
| `cv2.imread` fails on paths with non-English characters | already handled; photos are read via `imdecode` |

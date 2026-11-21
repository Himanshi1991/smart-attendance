# Requirement 2 — Datasets: provenance, licensing, duplicates, labels, leakage-safe splits

## 1. Which data does each part of the system need?

| system part | data needed | dataset we use | link |
|---|---|---|---|
| attentive / distracted / inactive, head count in real classrooms | classroom frames with every student boxed and labelled | **Classroom Student Engagement** (Kaggle) | https://www.kaggle.com/datasets/mustafaxgm/classroom-student-engagement-dataset |
| face attendance + fairness | many photos per person, plus gender/ethnicity attributes | **LFW** (Kaggle copy, or the Hugging Face mirror with identity labels) | https://www.kaggle.com/datasets/jessicali9530/lfw-dataset |
| person detection on varied classroom-like photos, segmentation | boxes + masks for every person | **COCO 2017 val** (official site; the Kaggle copy is 27 GB) | https://cocodataset.org |
| alertness over time (LSTM) | face frames with timestamps + eye/mouth/head features | **drowsiness faces** (Hugging Face, MIT) | https://huggingface.co/datasets/notgoodkeeper/cnn-based-drowsiness-detection-data |

Every link above was checked through the Kaggle or Hugging Face API while writing this guide.

## 2. Downloading in Google Colab

Open `notebooks/02_datasets_colab.ipynb` in Colab (File → Upload notebook). Its cells do the following:

1. **Mount Google Drive.** Colab's own disk is wiped when the session ends, but Drive keeps your data.
2. **Unzip the project** from Drive and `pip install -e .`
3. **Download with kagglehub.** For a public dataset you need no Kaggle account:
   ```python
   import kagglehub
   kaggle_path = kagglehub.dataset_download('mustafaxgm/classroom-student-engagement-dataset')
   ```
   (Some Kaggle datasets require accepting rules or logging in. For those, upload `kaggle.json` from
   Kaggle → Settings → API → *Create New Token*, then `!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/`.)
4. **Prepare the splits**: `prepare('data/raw', 'data/processed/kaggle_classroom', source=kaggle_path)`
5. **Optional**: `!python -m smartattend stage data` for COCO + LFW + drowsiness (about 1 GB in total).

**GPU tip:** none of this needs a GPU. Use a CPU runtime here and save your GPU quota for training (Requirement 7).

On the laptop, `python -m smartattend stage kaggle_data` does the same steps (it downloads the zip itself).

## 3. From 8 actions to 3 states

The dataset labels *actions*. Our product reports *states*, so we map them:

| action | state | why |
|---|---|---|
| look_forward, read, write, handrise, stand | attentive | following the lesson (writing notes counts as attentive) |
| turn_head, using_device | distracted | attention is somewhere else |
| sleep | inactive | not participating at all |

The mapping lives in one dictionary, `ACTION_TO_STATE` in `src/smartattend/data/kaggle_classroom.py`, so it
is easy to change and is unit-tested (every action must map to a state).

## 4. Duplicates

Frames cut from a video are often nearly identical. We compute a **difference hash** (dHash) for each image:
shrink it to 9×8 grey pixels and record, for each pair of neighbouring pixels, whether brightness goes up or
down. That gives 64 bits. Two images whose hashes differ in ≤ 3 bits are near-duplicates.

Result: 50 near-duplicate pairs, **all inside the same recording group**. That is harmless, because they
end up in the same split. 0 pairs cross groups, so nothing had to be removed.

## 5. Leakage-safe splitting: the most important idea in this requirement

**Leakage** means test data secretly resembles training data, so the test score looks better than reality.
Three levels of splitting, from worst to best:

| split by | problem |
|---|---|
| random frames | frame 41 in train, frame 42 in test: almost the same picture, so the model "remembers" the answer |
| video | `taimoor-class-2` in train, `taimoor-class-3` in test: **the same students in the same room** (we checked the frames by eye) |
| **recording group** (what we do) | test = `taimoor-zaeem`, a different room with different students |

```python
VIDEO_GROUP = {"owais-class-1": "owais", "owais-class-2": "owais", ...,
               "taimoor-class-2": "taimoor-class", ..., "taimoor-zaeem-1": "taimoor-zaeem", ...}
GROUP_SPLIT = {"owais": "train", "seminar": "train", "misc": "train",
               "taimoor-class": "val", "taimoor-zaeem": "test"}
```

A unit test checks that all videos of a group land in the same split.

The other datasets use the same principle: COCO-classroom splits by image and deduplicates first; LFW never
uses the same photo for enrolment and testing; the drowsiness frames are split into 5-second time blocks with
0.5-second gaps between them.

## 6. Run it and expected output

```bash
python -m smartattend stage kaggle_data
python -m pytest tests/test_kaggle_classroom.py
```

```
train 217 images 2277 people  groups ['misc', 'owais', 'seminar']  attentive 1759, distracted 448, inactive 70
val   179 images 1167 people  groups ['taimoor-class']             attentive 954,  distracted 143, inactive 70
test   85 images 1159 people  groups ['taimoor-zaeem']             attentive 421,  distracted 646, inactive 92
duplicates: 50 pairs within groups, 0 removed across groups
```

Figures: `reports/figures/data/kaggle_label_distribution.png` (state mix per split) and
`reports/figures/data/kaggle_example_crops.png` (random crops per state, so you can check the labels by eye).

## 7. What to notice

* **Class imbalance**: inactive is only 3–8 % of students. Accuracy alone would hide failures on it, which is
  why Requirement 1 uses **macro-F1** (each class counts equally).
* **Distribution shift**: the test classroom is 56 % distracted vs 20 % in training. Expect test scores to be
  lower than validation scores. That is the honest number to report.
* **Fairness limit**: almost all students in this dataset are young men from one university, so fairness
  cannot be measured on it. The face-recognition fairness analysis uses LFW instead.

## 8. Checklist
- [x] datasets chosen per system part, links verified
- [x] provenance and licences documented (`docs/DATASET_CARD.md`, `provenance.json`)
- [x] duplicates detected (dHash), cross-split duplicates removed (none found)
- [x] labels created (8 actions → 3 states; person boxes; per-student crops)
- [x] leakage-safe train / val / test by recording group, unit-tested
- [x] Colab download notebook (`notebooks/02_datasets_colab.ipynb`)

# Requirement 1 — Use case, measurable requirements, acceptance tests, deployment constraints

## 1. What problem are we solving? (in simple words)

In a normal lecture the teacher spends about 5 minutes calling names, and still has no idea whether the
class is following. Our system watches the classroom through a camera and answers five questions:

1. Who is present? (face recognition against the class list)
2. How many people are in the room, and which seats are empty?
3. Is each visible student **attentive**, **distracted** (head turned away) or **inactive** (drowsy)?
4. How does the class's attention change during the lecture?
5. How accurate is all of this? (measured, not guessed)

## 2. Who needs it? (stakeholders)

| stakeholder | what they need |
|---|---|
| Teacher | a correct register without roll-call; a view of when the class loses focus |
| School admin | no "proxy" attendance (a friend answering for an absent student) |
| Students | to be treated equally, whatever their gender or ethnicity; privacy |
| IT staff | it must run on an ordinary classroom PC, without a GPU |

## 3. Turning needs into numbers

A need like "the register should be correct" cannot be tested. "At least 95 % of present students are
marked present" can. Every need gets a **metric**, a **target**, and a **test** that measures it:

| ID | need | metric | target | measured by |
|---|---|---|---|---|
| R1 | correct register | attendance recall | ≥ 0.95 | `stage face` |
| R2 | absent students not marked present | false-present rate | ≤ 0.02 | `stage face` |
| R3 | no proxy attendance | stranger false-accept rate | ≤ 0.01 | `stage face` |
| R4 | equal treatment | largest recall gap between groups | ≤ 0.05 | `stage face` |
| R5 | find every person | detection mAP@0.5 | ≥ 0.80 | `stage detection` |
| R6 | trustworthy head count | count error (people per image) | ≤ 1 | `stage detection` |
| R7 | know when focus drops | engagement macro-F1 | ≥ 0.65 | `stage engagement_cv` |
| R8 | runs without a GPU | latency per frame (95th percentile) | ≤ 500 ms | `stage pipeline_latency` |
| R9 | fits on an ordinary PC | peak memory | ≤ 2000 MB | `stage pipeline_latency` |

**Why these metrics?**
* *Recall* answers "of the students who were really there, how many did we mark?"
* *False-present* and *FAR* answer "did we ever mark someone who was not there?" This matters more for
  trust than recall.
* *Fairness gap* is the difference in recall between the best- and worst-served group.
* *mAP* (mean average precision) is the standard detection score: it rewards finding people *and* drawing
  tight boxes.
* *Macro-F1* averages the F1 score of each class equally, so a rare class (drowsy) counts as much as a
  common one.
* *p95 latency* is the time that 95 % of frames finish within. It reflects the slow frames, not just the average.

## 4. Deployment constraints

* CPU-only on a classroom PC. A GPU is used only for training (for example, Google Colab).
* Offline: images never leave the school.
* Only consenting, enrolled students are identified, and only face *embeddings* (128 numbers) are stored.
* The teacher confirms every register. Engagement is shown for the whole class, never used to grade a student.

## 5. The code

`src/smartattend/requirements.py` stores each requirement as data:

```python
Requirement("R1", "Teacher", "Correct register without roll-call", "attendance recall", 0.95, ">=", "",
            "reports/eval/face_recognition.json", lambda r, c: r["sessions"]["attendance_recall"],
            "python -m smartattend stage face"),
```

Read it left to right: ID, stakeholder, need, metric, **target 0.95**, **higher is better (`>=`)**, unit,
**which result file holds the measurement**, **how to pull the number out of that file**, and the command
that creates the file. `evaluate()` opens each file, extracts the value, and compares it with the target.
If a file does not exist yet, the requirement is "NOT RUN" and the table tells you which command to run.

## 6. Run it

```bash
python -m smartattend acceptance
python -m pytest tests/test_requirements.py      # tests for the checker itself
```

Expected output (on our laptop):

```
ID  Stakeholder  Metric                                        Target      Measured  Status
R1  Teacher      attendance recall                             >= 0.95     0.9940    PASS
R2  Teacher      false-present rate                            <= 0.02     0.0010    PASS
R3  Admin        stranger false-accept rate (FAR)              <= 0.01     0.0011    PASS
R4  Students     max attendance-recall gap (gender, ethnicity) <= 0.05     0.0070    PASS
R5  Teacher      person detection mAP@0.5                      >= 0.8      0.8164    PASS
R6  Teacher      count error (MAE, fully annotated images)     <= 1 people 0.9882    PASS
R7  Teacher      engagement macro-F1 (5-fold CV)               >= 0.65     0.7109    PASS
R8  IT           CPU latency per frame, p95                    <= 500 ms   6345      FAIL
R9  IT           peak memory                                   <= 2000 MB  735       PASS
8 pass, 1 fail, 0 not run, 0 error
```

**Update after Requirement 10:** four requirements for real classroom video (R10 to R13) were added and the app
latency was re-measured. The check now reports **10 pass, 3 fail** (R8 latency 2,015 ms, R11 state macro-F1 0.467,
R12 back-row gap 0.345). See [10_integration_evaluation.md](10_integration_evaluation.md).


On a fresh copy, before any experiment has run, every row says **NOT RUN** with the command to run.
That is expected: Requirement 1 defines the tests, and the later requirements produce the numbers.

## 7. What the FAIL means

R8 fails: 6.3 s at p95, measured on battery with Windows capping the CPU at 1.2 GHz, on frames with many
faces. A failing acceptance test is useful information, not something to hide. It goes into the final
report, together with the fixes: sample one frame every few seconds for attendance, batch the face crops,
or use a GPU.

## 8. Checklist for this requirement
- [x] use case and stakeholders written (`docs/REQUIREMENTS.md`, this guide)
- [x] every need has a metric and a numeric target
- [x] acceptance tests run automatically (`python -m smartattend acceptance`)
- [x] the checker itself is unit-tested (`tests/test_requirements.py`)
- [x] deployment constraints listed

# Step 1 — Use case, requirements, acceptance tests and constraints

## Industry use case
A school or college wants to (a) take attendance automatically from a classroom camera by
recognising enrolled students' faces, and (b) give teachers an aggregate, per-session view of
classroom engagement so they can adjust their teaching. The system replaces manual roll-call
(≈5 min per lecture) and gives engagement trends that are otherwise invisible.

## Stakeholders and needs → measurable requirements

| Stakeholder | Need | Measurable requirement | Metric |
|---|---|---|---|
| Teacher | Correct register without roll-call | ≥ 95 % of present students marked present; ≤ 2 % false "present" | per-student recall / false-accept rate |
| Teacher | Head-count sanity check | Mean absolute count error ≤ 1 person per frame | count MAE |
| Admin | No impersonation / proxy attendance | False-accept rate for non-enrolled faces ≤ 1 % at chosen threshold | FAR on stranger set |
| Teacher | Engagement overview (aggregate, not per-student punishment) | Macro-F1 ≥ 0.65 on 3 classes (disengaged / neutral / engaged) | macro F1 |
| Teacher | Trend over a lecture | LSTM trend forecast beats "last value" baseline | MAE vs baseline |
| Students | Fair treatment | Gap in recall / F1 between groups (seat row, lighting, skin tone where labelled, gender where labelled) ≤ 5 pp | fairness gap |
| IT | Runs on a classroom PC without GPU | ≤ 500 ms per processed frame on CPU; ≤ 2 GB RAM | latency p95, peak memory |
| Detection | Person detection quality | mAP@0.5 ≥ 0.80 on test split | mAP |
| Privacy officer | Data protection | No raw images stored by default; only embeddings + register; consent recorded | checklist |

## Acceptance tests
Automated tests live in `tests/`. Tests marked *(planned)* arrive with the vision/eval stages.

| ID | Test | Status |
|---|---|---|
| AT-01 | Analytic gradients of every MLP layer match finite differences (rel. err < 1e-6) | `tests/test_mlp_gradients.py` ✅ |
| AT-02 | Optimisers, neurons, classic rules and RBM behave as theory predicts | `tests/test_neurons_optimizers.py`, `tests/test_classic_and_rbm.py` ✅ |
| AT-03 | Roster matching is one-to-one, rejects strangers, register needs ≥ N frames | `tests/test_attendance.py` ✅ |
| AT-04 | Train/val/test splits are disjoint; normalisation fitted on train only | `tests/test_neurons_optimizers.py` ✅ |
| AT-05 | Attendance recall ≥ 95 %, FAR ≤ 1 % on held-out session videos | *(planned)* |
| AT-06 | Detection mAP@0.5 ≥ 0.80, count MAE ≤ 1 | *(planned)* |
| AT-07 | Engagement macro-F1 ≥ 0.65, fairness gap ≤ 5 pp | *(planned)* |
| AT-08 | CPU latency p95 ≤ 500 ms/frame, memory ≤ 2 GB | *(planned)* |

## Requirements added at integration (real classroom video)

When the Kaggle classroom videos with per-student labels became available, four requirements were added. Their
targets were written down before the final evaluation ran (`python -m smartattend evaluate`).

| ID | Stakeholder | Need | Metric | Target | Measured | Status |
|---|---|---|---|---|---|---|
| R10 | Teacher | head count in a real class video | count error (MAE), test classroom | ≤ 2 students | 1.46 | PASS |
| R11 | Teacher | attentive / distracted / inactive in a new classroom | state macro-F1, test classroom | ≥ 0.60 | 0.467 | FAIL |
| R12 | Students | back-row students are not overlooked | detection recall gap, small vs large students | ≤ 0.15 | 0.345 | FAIL |
| R13 | IT | keeps working in a dim room | mAP@0.5 lost in low light | ≤ 0.05 | 0.000 | PASS |

Current acceptance result: **10 of 13 pass**. The fails are R8 (app latency with face recognition on every frame,
2.0 s at p95 on a laptop on battery), R11 and R12. See `docs/guide/10_integration_evaluation.md`.

## Deployment constraints
- CPU-only inference on a classroom PC (Windows); GPU used only for offline training (Colab).
- Works on a single fixed camera at the front of the room; minimum face size ≈ 30 px.
- Offline operation: models are local files; no images leave the premises.
- Configurable thresholds (`configs/default.yaml`); every run logs its config and seed.

## Responsible-use limits (face recognition)
- **Consent first:** only students who opted in are enrolled; others are never identified
  (they appear as "unknown" and are not stored).
- **Human in the loop:** the register is a *suggestion* the teacher confirms; it is never
  the sole basis for disciplinary action.
- **Engagement is aggregate:** reported per class/session, not used to grade individuals.
- **Data minimisation:** store 128-d embeddings, not photos; delete on student request
  (remove the folder and re-run `enroll`).
- **Known risks:** lower accuracy for back rows, low light, occlusion (masks, hands),
  and possible demographic performance gaps — these are measured in the evaluation package.

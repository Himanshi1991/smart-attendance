// Builds docs/presentation.pptx from the result JSONs and figures in reports/ (no hand-typed numbers).
//   NODE_PATH=<dir with pptxgenjs> node scripts/build_deck.js <path to apply_theme.js>
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const R = (p) => path.join(ROOT, p);
const J = (p) => (fs.existsSync(R(p)) ? JSON.parse(fs.readFileSync(R(p), "utf8")) : null);
const pct = (x, d = 1) => (x == null ? "n/a" : `${(100 * x).toFixed(d)}%`);
const f3 = (x) => (x == null ? "n/a" : x.toFixed(3));

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// fit an image inside a box, preserving aspect ratio, centred
function fit(file, x, y, w, h) {
  const s = pngSize(file);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  return { path: file, x: x + (w - iw) / 2, y: y + (h - ih) / 2, w: iw, h: ih };
}

const det = J("reports/experiments/detection/detection_benchmark.json");
const seat = J("reports/experiments/seating/seating.json");
const face = J("reports/eval/face_recognition.json");
const eng = J("reports/experiments/engagement/engagement.json");
const trend = J("reports/experiments/sequence/trend_models.json");
const agg = J("reports/experiments/sequence/transformer_aggregator.json");
const lat = J("reports/eval/pipeline_latency.json");
const demo = J("reports/demo/session_summary.json");
const opt = J("reports/experiments/nnfund/optimizer_comparison.json");
const cv = J("reports/experiments/engagement/engagement_cv.json");
const bm = J("reports/eval/benchmark.json");
const ce = J("reports/eval/classroom_eval.json");
const sq = J("reports/eval/sequence_models.json");
const fe = J("reports/eval/final_evaluation.json");
const acc = J("reports/eval/acceptance.json");

// ---------------------------------------------------------------- theme: chalkboard
const THEME = {
  name: "Chalkboard",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2B2A", lt1: "FFFFFF", dk2: "1F3D3A", lt2: "EEF3F1",
    accent1: "1F6F64", accent2: "F2A541", accent3: "C8553D", accent4: "6C8EAD",
    accent5: "8AA399", accent6: "4A5A58", hlink: "1F6F64", folHlink: "6C8EAD",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Smart Attendance and Classroom Engagement Analyzer";
pres.author = "Project 22";
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [],
  placeholders: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.8, fontSize: 28, bold: true, align: "left",
      color: C.text2, valign: "top", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.4, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

let n = 0;
function content(section, title, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  n++;
  return s;
}
function txt(s, text, o) {
  s.addText(text, { isTextBox: true, fontSize: 15, color: C.text1, valign: "top", margin: 0, ...o });
}
function bullets(s, items, o) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })),
    { isTextBox: true, fontSize: 15, color: C.text1, valign: "top", margin: 0, ...o });
}
function stat(s, x, y, w, value, label, color) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.55, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 } });
  s.addText(value, { isTextBox: true, x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.85, fontSize: 40, bold: true,
    color: color || C.accent1, fontFace: THEME.headFontFace, margin: 0, valign: "middle" });
  s.addText(label, { isTextBox: true, x: x + 0.2, y: y + 0.97, w: w - 0.4, h: 0.5, fontSize: 12, color: C.accent6, margin: 0, valign: "top" });
}
function caption(s, text, x, y, w) {
  s.addText(text, { isTextBox: true, x, y, w, h: 0.3, fontSize: 11, italic: true, color: C.accent6, margin: 0 });
}
const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
  legendFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: HEX.accent6, valAxisLabelColor: HEX.accent6,
  dataLabelColor: HEX.dk1, titleColor: HEX.dk2, catAxisLabelFontSize: 11, valAxisLabelFontSize: 10, dataLabelFontSize: 10,
  legendFontSize: 11, titleFontSize: 13 };

// ================================================================ 1. title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  n++;
  s.addText("Smart Attendance and Classroom Engagement Analyzer", { isTextBox: true, x: 0.8, y: 2.0, w: 11.5, h: 1.6,
    fontSize: 44, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, valign: "bottom" });
  s.addText("Face-recognition attendance, people counting, seating and class alertness trends — on a CPU laptop",
    { isTextBox: true, x: 0.8, y: 3.75, w: 11.5, h: 0.8, fontSize: 20, color: C.accent2, margin: 0 });
  s.addText("Project 22 · Education Technology · Python implementation", { isTextBox: true, x: 0.8, y: 6.3, w: 11.5, h: 0.4,
    fontSize: 14, color: C.background2, margin: 0 });
  s.addNotes("Goal: replace roll-call with a consent-based face-recognition register and give teachers an aggregate alertness signal. Everything shown runs on an Intel i3 laptop without a GPU.");
}

// ================================================================ 2. problem & targets
{
  const s = content("Introduction", "Roll-call costs minutes and says nothing about the room",
    "Targets come from docs/REQUIREMENTS.md. Each is checked by an automated or scripted acceptance test.");
  txt(s, "Stakeholder needs translated into measurable targets", { x: 0.6, y: 1.35, w: 6, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  bullets(s, [
    "Teacher: correct register without roll-call → recall ≥ 95%, false-present ≤ 2%",
    "Admin: no proxy attendance → strangers rejected (FAR ≤ 1%)",
    "Teacher: head-count check → count error ≤ 1 person per image",
    "Teacher: class alertness overview → engagement macro-F1 ≥ 0.65",
    "Students: equal treatment → group gaps ≤ 5 percentage points",
    "IT: runs on a classroom PC → CPU only, ≤ 2 GB RAM",
  ], { x: 0.6, y: 1.9, w: 6.6, h: 4.8 });
  stat(s, 7.8, 1.4, 4.9, "~5 min", "lost to manual roll-call in every lecture");
  stat(s, 7.8, 3.15, 4.9, "6", "measurable targets, each with an acceptance test");
  stat(s, 7.8, 4.9, 4.9, "0 GPUs", "everything runs on an Intel i3 laptop, 8 GB RAM", C.accent2);
}

// ================================================================ 3. architecture
{
  const s = content("Introduction", "One frame in, a register and an engagement curve out",
    "Per-frame pipeline. Each block was benchmarked against alternatives; chosen components are shown.");
  const steps = [
    ["Camera frame", "image / video / webcam"],
    ["People", "YOLOv8-s detector"],
    ["Faces", "YuNet detector"],
    ["Identity", "SFace embedding + roster"],
    ["Engagement", "ResNet-18 / GoogLeNet on face crop"],
    ["Seating", "row clustering + instance masks"],
    ["Class view", "Transformer aggregator + LSTM trend"],
  ];
  const w = 1.62, gap = 0.12, y = 1.55;
  steps.forEach(([h, d], i) => {
    const x = 0.6 + i * (w + gap);
    s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.9, rectRadius: 0.1, fill: { color: i === 0 || i === 6 ? C.accent1 : C.background2 },
      line: { color: i === 0 || i === 6 ? C.accent1 : C.background2 }, objectName: `step${i}` });
    s.addText(h, { isTextBox: true, x: x + 0.1, y: y + 0.2, w: w - 0.2, h: 0.5, fontSize: 16, bold: true,
      color: i === 0 || i === 6 ? C.background1 : C.text2, margin: 0, align: "center" });
    s.addText(d, { isTextBox: true, x: x + 0.1, y: y + 0.8, w: w - 0.2, h: 1.0, fontSize: 12,
      color: i === 0 || i === 6 ? C.background1 : C.accent6, margin: 0, align: "center" });
    if (i < steps.length - 1) s.addShape(pres.ShapeType.rightArrow, { x: x + w - 0.02, y: y + 0.82, w: gap + 0.04, h: 0.26, fill: { color: C.accent2 }, line: { color: C.accent2 } });
  });
  txt(s, "Outputs per session", { x: 0.6, y: 4.0, w: 6, h: 0.4, fontSize: 18, bold: true, color: C.accent1 });
  bullets(s, ["attendance.csv: suggested register for the teacher to confirm",
    "engagement_timeline.csv: class alertness over the lecture",
    "annotated.mp4: seat map, names and engagement colours"], { x: 0.6, y: 4.55, w: 6.2, h: 1.8 });
  bullets(s, ["CLI: python -m smartattend analyze --input lecture.mp4",
    "GUI: streamlit run app/streamlit_app.py",
    "Reproduce: python -m smartattend stage all"], { x: 7.0, y: 4.55, w: 5.8, h: 1.8, fontSize: 14 });
}

// ================================================================ 4. data
pres.addSection({ title: "Data and fundamentals" });
{
  const s = content("Data and fundamentals", "Three public datasets, split so the test set never leaks",
    "No real student data was collected. COCO val2017 is unseen by all pretrained detectors. The drowsiness frames are consecutive video, so a random split would leak near-duplicates; we split by 5-second blocks with buffers.");
  const cards = [
    ["COCO-Classroom", "598 images · 4,995 people", "Classroom-like scenes chosen from human captions and scene objects. 450 / 44 / 104 split, deduplicated by perceptual hash.", "detection · counting · seating"],
    ["LFW + attributes", "13,233 faces · 5,749 people", "Verification, simulated 30-student sessions, robustness. Gender/ethnicity groups from predicted attributes.", "attendance · fairness"],
    ["Drowsiness faces (MIT)", "2,376 frames · 1 person", "Eye/mouth/head-pose features at 20 fps. Weak heuristic labels mapped to engaged / neutral / disengaged.", "engagement · trends"],
  ];
  cards.forEach(([t, sub, body, use], i) => {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.5, w: 3.8, h: 4.6, rectRadius: 0.12, fill: { color: C.background2 }, line: { color: C.background2 } });
    txt(s, t, { x: x + 0.3, y: 1.75, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: C.text2 });
    txt(s, sub, { x: x + 0.3, y: 2.3, w: 3.2, h: 0.4, fontSize: 14, bold: true, color: C.accent1 });
    txt(s, body, { x: x + 0.3, y: 2.85, w: 3.2, h: 2.3, fontSize: 14 });
    txt(s, use, { x: x + 0.3, y: 5.4, w: 3.2, h: 0.4, fontSize: 13, italic: true, color: C.accent3 });
  });
  caption(s, "Synthetic data is used only for the NumPy fundamentals and the simulated class sessions, and is labelled as such everywhere.", 0.6, 6.45, 12);
}

// ================================================================ 5. fundamentals
{
  const s = content("Data and fundamentals", "NumPy fundamentals, every gradient verified",
    "Every layer's backward pass is checked against central differences in the test suite. The overfitting study shows L2 cutting the generalisation gap from 0.31 to 0.11.");
  s.addImage(fit(R("reports/experiments/nnfund/overfitting.png"), 6.4, 1.35, 6.4, 3.4));
  caption(s, "Overfitting study: 60 samples, 128-128 MLP. L2 keeps validation loss flat.", 6.4, 4.8, 6.4);
  bullets(s, [
    "Tensor sandbox: broadcasting, inner products, SVD, gradient checking",
    "McCulloch-Pitts gates, sigmoid neuron (fails XOR), MLP (solves XOR)",
    "Dense, dropout and batch-norm layers: every gradient checked (< 1e-6)",
    "Hebbian, Oja, competitive and delta-rule learning",
    "RBM with CD-1: data sits in low free-energy valleys",
  ], { x: 0.6, y: 1.4, w: 5.5, h: 3.6, fontSize: 14 });
  if (opt) {
    const names = Object.keys(opt);
    s.addChart(pres.ChartType.bar, [{ name: "final training loss", labels: names, values: names.map((k) => +opt[k].final_train_loss.toFixed(3)) }],
      { x: 0.6, y: 5.0, w: 5.6, h: 2.2, barDir: "bar", showTitle: true, title: "Optimiser comparison: final training loss (lower is better)",
        showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", chartColors: [HEX.accent1], showLegend: false,
        valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisHidden: true, ...chartText });
  }
  if (fs.existsSync(R("reports/figures/nnfund/rbm_energy_model.png")))
    s.addImage(fit(R("reports/figures/nnfund/rbm_energy_model.png"), 6.4, 5.15, 6.4, 1.95));
}

// ================================================================ 6. detection benchmark
pres.addSection({ title: "Vision" });
if (det) {
  const names = Object.keys(det);
  const short = { "yolov8n": "YOLOv8-n", "yolov8s": "YOLOv8-s", "yolov8n-classroom": "YOLOv8-n tuned",
    "fasterrcnn_resnet50_fpn": "Faster R-CNN", "retinanet_resnet50_fpn": "RetinaNet" };
  const s = content("Vision", "YOLOv8-s gives the best accuracy at a CPU-viable speed",
    "All models were tested on 104 classroom-like COCO val2017 images. Faster R-CNN is competitive but 10x slower on CPU. Fine-tuning YOLOv8-n on 450 images did not improve mAP: an honest negative result.");
  s.addChart(pres.ChartType.bar, [
    { name: "mAP@0.5", labels: names.map((k) => short[k]), values: names.map((k) => +det[k].mAP50.toFixed(3)) },
    { name: "mAP@[.5:.95]", labels: names.map((k) => short[k]), values: names.map((k) => +det[k].mAP50_95.toFixed(3)) },
  ], { x: 0.6, y: 1.35, w: 6.4, h: 4.0, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Detection accuracy (test split)",
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", chartColors: [HEX.accent1, HEX.accent5], showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 1, valGridLine: { color: "E3E9E7", size: 0.5 }, catGridLine: { style: "none" }, ...chartText });
  s.addChart(pres.ChartType.bar, [
    { name: "ms per image", labels: names.map((k) => short[k]), values: names.map((k) => Math.round(det[k].latency_ms_mean)) },
  ], { x: 7.2, y: 1.35, w: 5.6, h: 4.0, barDir: "bar", showTitle: true, title: "CPU latency per image (ms)", showValue: true,
    dataLabelPosition: "outEnd", chartColors: [HEX.accent2], showLegend: false, valAxisHidden: true,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, ...chartText });
  const y8s = det["yolov8s"];
  bullets(s, [
    `Selected YOLOv8-s: mAP@0.5 ${f3(y8s.mAP50)}, count error ${y8s.count_MAE_no_crowd.toFixed(2)} people/image on fully annotated images`,
    "Faster R-CNN is the most blur-robust, but 3.8 s per image rules it out without a GPU",
    "Fine-tuning YOLOv8-n on 450 varied photos did not beat the stock model; on real classroom video it nearly doubles mAP (see Real classroom video)",
  ], { x: 0.6, y: 5.6, w: 12.2, h: 1.4, fontSize: 14 });
}

// ================================================================ 7. back-row problem
if (det) {
  const s = content("Vision", "Students at the back are the hardest to see",
    "Recall on people smaller than 32x32 px is about a third of recall on large people for every model. This size gap is the vision analogue of a fairness gap and drives the camera-placement recommendation.");
  const names = Object.keys(det);
  const short = { "yolov8n": "YOLOv8-n", "yolov8s": "YOLOv8-s", "yolov8n-classroom": "YOLOv8-n tuned",
    "fasterrcnn_resnet50_fpn": "Faster R-CNN", "retinanet_resnet50_fpn": "RetinaNet" };
  const buckets = Object.keys(det[names[0]].recall_by_size);
  const labels = ["small (back rows)", "medium", "large (front rows)"];
  s.addChart(pres.ChartType.bar, names.map((k, i) => ({ name: short[k], labels, values: buckets.map((b) => +det[k].recall_by_size[b].toFixed(2)) })),
    { x: 0.6, y: 1.35, w: 7.6, h: 5.3, barDir: "col", barGrouping: "clustered", showTitle: true, title: "Recall by person size",
      chartColors: [HEX.accent5, HEX.accent1, HEX.accent4, HEX.accent2, HEX.accent3], showLegend: true, legendPos: "b",
      valAxisMinVal: 0, valAxisMaxVal: 1, valGridLine: { color: "E3E9E7", size: 0.5 }, catGridLine: { style: "none" }, ...chartText });
  const sm = det["yolov8s"].recall_by_size;
  const v = Object.values(sm);
  stat(s, 8.6, 1.5, 4.2, pct(v[0], 0), "YOLOv8-s recall on small people", C.accent3);
  stat(s, 8.6, 3.2, 4.2, pct(v[2], 0), "YOLOv8-s recall on large people");
  txt(s, "Mitigation: mount the camera high at the front, at most 8 m from the back row, so faces are larger than 40 px.",
    { x: 8.6, y: 5.0, w: 4.2, h: 1.6, fontSize: 14 });
}

// ================================================================ 8. seating
if (seat) {
  const s = content("Vision", "Instance masks separate students; semantic masks merge them",
    "Person-pixel IoU on the test set. The seat map is a heuristic without row-level ground truth; it merges rows in dense tiered photos, so deployment should use a one-time per-room seat calibration.");
  s.addImage(fit(R("reports/experiments/seating/seating_examples.png"), 0.6, 1.3, 7.8, 5.8));
  stat(s, 8.8, 1.4, 4.0, f3(seat.instance_YOLOv8nSeg_person_IoU), `instance IoU (YOLOv8-n-seg), ${Math.round(seat.instance_latency_ms)} ms`);
  stat(s, 8.8, 3.1, 4.0, f3(seat.semantic_LRASPP_person_IoU), `semantic IoU (LR-ASPP), ${Math.round(seat.semantic_latency_ms)} ms`, C.accent6);
  txt(s, "Limit: the seat-row heuristic is unvalidated (no row ground truth) and merges rows in dense photos. Deploy with a one-time seat calibration per room.",
    { x: 8.8, y: 4.85, w: 4.0, h: 2.0, fontSize: 14, color: C.accent3 });
}

// ================================================================ 9. 3D conv
{
  const s = content("Vision", "Convolution, made visible",
    "Left: explicit 2D convolution on a patch (unit-tested against OpenCV). Right: a 3D temporal-difference kernel cancels the static body and responds only to the moving arm.");
  s.addImage(fit(R("reports/figures/vision/conv2d_explained.png"), 0.6, 1.3, 6.1, 3.2));
  s.addImage(fit(R("reports/figures/vision/resnet_filters_featuremaps.png"), 0.6, 4.6, 6.1, 2.6));
  s.addImage(fit(R("reports/figures/vision/conv3d_motion.png"), 6.9, 1.6, 5.9, 2.6));
  bullets(s, ["2D: the kernel slides over the image computing inner products",
    "ResNet-18: edges in conv1 become object parts by layer4",
    "3D: a 2x3x3 kernel over time highlights only motion (a raised hand)"], { x: 6.9, y: 4.6, w: 5.9, h: 2.4, fontSize: 14 });
}

// ================================================================ 10. face attendance
pres.addSection({ title: "Attendance" });
if (face) {
  const ss = face.sessions;
  const s = content("Attendance", "Face recognition marks 99.4% of present students correctly",
    "200 simulated sessions on LFW: 30 enrolled students, 5 absent, 3 non-enrolled strangers, 2 frames each. Same matcher and min-frames rule as the product.");
  stat(s, 0.6, 1.45, 3.9, pct(ss.attendance_recall), "attendance recall (target ≥ 95%)");
  stat(s, 4.7, 1.45, 3.9, pct(ss.false_present_rate), "absent students marked present (≤ 2%)");
  stat(s, 8.8, 1.45, 3.9, ss.count_MAE.toFixed(2), "count error, students per session (≤ 1)");
  s.addImage(fit(R("reports/eval/face_recognition.png"), 0.6, 3.3, 12.2, 3.5));
  caption(s, `Verification on ${face.verification.genuine_pairs.toLocaleString("en-US")} genuine / ${face.verification.impostor_pairs.toLocaleString("en-US")} impostor pairs: TAR ${pct(face.verification["TAR@FAR=0.001"].TAR)} at FAR 0.1%`, 0.6, 6.9, 12.2);
}

// ================================================================ 11. fairness table
if (face) {
  const s = content("Attendance", "Fairness gaps stay under 1 point, but they are not zero",
    "Groups come from predicted LFW attributes, not self-identification, and group sizes are small. These gaps must be re-measured on the school's own consented data each term.");
  const ss = face.sessions, gv = face.group_verification;
  const head = ["Group", "Attendance recall", "False rejects (FRR)", "False accepts (FAR)"].map((t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } }));
  const rows = [head];
  for (const attr of ["gender", "ethnicity"]) {
    for (const [g, r] of Object.entries(ss[`recall_by_${attr}`])) {
      const v = gv[attr].by_group[g] || {};
      rows.push([`${g} (n=${r.n})`, pct(r.recall, 2), pct(v.FRR, 2), pct(v.FAR, 3)]);
    }
    rows.push([{ text: `max gap: ${attr}`, options: { bold: true } }, { text: `${(100 * ss[`recall_gap_${attr}`].gap).toFixed(2)} pp`, options: { bold: true } },
      { text: `${(100 * gv[attr].FRR_gap).toFixed(2)} pp`, options: { bold: true } }, { text: `${(100 * gv[attr].FAR_gap).toFixed(3)} pp`, options: { bold: true } }]);
  }
  s.addTable(rows, { x: 0.6, y: 1.4, w: 7.4, colW: [2.2, 1.7, 1.75, 1.75], fontSize: 13, fontFace: THEME.bodyFontFace, color: HEX.dk1,
    border: { type: "solid", pt: 0.5, color: "D5DEDB" }, rowH: 0.42 });
  const rob = face.robustness;
  const kinds = ["clean", "low_light", "small_face", "jpeg", "mask_occlusion", "blur", "noise"];
  s.addChart(pres.ChartType.bar, [{ name: "identified", labels: kinds.map((k) => k.replace("_", " ")), values: kinds.map((k) => +(100 * rob[k].rank1_identified).toFixed(1)) }],
    { x: 8.3, y: 1.35, w: 4.5, h: 4.4, barDir: "bar", showTitle: true, title: "Robustness: % correctly identified", showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.0", chartColors: [HEX.accent1], showLegend: false, valAxisMinVal: 90, valAxisMaxVal: 101.5, valAxisHidden: true,
      valGridLine: { style: "none" }, catGridLine: { style: "none" }, ...chartText });
  txt(s, "Female false rejects are double the male rate, and the black group's false-accept rate is 3.5x the others. Both are within target but must be monitored.",
    { x: 8.3, y: 5.9, w: 4.5, h: 1.2, fontSize: 13, color: C.accent3 });
}

// ================================================================ 12. engagement
pres.addSection({ title: "Engagement" });
if (eng) {
  const keys = ["resnet18_linear_probe", "googlenet_linear_probe", "resnet18_finetuned"];
  const lab = { resnet18_linear_probe: "ResNet-18 probe", googlenet_linear_probe: "GoogLeNet probe", resnet18_finetuned: "ResNet-18 fine-tuned" };
  const best = keys.reduce((a, b) => (eng[a].macro_f1 >= eng[b].macro_f1 ? a : b));
  const headline = cv ? `Alertness classifier: macro-F1 ${f3(Math.max(cv.resnet18.mean, cv.googlenet.mean))} in 5-fold CV (target 0.65)`
    : `Alertness classifier reaches macro-F1 ${f3(eng[best].macro_f1)} (target 0.65)`;
  const s = content("Engagement", headline,
    "Test split is 5-second blocks never seen in training. Caveat: one person, one webcam, weak heuristic labels. This measures alertness, not learning, and cannot be checked for fairness.");
  s.addChart(pres.ChartType.bar, [{ name: "macro-F1", labels: keys.map((k) => lab[k]), values: keys.map((k) => +eng[k].macro_f1.toFixed(3)) }],
    { x: 0.6, y: 1.35, w: 4.6, h: 3.6, barDir: "col", showTitle: true, title: "Test macro-F1", showValue: true, dataLabelPosition: "outEnd",
      dataLabelFormatCode: "0.000", chartColors: [HEX.accent1], showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 1,
      valGridLine: { color: "E3E9E7", size: 0.5 }, catGridLine: { style: "none" }, ...chartText });
  s.addImage(fit(R("reports/experiments/engagement/engagement_confusion.png"), 5.4, 1.35, 7.4, 3.6));
  const ci = eng.resnet18_finetuned.macro_f1_95CI;
  bullets(s, [
    `Fine-tuned ResNet-18 (product model): macro-F1 ${f3(eng.resnet18_finetuned.macro_f1)}, 95% CI [${ci[0].toFixed(2)}, ${ci[1].toFixed(2)}]`,
    cv ? `5-fold CV over time blocks: ResNet-18 ${f3(cv.resnet18.mean)} ± ${f3(cv.resnet18.std)}, GoogLeNet ${f3(cv.googlenet.mean)} ± ${f3(cv.googlenet.std)}. Single splits swing widely`
       : "Split by 5-second time blocks with buffers, so near-duplicate frames never cross splits",
    "Limit: one person, weak labels. Measures alertness, not learning; fairness cannot be tested",
  ], { x: 0.6, y: 5.25, w: 12.2, h: 1.8, fontSize: 14 });
}

// ================================================================ 13. sequence models
if (trend && agg) {
  const st = trend.student_level_real, cl = trend.class_level_simulated;
  const s = content("Engagement", "Persistence beats the LSTM; the transformer cuts error by a third",
    "Student-level LSTM uses real frames. The class-level LSTM and the transformer use simulated sessions built from the real classifier's errors.");
  s.addImage(fit(R("reports/experiments/sequence/trend_student_lstm.png"), 0.6, 1.3, 7.2, 2.6));
  s.addImage(fit(R("reports/experiments/sequence/transformer_aggregator.png"), 0.6, 4.1, 7.2, 2.7));
  bullets(s, [
    `Student-level (real): last value ${f3(st.last_value)} beats the best LSTM ${f3(Math.min(st.lstm_tanh.mean, st.lstm_sigmoid.mean, st.lstm_step.mean))}: too little data, too short a horizon`,
    `Cell activation: tanh ${f3(st.lstm_tanh.mean)}, sigmoid ${f3(st.lstm_sigmoid.mean)}, step ${f3(st.lstm_step.mean)}`,
    `Class-level 5-min forecast (simulated): LSTM ${f3(cl.lstm_MAE)} vs persistence ${f3(cl.persistence_MAE)}`,
    `Transformer aggregator (simulated): MAE ${f3(agg.transformer_MAE)} vs naive mean ${f3(agg.mean_hard_label_MAE)}`,
    `Attention follows face size (r = ${f3(agg.attention_vs_face_size_corr)}): it trusts front-row faces more`,
  ], { x: 8.1, y: 1.4, w: 4.7, h: 5.4, fontSize: 14 });
}

// ================================================================ 14. latent space
if (fs.existsSync(R("reports/experiments/sequence/latent_space.png"))) {
  const lt = J("reports/experiments/sequence/latent_space.json");
  const s = content("Engagement", "Labels shape the latent space: CNN vs VAE",
    "Silhouette score by engagement class. The VAE is the generative component: it reconstructs and samples faces, and a latent traversal moves from engaged to disengaged.");
  s.addImage(fit(R("reports/experiments/sequence/latent_space.png"), 0.6, 1.3, 7.2, 3.1));
  s.addImage(fit(R("reports/experiments/sequence/vae_faces.png"), 0.6, 4.5, 7.2, 2.6));
  if (lt) {
    stat(s, 8.3, 1.5, 4.5, f3(lt.silhouette_cnn_embedding), "silhouette: ResNet-18 embedding");
    stat(s, 8.3, 3.2, 4.5, f3(lt.silhouette_vae_latent), "silhouette: VAE latent (unsupervised)", C.accent6);
  }
  txt(s, "Labels shape the space. Without them the VAE spends its capacity on pose and lighting.", { x: 8.3, y: 5.0, w: 4.5, h: 1.6, fontSize: 14 });
}

// ================================================================ real classroom video (Requirements 7-10)
pres.addSection({ title: "Real classroom video" });
if (bm) {
  const d = bm.detectors;
  const yt = d["YOLOv8n (fine-tuned)"], fr = d["Faster R-CNN (fine-tuned)"], rc = d["R-CNN (ResNet18)"], fa = d["Fast R-CNN (ResNet18)"];
  const s = content("Real classroom video", "On real classroom video, fine-tuning matters more than the detector",
    "Kaggle test classroom: 85 frames, 1,159 labelled students, never used for training or for any choice. R-CNN and Fast R-CNN were implemented from their papers. Timing: all models run in turn on the same frames, laptop CPU on battery.");
  const f = R("reports/figures/vision/benchmark_detectors.png");
  if (fs.existsSync(f)) s.addImage(fit(f, 0.6, 1.3, 12.1, 3.9));
  bullets(s, [
    `Fine-tuned on 217 classroom frames, YOLOv8n, Faster R-CNN and RetinaNet all reach mAP@0.5 of about ${yt.mAP50.toFixed(2)}; as downloaded they score 0.26 to 0.33`,
    `At equal accuracy YOLOv8n needs ${Math.round(yt.ms_per_image_median)} ms per frame, Faster R-CNN ${Math.round(fr.ms_per_image_median)} ms`,
    `R-CNN ${(rc.ms_per_image_median / 1000).toFixed(1)} s vs Fast R-CNN ${(fa.ms_per_image_median / 1000).toFixed(2)} s per frame: sharing one CNN pass. Both are capped by hand-made proposals, which find only ${pct(bm.proposal_recall_ceiling["iou0.5"], 0)} of students`,
  ], { x: 0.6, y: 5.35, w: 12.2, h: 1.6, fontSize: 14 });
}
if (ce && sq) {
  const r18 = ce.states_cnn.resnet18, t = ce.trend;
  const s = content("Real classroom video", "Per-student state is hard, and sequence models need more data",
    "ResNet18 and GoogLeNet fine-tuned for attentive / distracted / inactive on the training classrooms. Sequence models were trained on three videos and tested on the unseen test classroom with 5 seeds.");
  stat(s, 0.6, 1.35, 3.0, f3(r18.test.macro_f1), "state macro-F1, new classroom (ResNet18)", C.accent3);
  stat(s, 0.6, 3.05, 3.0, r18.test.f1[2].toFixed(2), "F1 for 'inactive' (70 training examples)", C.accent3);
  stat(s, 0.6, 4.75, 3.0, t.attentive_share_corr.toFixed(2), "trend correlation, predicted vs labelled", C.accent2);
  const f = R("reports/figures/sequence/classroom_lstm_transformer.png");
  if (fs.existsSync(f)) s.addImage(fit(f, 3.9, 1.3, 8.9, 3.6));
  const lv = sq.leave_one_video_out.mean_mae;
  bullets(s, [
    `The trend follows the ups and downs, but the level is off: ${t.pred_mean_attentive.toFixed(2)} predicted vs ${t.true_mean_attentive.toFixed(2)} labelled attentive`,
    `LSTM and Transformer do not beat the per-frame output on new videos (error ${lv["LSTM (tanh)"].toFixed(2)} and ${lv["Transformer"].toFixed(2)} vs ${lv["current frame only"].toFixed(2)}): three training videos are too few`,
    "Decision: keep a simple moving average for the trend chart",
  ], { x: 3.9, y: 5.05, w: 8.9, h: 1.9, fontSize: 14 });
}
if (fe) {
  const rb = fe.robustness, fa = fe.fairness, er = fe.errors, la = fe.latency;
  const s = content("Real classroom video", "Robustness, fairness and errors on the test classroom",
    "Pipeline: fine-tuned YOLOv8n then ResNet18 state classifier. The same 85 test frames were degraded five ways. Fairness is measured by student size in the image because the dataset has no demographic labels.");
  const f = R("reports/figures/eval/final_robustness_fairness_errors.png");
  if (fs.existsSync(f)) s.addImage(fit(f, 0.6, 1.3, 12.1, 3.2));
  bullets(s, [
    `Detection holds in low light (${f3(rb["low light (30% brightness)"].mAP50)} vs ${f3(rb.clean.mAP50)} clean); sensor noise is the weak spot (${f3(rb["sensor noise (sigma 25)"].mAP50)}, count error ${rb["sensor noise (sigma 25)"].count_mae.toFixed(1)})`,
    `The state classifier is fragile: macro-F1 ${f3(rb.clean.state_macro_f1)} clean, ${f3(rb["low light (30% brightness)"].state_macro_f1)} in low light`,
    `Back-row students are found ${pct(fa.detection_recall_by_size["small (back rows)"], 0)} of the time, front-row students ${pct(fa.detection_recall_by_size["large (front rows)"], 0)}`,
    `Count error is ${rb.clean.count_mae.toFixed(1)} students per frame; ${er.false_boxes["poorly placed box on a student"]} boxes are on a student but loosely placed (labels mix head and body boxes)`,
    `Classroom pipeline: ${Math.round(la.classroom_pipeline.total.p95)} ms per frame at p95 on a laptop CPU on battery`,
  ], { x: 0.6, y: 4.65, w: 12.2, h: 2.3, fontSize: 13 });
}

// ================================================================ 15. product demo
pres.addSection({ title: "Product and decision" });
{
  const s = content("Product and decision", "The integrated product, end to end",
    "Demo: a synthetic classroom composed of LFW photos of 12 enrolled people (2 absent, 1 stranger), run through the full analyzer.");
  const frame = R("reports/demo/demo_frame.png");
  if (fs.existsSync(frame)) s.addImage(fit(frame, 0.6, 1.3, 7.6, 4.6));
  if (demo && demo.demo_check) {
    stat(s, 8.6, 1.4, 4.2, `${demo.demo_check.correct_register_entries}/${demo.demo_check.of}`, "register entries correct in the demo");
    stat(s, 8.6, 3.1, 4.2, `${Math.round(demo.mean_latency_ms)} ms`, "per frame: 13 faces, i3 on battery (1.2 GHz)", C.accent2);
  }
  if (lat) {
    txt(s, `Stage latency (classroom images, mean): people ${Math.round(lat.detect.mean)} ms · faces ${Math.round(lat.faces.mean)} ms · engagement ${Math.round(lat.engagement.mean)} ms · total ${Math.round(lat.total.mean)} ms (p95 ${Math.round(lat.total.p95)} ms) · peak RAM ${Math.round(lat.peak_rss_mb)} MB`,
      { x: 0.6, y: 6.1, w: 12.2, h: 0.8, fontSize: 13, color: C.accent6 });
  }
  txt(s, "Streamlit GUI: upload a photo or video and get a head count, seat map, engagement colours and a downloadable register.",
    { x: 8.6, y: 4.85, w: 4.2, h: 1.2, fontSize: 14 });
}

// ================================================================ 16. scorecard
{
  const s = content("Product and decision", "Scorecard against the requirements",
    "Pass/fail against docs/REQUIREMENTS.md. 'Partial' means met on the proxy dataset with a caveat that blocks unconditional deployment.");
  const y8s = det ? det["yolov8s"] : null;
  const ok = { text: "PASS", options: { bold: true, color: HEX.accent1 } };
  const part = { text: "PARTIAL", options: { bold: true, color: HEX.accent2 } };
  const fail = { text: "FAIL", options: { bold: true, color: HEX.accent3 } };
  const H = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.accent1 } } });
  const rows = [[H("Requirement"), H("Target"), H("Result"), H("Status")]];
  const caveat = { R4: true, R7: true };   // met on a proxy dataset with a documented caveat
  const fmtV = (v) => (v == null ? "not run" : Math.abs(v) < 10 ? v.toFixed(3) : String(Math.round(v)));
  if (acc) {
    acc.forEach((r) => rows.push([`${r.id}  ${r.metric}`, r.target, fmtV(r.measured),
      r.status === "PASS" ? (caveat[r.id] ? part : ok) : fail]));
  }
  s.addTable(rows, { x: 0.6, y: 1.3, w: 12.1, colW: [6.4, 2.2, 1.7, 1.8], fontSize: 12, fontFace: THEME.bodyFontFace, color: HEX.dk1,
    border: { type: "solid", pt: 0.5, color: "D5DEDB" }, rowH: 0.37 });
  caption(s, "R1 to R9: public proxy datasets. R10 to R13: real classroom video (Kaggle test classroom). PARTIAL = number met, with a documented caveat.", 0.6, 6.9, 12.1);
}

// ================================================================ 17. go / no-go
{
  const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Product and decision" });
  n++;
  s.addText("Recommendation: GO for a supervised pilot, NO-GO for unsupervised rollout", { isTextBox: true, x: 0.8, y: 0.6, w: 11.7, h: 1.3,
    fontSize: 32, bold: true, color: C.background1, fontFace: THEME.headFontFace, margin: 0, valign: "top" });
  const cols = [
    ["GO: attendance pilot", C.accent1, ["Face attendance meets every target on LFW", "Teacher confirms each register", "One classroom, consenting students, one term", "Measure recall and fairness on our own data"]],
    ["CONDITIONAL: counting and seating", C.accent2, ["Classroom video: count off by " + (fe ? fe.robustness.clean.count_mae.toFixed(1) : "1.5") + " students per frame, after fine-tuning", "Back rows found " + (fe ? pct(fe.fairness.detection_recall_by_size["small (back rows)"], 0) : "46%") + " vs front rows " + (fe ? pct(fe.fairness.detection_recall_by_size["large (front rows)"], 0) : "80%") + ": mount the camera high at the front", "Replace the seat heuristic with per-room calibration"]],
    ["NO-GO: individual engagement", C.accent3, ["New classroom: state macro-F1 " + (fe ? fe.robustness.clean.state_macro_f1.toFixed(2) : "0.47") + " against a target of 0.60", "Report only class-level trends: direction, not percentages", "Never grade or discipline individuals", "Needs many more labelled classroom videos first"]],
  ];
  cols.forEach(([h, col, items], i) => {
    const x = 0.8 + i * 4.0;
    s.addShape(pres.ShapeType.roundRect, { x, y: 2.3, w: 3.7, h: 3.9, rectRadius: 0.12, fill: { color: C.background1 }, line: { color: C.background1 } });
    s.addText(h, { isTextBox: true, x: x + 0.3, y: 2.5, w: 3.1, h: 0.8, fontSize: 18, bold: true, color: col, margin: 0, valign: "top" });
    s.addText(items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1, paraSpaceAfter: 8 } })),
      { isTextBox: true, x: x + 0.3, y: 3.4, w: 3.1, h: 3.0, fontSize: 14, color: C.text1, margin: 0, valign: "top" });
  });
  s.addNotes("The evidence supports a supervised attendance pilot. Engagement must stay aggregate until it is trained on representative, consented data.");
}

// ================================================================ 18. responsible use / next steps
{
  const s = content("Product and decision", "Responsible use and next steps",
    "These limits are also in docs/REQUIREMENTS.md and docs/MODEL_CARD.md.");
  txt(s, "Non-negotiable limits", { x: 0.6, y: 1.4, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.accent3 });
  bullets(s, ["Opt-in consent; non-enrolled people are never identified",
    "Store 128-d embeddings, not photos; delete on request",
    "Teacher confirms every register",
    "Engagement is class-level only: no grading or discipline",
    "Not for exam proctoring or behaviour scoring"], { x: 0.6, y: 1.95, w: 5.8, h: 4.5 });
  txt(s, "Next steps", { x: 7.0, y: 1.4, w: 5.8, h: 0.4, fontSize: 20, bold: true, color: C.accent1 });
  bullets(s, ["One-term pilot in one consenting classroom",
    "Collect consented classroom CCTV to fine-tune detection",
    "Build a multi-student engagement dataset with human labels",
    "Per-room seat calibration tool",
    "Term-by-term fairness and accuracy audit"], { x: 7.0, y: 1.95, w: 5.8, h: 4.5 });
}

const out = R("docs/presentation.pptx");
pres.writeFile({ fileName: out }).then(async () => {
  const applyThemePath = process.argv[2];
  if (applyThemePath) {
    const { applyTheme } = require(applyThemePath);
    await applyTheme(out, THEME);
  }
  console.log(`wrote ${out} (${n} slides)`);
});

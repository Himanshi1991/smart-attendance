"""Lightweight GUI:  streamlit run app/streamlit_app.py

Modes (sidebar):
  * Live attendance   - webcam: recognise enrolled students and mark them present with a timestamp
  * Enroll students   - take 3-5 photos with the camera (or upload) and add a student to the roster
  * Analyze photo/video - full analysis of an uploaded classroom image or video
Run locally, everything stays on this computer. Online (Streamlit Community Cloud) the live page uses the
viewer's browser camera, because the cloud server has no camera of its own.
"""
from __future__ import annotations

import base64
import hashlib
import os
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:          # online there is no `pip install -e .`; import the package from src/
    sys.path.insert(0, str(_SRC))

from smartattend.attendance.live import (LiveAttendance, check_enrolment_photo, draw_live, rebuild_roster,
                                         restore_roster_backup, roster_backup, safe_student_id,
                                         save_student_photos)
from smartattend.attendance.roster import Roster
from smartattend.attendance.session import AttendanceSession, iter_frames
from smartattend.utils import load_config

st.set_page_config(page_title="Smart Attendance & Engagement", layout="wide")
cfg = load_config()
ROOT = Path(cfg["paths"]["models_dir"]).parent
ROSTER_DIR = Path(cfg["paths"]["roster_dir"])
CLASS_ROSTER = Path(cfg["paths"]["roster_file"])
DEMO_ROSTER = ROOT / "data" / "demo" / "roster.npz"
REGISTER_DIR = ROOT / "reports" / "attendance"
# Streamlit Community Cloud runs apps from /mount/src/<repo>; SMARTATTEND_CLOUD=1 forces online mode for testing.
ON_CLOUD = str(Path(__file__).resolve()).startswith("/mount/src") or os.environ.get("SMARTATTEND_CLOUD") == "1"


@st.cache_resource
def get_engine():
    from smartattend.attendance.face import FaceEngine, ensure_models

    return FaceEngine(*ensure_models(cfg["paths"]["models_dir"]), score_threshold=cfg["face"]["detector_score_threshold"])


@st.cache_resource
def get_analyzer(detector: str, roster_path: str | None, roster_mtime: float):
    from smartattend.pipeline import ClassroomAnalyzer

    roster = Roster.load(roster_path) if roster_path and Path(roster_path).exists() else None
    cfg["product"]["detector"] = detector
    return ClassroomAnalyzer(cfg, roster), roster


def load_roster(path: Path) -> Roster | None:
    return Roster.load(path) if path.exists() else None


def decode(data: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def enrolled_students() -> list[str]:
    return sorted(p.name for p in ROSTER_DIR.iterdir() if p.is_dir()) if ROSTER_DIR.exists() else []


def now() -> datetime:
    """Current time in the school's time zone (configs/default.yaml: timezone). Cloud servers run on UTC."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(cfg.get("timezone", "Asia/Kolkata"))).replace(tzinfo=None)
    except Exception:          # no time-zone database on this machine: fall back to the computer's clock
        return datetime.now()


def register_download(rows: list[dict], key: str) -> None:
    df = pd.DataFrame(rows)
    st.download_button("Download register (CSV)", df.to_csv(index=False),
                       f"attendance_{now():%Y-%m-%d}.csv", "text/csv", key=key)


# Live camera in the browser (app/components/live_camera): smooth local preview, frames sent to the app over
# Streamlit's own connection, face boxes drawn back on the video. Works locally and online, on any network.
_live_camera = components.declare_component("live_camera",
                                            path=str(Path(__file__).resolve().parent / "components" / "live_camera"))


def face_boxes(faces, labels, scores, marked: set[str]) -> list[dict]:
    """Boxes for the live camera overlay, coloured like draw_live (green present, orange confirming, red unknown)."""
    out = []
    for f, lbl, sc in zip(faces, labels, scores):
        x, y, w, h = (int(v) for v in f.box)
        if lbl is None:
            color, text = "#e00000", f"unknown {sc:.2f}"
        elif lbl in marked:
            color, text = "#00a000", f"{lbl}  PRESENT"
        else:
            color, text = "#f0a000", f"{lbl}  confirming..."
        out.append({"x": x, "y": y, "w": w, "h": h, "color": color, "text": text})
    return out


st.title("Smart Attendance & Classroom Engagement Analyzer")

with st.sidebar:
    mode = st.radio("Mode", ["Live attendance", "Enroll students", "Analyze photo/video"])


# =====================================================================================  ENROLL
if mode == "Enroll students":
    st.header("Enroll a student")
    st.write("Take **3 to 5 photos** of the student facing the camera in good light: look straight, "
             "then slightly left and slightly right. Only the student should be in the frame.")
    engine = get_engine()
    st.session_state.setdefault("enroll_photos", [])

    c1, c2 = st.columns([1, 1])
    with c1:
        sid_raw = st.text_input("Student ID and name", placeholder="e.g. 22CS001 Anushka")
        sid = safe_student_id(sid_raw)
        if sid_raw and sid != sid_raw.strip():
            st.caption(f"Saved as: `{sid}`")
        shot = st.camera_input("Camera")
        if shot is not None and st.button("Add this photo", type="primary"):
            img = decode(shot.getvalue())
            ok, msg = check_enrolment_photo(engine, img)
            (st.success if ok else st.error)(msg)
            if ok:
                st.session_state["enroll_photos"].append(img)
        files = st.file_uploader("...or upload photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if files and st.button("Add uploaded photos"):
            for f in files:
                img = decode(f.getvalue())
                ok, msg = check_enrolment_photo(engine, img)
                (st.success if ok else st.error)(f"{f.name}: {msg}")
                if ok:
                    st.session_state["enroll_photos"].append(img)

    with c2:
        photos = st.session_state["enroll_photos"]
        st.subheader(f"Collected photos: {len(photos)}")
        if photos:
            st.image([p[:, :, ::-1] for p in photos], width=140)
            if st.button("Clear photos"):
                st.session_state["enroll_photos"] = []
                st.rerun()
        consent = st.checkbox("The student has given consent to be enrolled for face-recognition attendance")
        can_save = bool(sid) and len(photos) >= 3 and consent
        if st.button("Save student", type="primary", disabled=not can_save):
            save_student_photos(ROSTER_DIR, sid, photos)
            with st.spinner("Updating roster..."):
                roster = rebuild_roster(ROSTER_DIR, CLASS_ROSTER, engine)
            st.session_state["enroll_photos"] = []
            st.session_state.pop("live", None)
            st.success(f"Enrolled {sid}. The roster now has {len(roster)} student(s).")
        if not can_save:
            st.caption("Needs: a student ID, at least 3 photos, and consent.")

    st.divider()
    st.subheader("Enrolled students")
    students = enrolled_students()
    if not students:
        st.info("No students enrolled yet.")
    else:
        roster = load_roster(CLASS_ROSTER)
        counts = dict(zip(roster.student_ids, roster.n_images)) if roster else {}
        st.dataframe(pd.DataFrame({"student": students, "photos used": [counts.get(s, 0) for s in students]}),
                     width="stretch", hide_index=True)
        rm = st.selectbox("Remove a student (deletes their photos and embedding)", ["-"] + students)
        if rm != "-" and st.button(f"Remove {rm}"):
            shutil.rmtree(ROSTER_DIR / rm, ignore_errors=True)
            if enrolled_students():
                rebuild_roster(ROSTER_DIR, CLASS_ROSTER, engine)
            else:
                CLASS_ROSTER.unlink(missing_ok=True)
            st.session_state.pop("live", None)
            st.rerun()

    st.divider()
    st.subheader("Back up / restore students")
    st.caption("Download a backup after enrolling. If the students are ever gone (the online app's server deletes "
               "its files when the app restarts), upload the backup here to bring them back. The backup holds the "
               "students' photos: keep it private.")
    k1, k2 = st.columns(2)
    with k1:
        if students:
            st.download_button("Download backup of students", roster_backup(ROSTER_DIR),
                               f"students_backup_{now():%Y-%m-%d}.zip", "application/zip")
        else:
            st.caption("Nothing to back up yet.")
    with k2:
        backup = st.file_uploader("Restore from a backup (.zip)", type=["zip"], key="restore_zip")
        if backup is not None and st.button("Restore students", type="primary"):
            try:
                restored = restore_roster_backup(ROSTER_DIR, backup.getvalue())
            except zipfile.BadZipFile:
                restored = None
                st.error("That file is not a valid backup .zip.")
            if restored == []:
                st.error("No student photos found in that file.")
            elif restored:
                with st.spinner("Updating roster..."):
                    roster = rebuild_roster(ROSTER_DIR, CLASS_ROSTER, engine)
                st.session_state.pop("live", None)
                # rerun so the student list and backup button above include the restored students
                st.session_state["restore_msg"] = (f"Restored {len(restored)} student(s): {', '.join(restored)}. "
                                                   f"The roster now has {len(roster)} student(s).")
                st.rerun()
        if "restore_msg" in st.session_state:
            st.success(st.session_state.pop("restore_msg"))


# =====================================================================================  LIVE
elif mode == "Live attendance":
    st.header("Live attendance from the camera")
    roster = load_roster(CLASS_ROSTER)
    with st.sidebar:
        use_demo = (st.checkbox("Use the demo roster (12 LFW people) instead", value=roster is None)
                    if DEMO_ROSTER.exists() else False)
        SRC_LOCAL, SRC_LIVE, SRC_PHOTO = ("This computer's camera (live video)", "Browser camera (live video)",
                                          "Browser camera (take a photo)")
        sources = ([] if ON_CLOUD else [SRC_LOCAL]) + [SRC_LIVE, SRC_PHOTO]
        source = st.radio("Camera", sources, help="'This computer's camera' uses a camera attached to the computer "
                          "running the app. The browser camera works anywhere, including the online app: live "
                          "video, or one still photo at a time.")
        browser = source == SRC_PHOTO          # one still photo at a time
        live_video = source == SRC_LIVE        # live video from the browser (component)
        cam = st.number_input("Camera index", 0, 5, 0) if source == SRC_LOCAL else 0
        threshold = st.slider("Match threshold (higher = stricter)", 0.30, 0.60, float(cfg["face"]["match_threshold"]), 0.01)
    if use_demo:
        roster = load_roster(DEMO_ROSTER)
    if roster is None or len(roster) == 0:
        st.warning("No students enrolled yet. Switch to **Enroll students** in the sidebar first.")
        st.stop()

    key = (str(CLASS_ROSTER if not use_demo else DEMO_ROSTER), tuple(roster.student_ids), source)
    if st.session_state.get("live_key") != key or "live" not in st.session_state:   # "live" is cleared after enrolment changes
        # live video: present after 3 of 5 consecutive frames; photo: each photo is a deliberate, still shot -> 1 match
        st.session_state["live"] = (LiveAttendance(roster, confirm_hits=1, window=1) if browser
                                    else LiveAttendance(roster, confirm_hits=3, window=5))
        st.session_state["live_key"] = key
        st.session_state.pop("cam_boxes", None)
    live: LiveAttendance = st.session_state["live"]

    def count_recognitions(labels, scores):
        newly = live.update(labels, scores, now())
        for sid in newly:
            st.toast(f"{sid} marked present at {live.marked[sid]['time']}")
        if newly:
            live.save(REGISTER_DIR, now())

    if live_video:
        # The component's latest frame is in session_state before anything is drawn: process it FIRST, so the
        # counts, register and boxes drawn in this run already include it (frames arrive several times a second;
        # drawing first and filling in later would leave those parts blank most of the time). Each frame has an
        # id and is counted once.
        frame_msg = st.session_state.get("live_cam")
        if isinstance(frame_msg, dict) and frame_msg.get("id") and frame_msg["id"] != st.session_state.get("cam_ack"):
            frame = decode(base64.b64decode(frame_msg["image"].split(",", 1)[1]))
            if frame is not None:
                faces = get_engine().analyze(frame)
                labels, scores = (roster.match(np.stack([f.embedding for f in faces]), threshold)
                                  if faces else ([], []))
                count_recognitions(labels, scores)
                t = time.time()
                st.session_state["cam_fps"] = 1.0 / max(t - st.session_state.get("cam_t", t - 1.0), 1e-3)
                st.session_state["cam_t"] = t
                st.session_state["cam_boxes"] = {
                    "boxes": face_boxes(faces, labels, scores, set(live.marked)),
                    "frame_w": int(frame.shape[1]), "frame_h": int(frame.shape[0]),
                    "status": (f"{len(faces)} face(s) in view · {sum(l is not None for l in labels)} recognised · "
                               f"{len(live.marked)}/{len(roster)} present")}
            st.session_state["cam_ack"] = frame_msg["id"]


    top = st.columns([1, 1, 1, 2])
    top[0].metric("Enrolled", len(roster))
    present_ph, absent_ph = top[1].empty(), top[2].empty()      # filled at the end, after any new frame is counted
    with top[3]:
        # the same "Camera on" switch for this computer's camera and for the browser's live camera
        cam_on = source in (SRC_LOCAL, SRC_LIVE) and st.toggle("Camera on", value=False)
        run = source == SRC_LOCAL and cam_on
        if live_video and not cam_on:
            st.session_state.pop("cam_boxes", None)      # no stale boxes / status when the camera is off
        b1, b2 = st.columns(2)
        if b1.button("Save register"):
            st.success(f"Saved {live.save(REGISTER_DIR, now())}")
        if b2.button("New session (reset)"):
            st.session_state.pop("live", None)
            st.session_state.pop("last_photo", None)
            st.session_state.pop("cam_boxes", None)
            st.rerun()

    left, right = st.columns([3, 2])
    if browser:
        shot = left.camera_input("Take a photo of the class (students facing the camera)")
    cam_slot = left.container()          # the live camera is drawn here at the END of the run (see below)
    frame_ph = left.empty()
    status_ph = left.empty()
    reg_ph = right.empty()
    if browser:
        right.caption("Every enrolled student recognised in a photo is marked present. Take more photos to catch "
                      "students who were turned away. Unknown faces are never stored.")
    elif live_video:
        right.caption("Turn **Camera on** and allow camera access when the browser asks. A student is marked "
                      "present after being recognised in 3 of 5 consecutive frames. Unknown faces are never stored.")
    else:
        right.caption("A student is marked present after being recognised in 3 of 5 consecutive frames. "
                      "Unknown faces are never stored. The register is auto-saved to reports/attendance/.")

    if browser and shot is not None:
        data = shot.getvalue()
        digest = hashlib.sha1(data).hexdigest()
        frame = decode(data)
        engine = get_engine()
        faces = engine.analyze(frame)
        labels, scores = (roster.match(np.stack([f.embedding for f in faces]), threshold) if faces else ([], []))
        if st.session_state.get("last_photo") != digest:     # a rerun shows the same photo again: count it once
            st.session_state["last_photo"] = digest
            count_recognitions(labels, scores)
        frame_ph.image(draw_live(frame, faces, labels, scores, set(live.marked))[:, :, ::-1], channels="RGB",
                       width="stretch")
        known = sum(l is not None for l in labels)
        status_ph.caption(f"{len(faces)} face(s) in the photo · {known} recognised · "
                          f"{len(live.marked)}/{len(roster)} present")
    elif browser:
        frame_ph.info("Click **Take Photo** above. Allow camera access if your browser asks.")
    if live_video and "cam_boxes" in st.session_state:
        status_ph.caption(st.session_state["cam_boxes"]["status"] +
                          f" · {st.session_state.get('cam_fps', 0):.1f} frames/s checked")
    present_ph.metric("Present", len(live.marked))
    absent_ph.metric("Absent", len(roster) - len(live.marked))
    reg_ph.dataframe(pd.DataFrame(live.register()), width="stretch", hide_index=True)
    with right:
        register_download(live.register(), "live_register")

    if live_video and cam_on:
        # Drawn last: the component sends its next frame as soon as it sees this ack, which starts a new run; by
        # then the counts and register above are already on the page. Switching "Camera on" off removes the
        # component, which stops the browser camera.
        boxes = st.session_state.get("cam_boxes", {})
        with cam_slot:
            _live_camera(key="live_cam", default=None, ack=st.session_state.get("cam_ack"),
                         boxes=boxes.get("boxes", []), frame_w=boxes.get("frame_w"), frame_h=boxes.get("frame_h"),
                         capture_width=640, controls=False)

    if run:
        engine = get_engine()
        cap = cv2.VideoCapture(int(cam), cv2.CAP_DSHOW)
        if not cap.isOpened():
            st.error(f"Could not open camera {cam}. Close other apps using the camera, or try another index.")
        else:
            try:
                fps_t, n = time.time(), 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        status_ph.error("Camera stopped sending frames.")
                        break
                    faces = engine.analyze(frame)
                    labels, scores = (roster.match(np.stack([f.embedding for f in faces]), threshold)
                                      if faces else ([], []))
                    newly = live.update(labels, scores, now())
                    for sid in newly:
                        st.toast(f"{sid} marked present at {live.marked[sid]['time']}")
                        live.save(REGISTER_DIR, now())
                    if newly:
                        reg_ph.dataframe(pd.DataFrame(live.register()), width="stretch", hide_index=True)
                        present_ph.metric("Present", len(live.marked))
                        absent_ph.metric("Absent", len(roster) - len(live.marked))
                    vis = draw_live(frame, faces, labels, scores, set(live.marked))
                    frame_ph.image(vis[:, :, ::-1], channels="RGB", width="stretch")
                    n += 1
                    if n % 10 == 0:
                        fps = 10 / (time.time() - fps_t)
                        fps_t = time.time()
                        status_ph.caption(f"{fps:.1f} frames/s · {len(faces)} face(s) in view · "
                                          f"{len(live.marked)}/{len(roster)} present")
            finally:
                cap.release()
    elif source in (SRC_LOCAL, SRC_LIVE) and not cam_on:
        frame_ph.info("Turn **Camera on** to start taking attendance.")


# =====================================================================================  ANALYZE
else:
    st.header("Analyze a classroom photo or video")
    with st.sidebar:
        detectors = ["yolov8s", "yolov8n", "yolov8n-classroom"]
        if not ON_CLOUD:      # the R-CNN family is 10x slower and too heavy for the free online server
            detectors += ["fasterrcnn_resnet50_fpn", "retinanet_resnet50_fpn"]
        detector = st.selectbox("Person detector", detectors)
        roster_options = ["class roster"] + (["demo roster (12 LFW identities)"] if DEMO_ROSTER.exists() else [])
        roster_choice = st.radio("Roster", roster_options,
                                 index=0 if CLASS_ROSTER.exists() or len(roster_options) == 1 else 1)
        roster_file = CLASS_ROSTER if roster_choice == "class roster" else DEMO_ROSTER
        st.write(f"Roster: {'loaded' if roster_file.exists() else 'not found'}")
        demo_video = ROOT / "data" / "demo" / "class_demo.mp4"
        if demo_video.exists():
            st.download_button("Download demo classroom video", demo_video.read_bytes(), "class_demo.mp4")
        every = st.slider("Process every Nth video frame", 1, 60, 15)
        max_frames = st.slider("Max frames to analyse", 1, 200, 40)

    mtime = roster_file.stat().st_mtime if roster_file.exists() else 0.0
    analyzer, roster = get_analyzer(detector, str(roster_file) if roster_file.exists() else None, mtime)
    upload = st.file_uploader("Classroom image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

    if upload is not None:
        tmp = Path(tempfile.mkdtemp()) / f"input{Path(upload.name).suffix.lower()}"
        tmp.write_bytes(upload.read())
        session = AttendanceSession(roster, cfg["attendance"]["min_frames_present"],
                                    cfg["attendance"].get("min_fraction_present", 0.5)) if roster else None
        rows, last_vis, last_res = [], None, None
        progress = st.progress(0.0)
        for n, (idx, frame) in enumerate(iter_frames(tmp, every)):
            if n >= max_frames:
                break
            res = analyzer.analyze_frame(frame)
            if session:
                session.update(idx, res["_labels"], [s["similarity"] for s in res["students"]])
            nf = max(sum(res["states"][s] for s in ("attentive", "distracted", "inactive")), 1)
            rows.append({"frame": idx, "persons": res["persons"], "faces": res["faces"],
                         "seats occupied": res["seats_occupied"], "seats empty": res["seats_empty"],
                         **{s: res["states"][s] / nf for s in ("attentive", "distracted", "inactive")},
                         "class_engagement": res["class_engagement"].get("engaged_fraction_transformer"),
                         "latency_ms": res["timing_ms"]["total"]})
            last_vis, last_res = analyzer.draw(frame, res), res
            progress.progress(min(1.0, (n + 1) / max_frames))
        progress.empty()

        if last_res is None:
            st.error("Could not read any frame from the upload.")
        else:
            stc = last_res["states"]
            seen = max(stc["attentive"] + stc["distracted"] + stc["inactive"], 1)
            st.subheader("1 · Students present   2 · Seats")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("People detected", last_res["persons"])
            c2.metric("Seats occupied", last_res["seats_occupied"])
            c3.metric("Seats empty", "-" if last_res["seats_empty"] is None else last_res["seats_empty"],
                      help="Chairs, couches or benches with nobody on them (YOLO detectors only)")
            c4.metric("Rows", last_res["rows"])
            st.subheader("3 · Attentive / distracted / inactive")
            if stc["unclear"]:
                st.caption(f"{stc['unclear']} face(s) under 30 px are too small to judge and are left out of the percentages.")
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Attentive", f"{stc['attentive']} ({stc['attentive'] / seen:.0%})")
            d2.metric("Distracted", f"{stc['distracted']} ({stc['distracted'] / seen:.0%})", help="Alert but head turned away")
            d3.metric("Inactive", f"{stc['inactive']} ({stc['inactive'] / seen:.0%})", help="Eyes closing / drowsy")
            ce = last_res["class_engagement"].get("engaged_fraction_transformer")
            d4.metric("Class engagement (transformer)", "-" if ce is None else f"{ce:.0%}")
            st.image(last_vis[:, :, ::-1], caption="seat map (coloured by row, grey = empty seat) + faces "
                     "(green attentive / orange distracted / red inactive / grey too small to judge)", width="stretch")
            tl = pd.DataFrame(rows)
            if len(tl) > 1:
                st.subheader("4 · Engagement over time")
                st.line_chart(tl.set_index("frame")[["attentive", "distracted", "inactive"]],
                              color=["#2e9e44", "#f59e0b", "#d43c3c"])
                st.line_chart(tl.set_index("frame")[["persons", "seats occupied", "seats empty"]])
            st.caption("5 · Accuracy of each component on held-out test data: see reports/RESULTS.md "
                       "(detection mAP 0.816, count error 0.99, alertness macro-F1 0.68-0.86). "
                       "'Distracted' uses a head-turn rule that has not been validated against labelled data.")
            st.subheader("Students in the last frame")
            st.dataframe(pd.DataFrame([{k: v for k, v in s.items() if k != "face_box"} for s in last_res["students"]]),
                         width="stretch")
            if session:
                st.subheader("Attendance register (suggested - please confirm)")
                reg = pd.DataFrame(session.report())
                st.dataframe(reg, width="stretch")
                st.download_button("Download attendance CSV", reg.to_csv(index=False), "attendance.csv")
            st.caption(f"Mean latency {tl['latency_ms'].mean():.0f} ms per processed frame on this machine.")

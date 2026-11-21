# Putting the app online (Streamlit Community Cloud)

## 1. Deploy

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. **Create app** → repository `<your-account>/smart-attendance`, branch `main`, main file path
   `app/streamlit_app.py`, App URL e.g. `smart-attendance`.
3. **Deploy**. The first start takes 5–10 minutes (it installs PyTorch). Packages come from
   `app/requirements.txt`, system libraries from `packages.txt`.

## 2. Live video online

Nothing to set up. In **Live attendance**, choose **Browser camera (live video)** (the first option online) and
allow camera access. The browser shows the camera itself, so the video is smooth; frames are sent to the app over
Streamlit's own connection (as fast as the server can check them, usually several per second) and the names and
boxes are drawn on the video. It works on any network: no relay, token or Secrets are needed.

**Browser camera (take a photo)** remains available: one still photo at a time.

## 3. Differences from running on the classroom computer

| | classroom computer (`start_app.bat`) | online |
|---|---|---|
| camera | the computer's own camera (live video), or the browser camera | the browser camera (live video or photo); the server has no camera of its own |
| enrolled students | kept on disk | deleted when the app restarts: use **Enroll students → Download backup of students**, and **Restore students** after a restart |
| registers | saved in `reports/attendance/` | saved until the app restarts; use **Download register (CSV)** |
| detectors | all five | the three YOLO models (the R-CNN family is too heavy for the free server) |
| who can open it | only this computer | anyone with the link, unless you restrict viewers in the app's **Settings → Sharing** |

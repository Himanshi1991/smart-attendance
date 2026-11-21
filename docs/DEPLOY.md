# Putting the app online (Streamlit Community Cloud)

## 1. Deploy

1. Sign in at https://share.streamlit.io with the GitHub account that owns the repository.
2. **Create app** → repository `<your-account>/smart-attendance`, branch `main`, main file path
   `app/streamlit_app.py`, App URL e.g. `smart-attendance`.
3. **Deploy**. The first start takes 5–10 minutes (it installs PyTorch). Packages come from
   `app/requirements.txt`, system libraries from `packages.txt`.

## 2. Live video online: add a TURN relay (once)

Live video from the browser uses WebRTC. The cloud server sits behind a firewall, so without a relay the video
usually never connects ("Connection is taking longer than expected"). A **TURN relay** passes the video through.
streamlit-webrtc gets one by itself when it finds a key; the simplest is a free Hugging Face token:

1. Create a free account at https://huggingface.co.
2. Open https://huggingface.co/settings/tokens → **Create new token** → type **Read** → any name → **Create**.
   Copy the token (it starts with `hf_`).
3. On https://share.streamlit.io, open the app's **⋮ → Settings → Secrets**, paste one line and **Save**:

   ```toml
   HF_TOKEN = "hf_paste_your_token_here"
   ```

4. Reboot the app (**⋮ → Reboot app**). In **Live attendance**, *Browser camera (live video)* is now the first
   option; click **START** and allow the camera.

Other providers also work (Cloudflare Realtime TURN, Twilio): put their keys in Secrets instead
(`CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_API_TOKEN`, or `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN`; Twilio
also needs `twilio` added to `app/requirements.txt`).

Keep the token secret: put it only in Streamlit's Secrets, never in a file in the repository.

Without a relay, **Browser camera (take a photo)** always works: take a photo of the class, everyone recognised is
marked present.

## 3. Differences from running on the classroom computer

| | classroom computer (`start_app.bat`) | online |
|---|---|---|
| camera | the computer's own camera (live video), or the browser camera | the browser camera only (the server has no camera) |
| enrolled students | kept on disk | deleted when the app restarts: use **Enroll students → Download backup of students**, and **Restore students** after a restart |
| registers | saved in `reports/attendance/` | saved until the app restarts; use **Download register (CSV)** |
| detectors | all five | the three YOLO models (the R-CNN family is too heavy for the free server) |
| who can open it | only this computer | anyone with the link, unless you restrict viewers in the app's **Settings → Sharing** |

@echo off
rem Double-click this file to start the Smart Attendance app.
rem The app stops when the computer restarts or when this window is closed, so start it again with this file.
cd /d "%~dp0"
echo Starting Smart Attendance... the browser opens by itself in a few seconds.
echo Keep this window open while you use the app. Close it to stop the app.
echo.
python -m streamlit run app\streamlit_app.py --server.port 8501 --server.address localhost
echo.
echo The app has stopped. If you see an error above, copy it and ask for help.
pause

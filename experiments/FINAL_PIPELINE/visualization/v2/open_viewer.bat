@echo off
rem Serves the visualization folder on http://127.0.0.1:8766 and opens viewer v2. Close the server window to stop.
cd /d "%~dp0.."
start "Scan2Scene viewer v2 server" "Q:\OPEN_CV_SEM5\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe" -m http.server 8766 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8766/v2/viewer.html

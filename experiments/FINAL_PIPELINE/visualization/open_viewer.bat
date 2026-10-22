@echo off
rem Serves this folder on http://127.0.0.1:8765 and opens the GLB viewer. Close the server window to stop.
cd /d "%~dp0"
start "Scan2Scene viewer server" "Q:\OPEN_CV_SEM5\3d-reconstruction\3d-reconstruction\venv\Scripts\python.exe" -m http.server 8765 --bind 127.0.0.1
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8765/viewer.html

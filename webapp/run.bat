@echo off
echo ==================================================
echo   Image Deblurring Website - Starting...
echo   Jangan tutup window ini selagi website berjalan
echo ==================================================
echo.
echo [INFO] Mengaktifkan virtual environment...
echo [INFO] Memuat semua model (Restormer, Real-ESRGAN, DiffIR)...
echo [INFO] Ini mungkin butuh beberapa menit pertama kali...
echo.

set PYTHONUNBUFFERED=1
cd /d "%~dp0"
call "%~dp0venv\Scripts\activate.bat"

python -u app.py

echo.
echo [INFO] Server berhenti. Tekan tombol apapun untuk tutup...
taskkill /F /IM python.exe /T >nul 2>&1
pause

@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Starting CRAS Defect Detection Web Application...
echo ========================================================
echo.

if not exist ".venv\Scripts\streamlit.exe" (
    echo [ERROR] Virtual environment or Streamlit not found.
    echo Please run run_cras.bat first.
    pause
    exit /b 1
)

:: Free port 8501 if an old/zombie instance is occupying it
echo [INFO] Checking if port 8501 is already in use...
powershell -NoProfile -Command "try { Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction Stop | ForEach-Object { Write-Host ('[INFO] Releasing port 8501 from old PID ' + $_.OwningProcess); Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } } catch {}; exit 0"

echo Opening web browser at http://localhost:8501 ...
start http://localhost:8501
".venv\Scripts\streamlit.exe" run app.py --server.port 8501
pause

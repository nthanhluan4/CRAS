@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Creating one...
    python -m venv .venv
    echo Installing requirements...
    .venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    .venv\Scripts\pip.exe install click matplotlib opencv-python pandas openpyxl scikit-image scikit-learn scipy tensorboard timm tqdm
)

echo Running CRAS via PowerShell...
powershell -ExecutionPolicy Bypass -File "%~dp0run_cras.ps1" %*

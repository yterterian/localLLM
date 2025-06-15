@echo off
REM ==========================================
REM Cline Local LLM System - Windows Setup
REM ==========================================

echo Setting up Local LLM System...

REM Check for Python venv in project root
if not exist ..\.venv (
    echo Creating Python virtual environment in .venv...
    python -m venv ..\.venv
    if errorlevel 1 (
        echo Failed to create virtual environment. Please ensure Python is installed and on PATH.
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Python virtual environment (.venv) already exists.
)

REM Instruct user to activate venv before continuing
echo.
echo Please activate the virtual environment before continuing:
echo   For cmd:    ..\.venv\Scripts\activate
echo   For PowerShell: ..\.venv\Scripts\Activate.ps1
echo After activation, re-run this script to complete setup.
pause
exit /b

REM The following lines will only run if the user manually continues after activating venv

REM Create directories
mkdir logs
mkdir vector_stores
mkdir data
mkdir data\projects

REM Install Python dependencies
echo Installing Python dependencies...
pip install -r requirements.txt
pip install -r requirements-dev.txt

REM Copy environment file if not present
if not exist ..\.env (
    copy ..\.env.example ..\.env
    echo Created .env file - please review and modify as needed
)

REM Install pre-commit hooks
pre-commit install

echo Setup complete!
echo Next steps:
echo 1. Review and update .env file
echo 2. Run "scripts\install_models.bat" to download models
echo 3. Run "scripts\run_system.bat" to start the system

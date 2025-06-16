@echo off
REM ==========================================
REM Cline Local LLM System - Install Ollama Models (Windows)
REM ==========================================

echo 📥 Installing Ollama models...

REM Check if Ollama is installed
where ollama >nul 2>&1
if errorlevel 1 (
    echo ❌ Ollama not found. Please install Ollama first.
    echo Visit: https://ollama.com/download
    exit /b 1
)

REM Start Ollama service in background
echo 🔄 Starting Ollama service...
start "" /B ollama serve
timeout /t 5 >nul

REM List of required models
set MODELS=qwen2.5:7b qwen2.5-coder:32b llama4:16x17b qwen3:30b-a3b mxbai-embed-large

for %%M in (%MODELS%) do (
    ollama list | findstr /C:"%%M" >nul
    if errorlevel 1 (
        echo 📥 Model %%M not found. Installing...
        ollama pull %%M
    ) else (
        echo 🔄 Model %%M already installed. Checking for updates...
        ollama pull %%M
    )
)

echo ✅ Models installed and up to date!

REM Stop the background Ollama process
taskkill /IM ollama.exe /F >nul 2>&1

exit /b

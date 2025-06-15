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

REM Install models
echo 📥 Downloading small model (qwen2.5:7b)...
ollama pull qwen2.5:7b

echo 📥 Downloading large model (qwen2.5-coder:32b)...
ollama pull qwen2.5-coder:32b

echo 📥 Downloading llama4:16x17b...
ollama pull llama4:16x17b

echo 📥 Downloading qwen3:30b-a3b...
ollama pull qwen3:30b-a3b

echo ✅ Models installed successfully!

REM Stop the background Ollama process
taskkill /IM ollama.exe /F >nul 2>&1

exit /b

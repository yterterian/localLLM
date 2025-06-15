@echo off
REM ==========================================
REM Cline Local LLM System - Windows Launcher
REM ==========================================

REM Load environment variables from .env if present
IF EXIST ..\.env (
    for /f "usebackq delims=" %%A in ("..\..env") do set "%%A"
)

echo 🚀 Starting Local LLM System...

REM Start Ollama in background
echo 🔄 Starting Ollama service...
start "" /B ollama serve
timeout /t 3 >nul

REM Start the router
echo 🤖 Starting Cline Memory Bank Router...
start "" /B python src\cline\memory_aware_router.py

REM Start the context expansion server
echo 🔍 Starting Context Expansion Server...
start "" /B python src\integration\cline_server.py

echo ✅ System started successfully!
echo 📊 Router API: http://localhost:8000
echo 🔍 Context API: http://localhost:8001
echo 📖 API docs: http://localhost:8000/docs

echo.
echo Press any key to exit and terminate all background processes...
pause >nul

REM Cleanup: kill all started processes (Ollama, router, context server)
REM WARNING: This will kill all python and ollama processes!
taskkill /IM python.exe /F >nul 2>&1
taskkill /IM ollama.exe /F >nul 2>&1

echo 🛑 Shutting down system...
exit /b

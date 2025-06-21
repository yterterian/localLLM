@echo off
REM ==========================================
REM Cline Local LLM System - Windows Launcher
REM ==========================================

REM Ensure working directory is project root (parent of scripts)
cd /d "%~dp0.."

REM Load environment variables from .env if present
IF EXIST ..\.env (
    for /f "usebackq delims=" %%A in ("..\..env") do set "%%A"
)

echo 🚀 Starting Local LLM System...

REM Start Ollama in a new Command Prompt window
echo 🔄 Starting Ollama service in a new window...
start "Ollama" cmd /k "ollama serve"

REM Start the router in a new Command Prompt window
echo 🤖 Starting Cline Memory Bank Router in a new window...
start "Cline Router" cmd /k "call .venv\Scripts\activate && python src\llm\memory_aware_router.py"

REM Start the context expansion server in a new Command Prompt window
echo 🔍 Starting Context Expansion Server in a new window...
start "Context Server" cmd /k "call .venv\Scripts\activate && python src\integration\cline_server.py"

REM Start Open WebUI in a new Command Prompt window
echo 🌐 Starting Open WebUI in a new window...
start "Open WebUI" cmd /k "call .webui-venv\Scripts\activate && set OPENAI_API_BASE_URL=http://localhost:8000/v1 && set OPENAI_API_KEY=dummy-key && open-webui serve --host 0.0.0.0 --port 3000"

echo ✅ All services started in separate windows!
echo 📊 Router API: http://localhost:8000
echo 🔍 Context API: http://localhost:8001
echo 🌐 Open WebUI: http://localhost:3000
echo 📖 API docs: http://localhost:8000/docs

echo.
echo Press any key to exit this launcher window. Services will keep running in their own windows.
pause >nul

echo 🛑 Launcher window closed. To stop services, close their respective Command Prompt windows.
exit /b

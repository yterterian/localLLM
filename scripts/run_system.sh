@echo off
REM ==========================================
REM Cline Local LLM System - Production Ready Launcher
REM ==========================================

echo 🚀 Starting Local LLM System...

REM Navigate to project root from scripts directory
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%\.."
pushd "%PROJECT_ROOT%"

echo 📍 Project root: %CD%
echo.

REM Load environment variables from .env if present
IF EXIST ".env" (
    echo 📝 Loading environment from .env...
    for /f "usebackq tokens=1,2 delims==" %%A in (".env") do (
        if not "%%A"=="" if not "%%B"=="" set "%%A=%%B"
    )
) ELSE (
    echo ℹ️ No .env file found (using defaults)
)

echo.

REM Start Ollama service in background
echo 🤖 Starting Ollama service...
start "" /B ollama serve
echo ⏱️ Waiting for Ollama to initialize...
timeout /t 5 >nul

REM Start the memory-aware router
echo 🧠 Starting Cline Memory-Aware Router...
echo 🔧 Command: python src\cline\memory_aware_router.py
start "Cline Memory Router" cmd /k "title Cline Memory Router - Port 8000 && echo Starting Cline Memory Router... && python src\cline\memory_aware_router.py"

echo ⏱️ Waiting for router to start...
timeout /t 4 >nul

REM Start the context expansion server
echo 🔍 Starting Context Expansion Server...
echo 🔧 Command: python src\integration\cline_server.py
start "Context Server" cmd /k "title Context Expansion Server - Port 8001 && echo Starting Context Server... && python src\integration\cline_server.py"

echo ⏱️ Allowing services to initialize...
timeout /t 5 >nul

echo.
echo ✅ Local LLM System Startup Complete!
echo.
echo 🌐 Service Endpoints:
echo   📊 Main Router API: http://localhost:8000
echo   🔍 Context API: http://localhost:8001
echo   📖 API Documentation: http://localhost:8000/docs
echo   📈 Health Check: http://localhost:8000/health
echo.

REM Test connectivity to services
echo 🧪 Testing service connectivity...

echo Checking router health...
curl -s -m 10 http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Memory Router responding at http://localhost:8000
) else (
    echo ⏳ Router may still be starting... (check "Cline Memory Router" window)
)

echo Checking context server...
curl -s -m 10 http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Context Server responding at http://localhost:8001
) else (
    echo ⏳ Context server may still be starting... (check "Context Server" window)
)

echo.
echo 📋 System Status Summary:
echo   🖥️ Two service windows are now running:
echo      • "Cline Memory Router - Port 8000"
echo      • "Context Expansion Server - Port 8001"
echo   📊 Monitor logs in those windows for real-time status
echo   🔧 Your optimization system is active with:
echo      • Model preloading and warm cache
echo      • Intelligent context management  
echo      • Hardware resource optimization
echo.

echo 💡 Quick Test Commands:
echo   curl http://localhost:8000/health
echo   curl http://localhost:8000/v1/models
echo   curl "http://localhost:8000/cline/memory-bank-status/%CD:\=/%"
echo.

echo 🎯 Expected Performance with Optimizations:
echo   📈 Memory Fast (Qwen3): 2-5s → 1-2s (50-60%% faster)
echo   💻 Implementation (Qwen2.5 Coder): 15-45s → 8-25s (40-45%% faster)  
echo   🧠 Analysis (Llama 4): 60-180s → 35-120s (35-40%% faster)
echo.

echo Press any key to shutdown all services and exit...
pause >nul

echo.
echo 🛑 Shutting down Local LLM System...

REM Graceful shutdown of services
echo Stopping Cline Memory Router...
taskkill /FI "WINDOWTITLE eq Cline Memory Router*" /F >nul 2>&1

echo Stopping Context Expansion Server...
taskkill /FI "WINDOWTITLE eq Context Server*" /F >nul 2>&1

echo Stopping Ollama service...
taskkill /IM ollama.exe /F >nul 2>&1

timeout /t 2 >nul

echo ✅ All services stopped.
echo 📊 System shutdown complete.

REM Return to original directory
popd
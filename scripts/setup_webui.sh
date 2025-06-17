@echo off
REM File: scripts/setup_openwebui_native.bat
REM Native Open WebUI setup without Docker (Windows)

setlocal enabledelayedexpansion

echo 🚀 Setting up Open WebUI (Native Python Installation)

REM Check if we're in a virtual environment
if defined VIRTUAL_ENV (
    echo ✅ Using virtual environment: %VIRTUAL_ENV%
) else (
    echo ⚠️  Warning: Not in a virtual environment
    echo    Consider activating your conda/venv environment first
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" (
        exit /b 1
    )
)

REM Install Open WebUI
echo 📦 Installing Open WebUI...
pip install open-webui
if errorlevel 1 (
    echo ❌ Failed to install Open WebUI
    pause
    exit /b 1
)

REM Create configuration directory
echo 📁 Creating directories...
if not exist "config\openwebui" mkdir config\openwebui
if not exist "data\openwebui" mkdir data\openwebui

REM Generate secret key using Python
echo 📝 Generating secret key...
for /f %%i in ('python -c "import secrets; print(secrets.token_hex(32))"') do set SECRET_KEY=%%i

REM Create Open WebUI environment file
echo 📝 Creating Open WebUI configuration...
(
echo # Open WebUI Configuration
echo WEBUI_SECRET_KEY=%SECRET_KEY%
echo WEBUI_URL=http://localhost:3000
echo DATA_DIR=./data/openwebui
echo.
echo # Connect to your local router
echo OPENAI_API_BASE_URL=http://localhost:8000/v1
echo OPENAI_API_KEY=dummy-key
echo.
echo # Optional: Default models ^(your router will handle routing^)
echo DEFAULT_MODELS=qwen2.5:7b,qwen2.5-coder:32b,llama4:16x17b
echo.
echo # Enable features
echo ENABLE_RAG_INGESTION=true
echo ENABLE_RAG_WEB_SEARCH=true
echo ENABLE_IMAGE_GENERATION=false
) > config\openwebui\.env

REM Add CORS support to your router
echo 🔧 Adding CORS support to your router...

set ROUTER_FILE=src\cline\memory_aware_router.py
if exist "%ROUTER_FILE%" (
    REM Check if CORS is already added
    findstr /c:"CORSMiddleware" "%ROUTER_FILE%" >nul
    if errorlevel 1 (
        echo   Adding CORS middleware...
        
        REM Create backup
        copy "%ROUTER_FILE%" "%ROUTER_FILE%.backup" >nul
        
        REM Create temporary file with CORS additions
        set TEMP_FILE=%ROUTER_FILE%.temp
        
        REM Process the file line by line
        (
            for /f "delims=" %%a in ('%ROUTER_FILE%') do (
                echo %%a
                REM Add CORS import after FastAPI import
                echo %%a | findstr /c:"from fastapi import FastAPI" >nul
                if not errorlevel 1 (
                    echo from fastapi.middleware.cors import CORSMiddleware
                )
                REM Add CORS middleware after app creation
                echo %%a | findstr /r /c:"^app = FastAPI(" >nul
                if not errorlevel 1 (
                    echo.
                    echo # Add CORS middleware for Open WebUI
                    echo app.add_middleware^(
                    echo     CORSMiddleware,
                    echo     allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
                    echo     allow_credentials=True,
                    echo     allow_methods=["*"],
                    echo     allow_headers=["*"],
                    echo ^)
                )
            )
        ) > "!TEMP_FILE!"
        
        REM Replace original file
        move "!TEMP_FILE!" "%ROUTER_FILE%" >nul
        
        echo   ✅ CORS support added to router
    ) else (
        echo   ✅ CORS support already present
    )
) else (
    echo   ⚠️  Router file not found at %ROUTER_FILE%
)

REM Create startup script
echo 📜 Creating startup script...
(
echo @echo off
echo REM File: scripts/start_with_openwebui.bat
echo REM Start the complete system with Open WebUI
echo.
echo setlocal enabledelayedexpansion
echo.
echo echo 🚀 Starting Local LLM System with Open WebUI...
echo.
echo REM Function to check if port is in use
echo set "CHECK_PORT_CMD=netstat -an | findstr"
echo.
echo REM Check ports
echo echo 🔍 Checking ports...
echo.
echo REM Check Ollama ^(port 11434^)
echo !CHECK_PORT_CMD! ":11434 " ^| findstr "LISTENING" ^>nul
echo if errorlevel 1 ^(
echo     echo   Starting Ollama...
echo     start "Ollama Server" cmd /k "title Ollama Server - Port 11434 && ollama serve"
echo     timeout /t 5 ^>nul
echo ^) else ^(
echo     echo   Ollama already running on port 11434
echo ^)
echo.
echo REM Check Router ^(port 8000^)
echo !CHECK_PORT_CMD! ":8000 " ^| findstr "LISTENING" ^>nul
echo if errorlevel 1 ^(
echo     echo   Starting Cline Memory Router...
echo     start "Cline Router" cmd /k "title Cline Memory Router - Port 8000 && cd /d %%~dp0.. && python src\cline\memory_aware_router.py"
echo     echo   ⏳ Waiting for router to start...
echo     timeout /t 8 ^>nul
echo ^) else ^(
echo     echo   Router already running on port 8000
echo ^)
echo.
echo REM Check Open WebUI ^(port 3000^)
echo !CHECK_PORT_CMD! ":3000 " ^| findstr "LISTENING" ^>nul
echo if not errorlevel 1 ^(
echo     echo ❌ Port 3000 is already in use. Please stop the service using it.
echo     pause
echo     exit /b 1
echo ^)
echo.
echo echo 🌐 Starting Open WebUI...
echo cd /d %%~dp0..\config\openwebui
echo start "Open WebUI" cmd /k "title Open WebUI - Port 3000 && open-webui serve --port 3000 --host 0.0.0.0"
echo cd /d %%~dp0..
echo.
echo echo ⏳ Waiting for Open WebUI to start...
echo timeout /t 10 ^>nul
echo.
echo echo.
echo echo 🎉 System started successfully!
echo echo.
echo echo 🔗 Access Points:
echo echo    • Open WebUI:     http://localhost:3000
echo echo    • Router API:     http://localhost:8000
echo echo    • API Docs:       http://localhost:8000/docs
echo echo    • Ollama API:     http://localhost:11434
echo echo.
echo echo 💡 First time setup:
echo echo    1. Go to http://localhost:3000
echo echo    2. Create admin account
echo echo    3. Models should be auto-detected from your router
echo echo.
echo echo 🛑 To stop all services, close the service windows or run:
echo echo    scripts\stop_system.bat
echo echo.
echo echo 📊 Service Windows:
echo echo    • Check "Ollama Server - Port 11434" window for Ollama logs
echo echo    • Check "Cline Memory Router - Port 8000" window for Router logs  
echo echo    • Check "Open WebUI - Port 3000" window for WebUI logs
echo echo.
echo pause
) > scripts\start_with_openwebui.bat

REM Create stop script
echo 📜 Creating stop script...
(
echo @echo off
echo REM File: scripts/stop_system.bat
echo REM Stop all system services
echo.
echo echo 🛑 Stopping Local LLM System...
echo.
echo echo   Stopping Open WebUI ^(port 3000^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":3000 "'^) do ^(
echo     taskkill /PID %%%%a /F ^>nul 2^>^&1
echo ^)
echo.
echo echo   Stopping Router ^(port 8000^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":8000 "'^) do ^(
echo     taskkill /PID %%%%a /F ^>nul 2^>^&1
echo ^)
echo.
echo echo   Stopping Ollama ^(port 11434^)...
echo for /f "tokens=5" %%%%a in ^('netstat -aon ^| findstr ":11434 "'^) do ^(
echo     taskkill /PID %%%%a /F ^>nul 2^>^&1
echo ^)
echo.
echo REM Alternative: kill by process name
echo taskkill /IM "python.exe" /FI "WINDOWTITLE eq *memory_aware_router*" /F ^>nul 2^>^&1
echo taskkill /IM "python.exe" /FI "WINDOWTITLE eq *open-webui*" /F ^>nul 2^>^&1
echo taskkill /IM "ollama.exe" /F ^>nul 2^>^&1
echo.
echo REM Close service windows
echo taskkill /FI "WINDOWTITLE eq Ollama Server*" /F ^>nul 2^>^&1
echo taskkill /FI "WINDOWTITLE eq Cline Memory Router*" /F ^>nul 2^>^&1
echo taskkill /FI "WINDOWTITLE eq Open WebUI*" /F ^>nul 2^>^&1
echo.
echo echo ✅ All services stopped
echo pause
) > scripts\stop_system.bat

echo.
echo 🎉 Native Open WebUI setup complete!
echo.
echo 📋 Next steps:
echo 1. Run: scripts\start_with_openwebui.bat
echo 2. Open http://localhost:3000
echo 3. Create admin account
echo 4. Start chatting!
echo.
echo 📁 Configuration:
echo    • Open WebUI config: config\openwebui\.env
echo    • Data directory: data\openwebui\
echo    • Logs: Check service windows
echo.
echo 🔧 Your router will automatically handle:
echo    • Intelligent model selection
echo    • Memory bank integration  
echo    • Context optimization
echo    • Performance improvements
echo.
pause
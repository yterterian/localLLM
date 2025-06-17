@echo off
REM ==========================================
REM Setup and launch Open WebUI (Windows)
REM ==========================================

REM Set environment variables for Open WebUI
set OPENAI_API_BASE_URL=http://localhost:8000/v1
set OPENAI_API_KEY=dummy-key

REM Start Open WebUI on port 3000
open-webui serve --port 3000

exit /b

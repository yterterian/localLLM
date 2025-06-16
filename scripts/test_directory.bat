@echo off
echo 🔍 Directory Test Script
echo =====================

echo Script location: %~dp0
echo Current directory: %CD%

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR:~0,-1%\.."

echo Script dir: %SCRIPT_DIR%
echo Project root: %PROJECT_ROOT%

pushd "%PROJECT_ROOT%"
echo After pushd: %CD%

echo.
echo 📁 Checking file existence:
IF EXIST "src\cline\memory_aware_router.py" (
    echo ✅ src\cline\memory_aware_router.py found
) ELSE (
    echo ❌ src\cline\memory_aware_router.py NOT found
)

IF EXIST "src\integration\cline_server.py" (
    echo ✅ src\integration\cline_server.py found  
) ELSE (
    echo ❌ src\integration\cline_server.py NOT found
)

echo.
echo 📂 Contents of src\ directory:
if exist "src\" (
    dir src\ /B
) else (
    echo src\ directory doesn't exist
)

echo.
echo 📂 Contents of src\cline\ directory:
if exist "src\cline\" (
    dir src\cline\ /B
) else (
    echo src\cline\ directory doesn't exist
)

popd
pause
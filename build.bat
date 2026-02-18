@echo off
setlocal

echo ============================================================
echo   Suno Prompt Generator - Build Script
echo ============================================================
echo.

:: Set project directory to wherever this script lives
cd /d "%~dp0"

:: ─────────────────────────────────────────────
:: STEP 1: Check prerequisites
:: ─────────────────────────────────────────────
echo [1/4] Checking prerequisites...

:: Check Python venv
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Python venv not found. Run: python -m venv venv
    echo        Then: venv\Scripts\pip install -r requirements.txt pyinstaller
    pause
    exit /b 1
)

:: Check PyInstaller
venv\Scripts\python.exe -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo ERROR: PyInstaller not installed. Run: venv\Scripts\pip install pyinstaller
    pause
    exit /b 1
)

echo    Python venv: OK
echo    PyInstaller: OK

:: Check for Inno Setup (optional — only needed for installer)
set "ISCC="
where iscc.exe >nul 2>&1 && set "ISCC=iscc.exe"
if "%ISCC%"=="" (
    if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
)
if "%ISCC%"=="" (
    if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC=C:\Program Files\Inno Setup 6\ISCC.exe"
)
if "%ISCC%"=="" (
    echo    Inno Setup: NOT FOUND - will skip installer creation
    echo    Download from: https://jrsoftware.org/isdl.php
) else (
    echo    Inno Setup: OK
)
echo.

:: ─────────────────────────────────────────────
:: STEP 2: Clean previous build
:: ─────────────────────────────────────────────
echo [2/4] Cleaning previous build artifacts...
if exist "dist" rmdir /s /q "dist"
if exist "build" rmdir /s /q "build"
if exist "SunoPromptGenerator.spec" del /q "SunoPromptGenerator.spec"
echo    Done.
echo.

:: ─────────────────────────────────────────────
:: STEP 3: Run PyInstaller
:: ─────────────────────────────────────────────
echo [3/4] Running PyInstaller...
echo    This may take a few minutes...
echo.

venv\Scripts\pyinstaller.exe ^
    --noconfirm ^
    --windowed ^
    --name "SunoPromptGenerator" ^
    --additional-hooks-dir hooks ^
    --runtime-hook runtime_hook.py ^
    --collect-data gradio ^
    --collect-data gradio_client ^
    --collect-data safehttpx ^
    --add-data "knowledge_base.py;." ^
    desktop_app.py

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed!
    pause
    exit /b 1
)

echo.
echo    PyInstaller build complete: dist\SunoPromptGenerator\
echo.

:: ─────────────────────────────────────────────
:: STEP 4: Run Inno Setup (if available)
:: ─────────────────────────────────────────────
if "%ISCC%"=="" (
    echo [4/4] Skipping installer creation (Inno Setup not found).
    echo    You can still run the app directly from: dist\SunoPromptGenerator\SunoPromptGenerator.exe
    echo    To create an installer, install Inno Setup 6 and re-run this script.
) else (
    echo [4/4] Creating Windows installer with Inno Setup...
    "%ISCC%" installer.iss
    if errorlevel 1 (
        echo.
        echo ERROR: Inno Setup build failed!
        pause
        exit /b 1
    )
    echo.
    echo    Installer created: Output\SunoPromptGenerator_Setup.exe
)

echo.
echo ============================================================
echo   BUILD COMPLETE!
echo ============================================================
echo.
if not "%ISCC%"=="" (
    echo   Installer: Output\SunoPromptGenerator_Setup.exe
)
echo   Portable:  dist\SunoPromptGenerator\SunoPromptGenerator.exe
echo.
pause

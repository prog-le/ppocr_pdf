@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

:: ============================================================================
:: setup_env.bat — One-click PaddleOCR + PP-StructureV3 environment setup
:: ============================================================================
:: Detects:
::   - Python version (3.8–3.12 supported)
::   - NVIDIA GPU + CUDA version → maps to PaddlePaddle tag (cu118, cu126, etc.)
::   - Falls back to CPU-only if no GPU / no compatible CUDA
:: Installs:
::   1. Matching PaddlePaddle (GPU or CPU)
::   2. PaddleOCR (pip)
::   3. requirements.txt
::   4. PP-StructureV3 extras (optional prompt)
:: ============================================================================

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set REQS=%PROJECT_DIR%\requirements.txt

title PaddleOCR Environment Setup — Auto Detect
color 1f
echo.
echo ============================================================
echo   PaddleOCR Environment Setup
echo   Auto-detecting Python, GPU ^& CUDA ...
echo ============================================================
echo.

:: ---------------------------------------------------------------
:: 1. Check Python
:: ---------------------------------------------------------------
:check_python
echo [1/7] Checking Python installation ...
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo         Please install Python 3.8–3.12 from https://python.org
    pause
    exit /b 1
)

for /f "tokens=2 delims=." %%v in ('python --version 2^>^&1 ^| findstr /i "Python"') do (
    set PYTHON_MAJOR=%%v
)
for /f "tokens=3 delims=. " %%v in ('python --version 2^>^&1 ^| findstr /i "Python"') do (
    set PYTHON_MINOR=%%v
)
echo   ^> Python !PYTHON_MAJOR!.!PYTHON_MINOR! detected

:: Validate version range (3.8–3.12)
if !PYTHON_MAJOR! lss 3 (
    echo [ERROR] Python ^< 3 is not supported.
    pause
    exit /b 1
)
if !PYTHON_MAJOR! equ 3 (
    if !PYTHON_MINOR! lss 8 (
        echo [ERROR] Python 3.!PYTHON_MINOR! is too old. 3.8–3.12 required.
        pause
        exit /b 1
    )
    if !PYTHON_MINOR! gtr 12 (
        echo [ERROR] Python 3.!PYTHON_MINOR! is not yet supported. 3.8–3.12 required.
        pause
        exit /b 1
    )
)
echo.
goto check_gpu

:: ---------------------------------------------------------------
:: 2. Check GPU / CUDA
:: ---------------------------------------------------------------
:check_gpu
echo [2/7] Checking for NVIDIA GPU ...
set HAS_GPU=0
set CUDA_VERSION=
set CUDNN_VERSION=

nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader') do (
        set GPU_NAME=%%g
    )
    echo   ^> NVIDIA GPU found: !GPU_NAME!
    set HAS_GPU=1

    :: CUDA version from nvidia-smi query output
    for /f "tokens=4 delims= " %%c in ('nvidia-smi -q 2^>^&1 ^| findstr /i /c:"CUDA Version"') do (
        set CUDA_VERSION=%%c
    )
    if defined CUDA_VERSION (
        echo   ^> CUDA !CUDA_VERSION! detected
    ) else (
        echo   [WARN] Could not detect CUDA version from nvidia-smi.
    )
) else (
    echo   ^> No NVIDIA GPU detected ^(--^> CPU mode^)
)
echo.
goto map_paddle_tag

:: ---------------------------------------------------------------
:: 3. Map CUDA → PaddlePaddle tag
:: ---------------------------------------------------------------
:map_paddle_tag
echo [3/7] Selecting PaddlePaddle version ...

if "%HAS_GPU%"=="0" goto install_cpu_paddle

:: Parse CUDA major.minor
for /f "tokens=1,2 delims=." %%a in ("%CUDA_VERSION%") do (
    set CUDA_MAJOR=%%a
    set CUDA_MINOR=%%b
)

:: Map to PaddlePaddle compute tag
:: cu126 → PaddlePaddle 3.0+ (CUDA 12.6)
:: cu124 → PaddlePaddle 3.0+ (CUDA 12.4)
:: cu122 → PaddlePaddle 3.0+ (CUDA 12.2)
:: cu121 → PaddlePaddle 3.0+ (CUDA 12.1)
:: cu120 → PaddlePaddle 3.0+ (CUDA 12.0)
:: cu118 → PaddlePaddle 2.6+ (CUDA 11.8)
:: cu117 → PaddlePaddle 2.5+ (CUDA 11.7)
:: cu116 → PaddlePaddle 2.5+ (CUDA 11.6)
:: cu115 → PaddlePaddle 2.5+ (CUDA 11.5)
:: cu114 → PaddlePaddle 2.4+ (CUDA 11.4)
:: cu113 → PaddlePaddle 2.3+ (CUDA 11.3)
:: cu112 → PaddlePaddle 2.3+ (CUDA 11.2)
:: cu111 → PaddlePaddle 2.2+ (CUDA 11.1)
:: cu102 → PaddlePaddle 2.2+ (CUDA 10.2)
:: cu101 → PaddlePaddle 2.1+ (CUDA 10.1)

if !CUDA_MAJOR! gtr 12 (
    set PADDLE_TAG=cu126
    set PADDLE_DEVICE=gpu
    goto install_gpu_paddle
)
if !CUDA_MAJOR! equ 12 (
    if !CUDA_MINOR! geq 6 (
        set PADDLE_TAG=cu126
    ) else if !CUDA_MINOR! geq 4 (
        set PADDLE_TAG=cu124
    ) else if !CUDA_MINOR! geq 2 (
        set PADDLE_TAG=cu122
    ) else if !CUDA_MINOR! geq 1 (
        set PADDLE_TAG=cu121
    ) else (
        set PADDLE_TAG=cu120
    )
    set PADDLE_DEVICE=gpu
    goto install_gpu_paddle
)
if !CUDA_MAJOR! equ 11 (
    if !CUDA_MINOR! geq 8 (
        set PADDLE_TAG=cu118
    ) else if !CUDA_MINOR! geq 7 (
        set PADDLE_TAG=cu117
    ) else if !CUDA_MINOR! geq 6 (
        set PADDLE_TAG=cu116
    ) else if !CUDA_MINOR! geq 5 (
        set PADDLE_TAG=cu115
    ) else if !CUDA_MINOR! geq 4 (
        set PADDLE_TAG=cu114
    ) else if !CUDA_MINOR! geq 3 (
        set PADDLE_TAG=cu113
    ) else if !CUDA_MINOR! geq 2 (
        set PADDLE_TAG=cu112
    ) else (
        set PADDLE_TAG=cu111
    )
    set PADDLE_DEVICE=gpu
    goto install_gpu_paddle
)
if !CUDA_MAJOR! equ 10 (
    if !CUDA_MINOR! geq 2 (
        set PADDLE_TAG=cu102
    ) else (
        set PADDLE_TAG=cu101
    )
    set PADDLE_DEVICE=gpu
    goto install_gpu_paddle
)

:: Fallback: unknown CUDA → CPU
echo   [WARN] CUDA !CUDA_VERSION! not in known PaddlePaddle tag map.
echo          Falling back to CPU install.
goto install_cpu_paddle

:: ---------------------------------------------------------------
:: 4a. Install PaddlePaddle (GPU)
:: ---------------------------------------------------------------
:install_gpu_paddle
echo   ^> PaddlePaddle tag: !PADDLE_TAG! (GPU)
echo.
echo [4/7] Installing PaddlePaddle (GPU) ...
python -m pip install paddlepaddle-gpu==!PADDLE_TAG! -f https://www.paddlepaddle.org.cn/whl/!PADDLE_TAG!/simple
if %ERRORLEVEL% neq 0 (
    echo [WARN] paddlepaddle-gpu install failed. Trying CPU fallback ...
    goto install_cpu_paddle
)
echo   ^> PaddlePaddle GPU installed successfully
echo.
goto install_paddleocr

:: ---------------------------------------------------------------
:: 4b. Install PaddlePaddle (CPU)
:: ---------------------------------------------------------------
:install_cpu_paddle
echo   ^> PaddlePaddle (CPU)
echo.
echo [4/7] Installing PaddlePaddle (CPU) ...
python -m pip install paddlepaddle
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PaddlePaddle CPU.
    pause
    exit /b 1
)
echo   ^> PaddlePaddle CPU installed successfully
echo.
goto install_paddleocr

:: ---------------------------------------------------------------
:: 5. Install PaddleOCR
:: ---------------------------------------------------------------
:install_paddleocr
echo [5/7] Installing PaddleOCR ...
python -m pip install paddleocr
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install PaddleOCR.
    pause
    exit /b 1
)
echo   ^> PaddleOCR installed successfully
echo.
goto install_requirements

:: ---------------------------------------------------------------
:: 6. Install project dependencies
:: ---------------------------------------------------------------
:install_requirements
echo [6/7] Installing project requirements ...
if exist "%REQS%" (
    python -m pip install -r "%REQS%"
    if %ERRORLEVEL% neq 0 (
        echo [WARN] Some requirements may have failed. Check the output above.
    ) else (
        echo   ^> All requirements installed successfully
    )
) else (
    echo   [WARN] requirements.txt not found at %REQS%
    echo          Skipping ...
)
echo.
goto prompt_ppstructure

:: ---------------------------------------------------------------
:: 7. PP-StructureV3 extras (optional)
:: ---------------------------------------------------------------
:prompt_ppstructure
echo [7/7] PP-StructureV3 extra dependencies (optional)
echo.
echo   PP-StructureV3 requires additional packages for layout analysis,
echo   table recognition, OCR visualization, and PDF parsing.
echo.
set /p INSTALL_PPSTRUCT="   Install PP-StructureV3 extras? (Y/n): "
if /i "!INSTALL_PPSTRUCT!"=="" set INSTALL_PPSTRUCT=Y
if /i "!INSTALL_PPSTRUCT!"=="Y" (
    echo.
    echo   Installing PP-StructureV3 dependencies ...
    echo.

    :: Core layout detection
    python -m pip install "paddleocr[layout]"

    :: Table recognition
    python -m pip install "paddleocr[table]"

    :: Visualization
    python -m pip install "paddleocr[vis]"

    :: PDF parsing
    python -m pip install "paddleocr[paddle]"
    python -m pip install PyMuPDF

    :: Optional: structured analysis
    python -m pip install lxml openpyxl

    if %ERRORLEVEL% neq 0 (
        echo [WARN] Some PP-StructureV3 extras may have failed.
    ) else (
        echo   ^> PP-StructureV3 extras installed
    )
) else (
    echo   ^> Skipped PP-StructureV3 extras
)
echo.

:: ---------------------------------------------------------------
:: Done
:: ---------------------------------------------------------------
echo ============================================================
echo   Setup Complete!
echo.
echo   Python : !PYTHON_MAJOR!.!PYTHON_MINOR!
echo   GPU    : !GPU_NAME!
if defined CUDA_VERSION echo   CUDA   : !CUDA_VERSION!
echo   Paddle : !PADDLE_DEVICE!
echo.
echo   Run your application with:
echo     python app.py
echo ============================================================
echo.
pause

@echo off
REM Launcher for CUDA-enabled llama-server
REM Sets CUDA PATH and starts server

setlocal

REM Add CUDA to PATH
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin;%PATH%"

REM Verify CUDA is accessible
where nvcc >nul 2>&1
if errorlevel 1 (
    echo ERROR: CUDA not found in PATH!
    echo Please ensure CUDA 12.5 is installed correctly.
    pause
    exit /b 1
)

echo CUDA found: 
nvcc --version | findstr "release"

echo.
echo Starting llama-server with CUDA...
echo.

cd /d "%~dp0"

REM Start server with CUDA (FULL GPU OFFLOAD - 99 layers)
REM Context: 32768 (32K - expanded for 60+ iteration projects)
REM GPU Layers: 99 (ALL layers on GPU for max speed)
REM Batch: 256 (optimal for throughput)
REM Flash Attention: on (faster attention)
REM Threads: 8 (stable performance)
REM Model: Qwen3.5-9B (better reasoning than 4B)
llama-src\build\bin\llama-server.exe ^
    -m Qwen3.5-9B-Q4_0.gguf ^
    --port 1234 ^
    -c 32768 ^
    -ngl 99 ^
    -b 256 ^
    -fa on ^
    -t 8

if errorlevel 1 (
    echo.
    echo Server failed to start!
    echo Check for CUDA DLL errors above.
    pause
    exit /b 1
)

endlocal

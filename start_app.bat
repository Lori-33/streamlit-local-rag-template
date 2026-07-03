@echo off
title Streamlit Local RAG Template

set "PROJECT_DIR=%~dp0"
set "PYTHON_EXE=%PROJECT_DIR%.venv\Scripts\python.exe"
set "LMS_EXE=%USERPROFILE%\.lmstudio\bin\lms.exe"

if not exist "%PYTHON_EXE%" (
    echo [WARN] .venv Python not found. Falling back to system python.
    set "PYTHON_EXE=python"
)

if exist "%LMS_EXE%" (
    "%LMS_EXE%" server start
    "%LMS_EXE%" load qwen3-8b --identifier qwen3-8b -y
    "%LMS_EXE%" load text-embedding-bge-m3 --identifier text-embedding-bge-m3 -y
)

cd /d "%PROJECT_DIR%"
start "" http://localhost:8501
"%PYTHON_EXE%" -m streamlit run app.py
pause

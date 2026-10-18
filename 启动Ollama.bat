@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 启动 Ollama（本地知识库前置服务）
echo ============================================
echo   启动 Ollama（本地知识库前置服务）
echo ============================================
echo.
tasklist /FI "IMAGENAME eq ollama.exe" 2>nul | find /I "ollama.exe" >nul
if %errorlevel%==0 (
  echo Ollama 已在运行，无需重复启动。
) else (
  start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
  echo 已拉起 Ollama（托盘图标），等 5 秒让它就绪...
  timeout /t 5 /nobreak >nul
)
echo.
echo 现在可以双击「考插资料.bat」或「课业资料.bat」开始查。
pause

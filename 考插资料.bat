@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 考插资料检索（数学 + 英语）
echo ============================================
echo   考插资料检索（数学 + 英语）
echo ============================================
echo.
py -3.12 -X utf8 kb_ask.py --scope kaocha %*
echo.
pause

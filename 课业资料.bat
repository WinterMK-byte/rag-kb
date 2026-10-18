@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 课业资料检索（华理课程 + 实验 + 课件）
echo ============================================
echo   课业资料检索（华理课程 + 实验 + 课件）
echo ============================================
echo.
py -3.12 -X utf8 kb_ask.py --scope course %*
echo.
pause

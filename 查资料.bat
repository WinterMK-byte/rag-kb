@echo off
chcp 65001 >nul
cd /d "%~dp0"
title 本地知识库查询
echo ============================================
echo   本地知识库查询
echo   直接输入问题回车即可；输入 h 看帮助，q 退出
echo ============================================
echo.
py -3.12 -X utf8 kb_ask.py %*
echo.
pause

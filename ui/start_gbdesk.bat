@echo off
chcp 65001 >nul
title 工兵台
echo 正在启动工兵台（首次检索约需 1 分钟加载 BGE 模型）...
start "" "http://127.0.0.1:8765"
"C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe" "%~dp0kb_ui_server.py" --port 8765

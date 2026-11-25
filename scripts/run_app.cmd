@echo off
rem 本地启动界面:已下载过嵌入模型时用离线模式,避免每次联网校验卡住
set HF_HUB_OFFLINE=1
set HF_ENDPOINT=https://hf-mirror.com
python "%~dp0..\app.py"

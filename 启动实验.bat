@echo off
chcp 65001 >nul
title 702solver 实验入口

echo.
echo ========================================
echo   702solver 实验入口 (WSL)
echo ========================================
echo.

where wsl >nul 2>&1
if errorlevel 1 (
  echo 错误：未找到 wsl。请安装 WSL，或在 WSL 终端里运行 启动实验.sh
  pause
  exit /b 1
)

REM 用 wslpath 把当前脚本所在目录转为 WSL 路径（支持 \\wsl$\ 下的项目文件夹）
for /f "usebackq delims=" %%P in (`wsl wslpath -u "%~dp0."`) do set "WSL_DIR=%%P"

wsl -e bash -lc "cd '%WSL_DIR%' && chmod +x ./启动实验.sh 2>/dev/null; exec ./启动实验.sh %*"
set EXITCODE=%ERRORLEVEL%
if not "%EXITCODE%"=="0" pause
exit /b %EXITCODE%

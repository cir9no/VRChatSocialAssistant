@echo off
chcp 65001 > nul
echo ========================================
echo VRChat 社交辅助工具 - STT 快速测试
echo ========================================
echo.

cd /d %~dp0

if exist .venv\Scripts\activate.bat (
    echo 激活虚拟环境...
    call .venv\Scripts\activate.bat
) else (
    echo 警告: 未找到虚拟环境，使用系统 Python
)

echo.
echo 启动 STT 测试程序...
echo.

python tests\integrated_test.py --module stt

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo 测试过程中出现错误！
    pause
) else (
    echo.
    echo 测试完成！
    pause
)

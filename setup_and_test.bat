@echo off
chcp 65001 >nul
echo ========================================
echo VRChat 社交助手 - 音频采集模块测试
echo ========================================
echo.

echo [1/3] 检查 Python 环境...
python --version
if %errorlevel% neq 0 (
    echo 错误: 未找到 Python，请先安装 Python 3.10+
    pause
    exit /b 1
)
echo.

echo [2/3] 安装依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo 错误: 依赖安装失败
    pause
    exit /b 1
)
echo.

echo [3/3] 运行音频采集测试...
python tests\test_audio_capture.py
echo.

echo ========================================
echo 测试完成！
echo ========================================
pause

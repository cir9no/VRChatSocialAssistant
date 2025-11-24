"""
测试 VAD 模型存储位置
"""
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vad.silero_vad import SileroVAD


def test_model_location():
    """测试模型是否保存在项目目录下"""
    print("正在初始化 Silero VAD 模型...")
    
    # 初始化模型
    vad = SileroVAD(sample_rate=16000, device="cpu")
    
    # 检查模型目录
    print(f"\n模型存储目录: {vad.model_dir}")
    print(f"目录是否存在: {vad.model_dir.exists()}")
    
    # 列出模型目录内容
    if vad.model_dir.exists():
        print("\n模型目录内容:")
        for item in vad.model_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  {item.relative_to(vad.model_dir)} ({size_mb:.2f} MB)")
    
    print("\n✓ 模型初始化成功!")


if __name__ == "__main__":
    test_model_location()

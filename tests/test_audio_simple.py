"""
简单的音频采集测试
"""

import sys
from pathlib import Path
import time
import logging

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from audio_capture.device_manager import DeviceManager
from audio_capture.audio_capturer import AudioCapturer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 计数器
callback_count = 0

def main():
    global callback_count
    
    print("=" * 70)
    print("简单音频采集测试")
    print("=" * 70)
    
    # 检测设备
    print("\n正在检测音频设备...")
    device_manager = DeviceManager()
    
    # 使用默认 WASAPI Loopback 设备
    default_loopback = device_manager.get_default_wasapi_loopback()
    
    if not default_loopback:
        print("错误: 未找到默认 WASAPI Loopback 设备")
        return 1
    
    loopback_device = default_loopback['index']
    print(f"使用默认 Loopback 设备: [{loopback_device}] {default_loopback['name']}")
    
    device_manager.close()
    
    # 创建采集器
    print("\n创建音频采集器...")
    capturer = AudioCapturer(
        loopback_device=loopback_device,
        microphone_device=None,
        samplerate=16000,
        channels=1,
        chunk_size=480
    )
    
    # 设置回调
    def audio_callback(audio_data, timestamp):
        global callback_count
        callback_count += 1
        if callback_count % 10 == 0:
            print(f"\n收到音频数据: 样本数={len(audio_data)}, 时间戳={timestamp:.3f}")
    
    capturer.set_loopback_callback(audio_callback)
    
    # 启动采集
    print("\n启动音频采集（5秒）...")
    capturer.start()
    
    # 等待
    start_time = time.time()
    while time.time() - start_time < 5:
        time.sleep(0.5)
        print(f"\r进度: {time.time() - start_time:.1f}s, 回调次数: {callback_count}", end='', flush=True)
    
    print()
    
    # 停止
    capturer.stop()
    
    # 统计
    stats = capturer.get_statistics()
    print(f"\n统计信息:")
    print(f"  Loopback 帧数: {stats['loopback_frames_captured']}")
    print(f"  回调次数: {callback_count}")
    print(f"  溢出次数: {stats['loopback_overflows']}")
    
    if callback_count == 0:
        print("\n✗ 测试失败: 未收到任何音频数据")
        return 1
    else:
        print(f"\n✓ 测试成功: 收到 {callback_count} 次回调")
        return 0

if __name__ == '__main__':
    sys.exit(main())

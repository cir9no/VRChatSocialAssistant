"""
音频采集模块测试脚本

测试 WASAPI Loopback 和麦克风音频采集功能
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio_capture import AudioCapturer, DeviceManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_rms(audio_data: np.ndarray) -> float:
    """计算音频 RMS（均方根）音量"""
    return np.sqrt(np.mean(audio_data ** 2))


def loopback_callback(audio_data: np.ndarray, timestamp: float):
    """回环音频回调函数"""
    rms = calculate_rms(audio_data)
    if rms > 0.01:  # 只显示有声音的片段
        logger.info(f"[WASAPI Loopback] 时间: {timestamp:.3f}, 音量: {rms:.4f}, 样本数: {len(audio_data)}")


def microphone_callback(audio_data: np.ndarray, timestamp: float):
    """麦克风音频回调函数"""
    rms = calculate_rms(audio_data)
    if rms > 0.01:  # 只显示有声音的片段
        logger.info(f"[麦克风] 时间: {timestamp:.3f}, 音量: {rms:.4f}, 样本数: {len(audio_data)}")


def test_device_manager():
    """测试设备管理器"""
    print("\n" + "="*60)
    print("测试设备管理器")
    print("="*60)
    
    manager = DeviceManager()
    manager.print_device_list()
    
    return manager


def test_audio_capture(manager: DeviceManager, duration: int = 10):
    """
    测试音频采集
    
    Args:
        manager: 设备管理器
        duration: 测试持续时间（秒）
    """
    print("\n" + "="*60)
    print("测试音频采集")
    print("="*60)
    
    # 选择设备
    print("\n请选择要使用的设备：")
    
    # 选择 WASAPI Loopback 设备
    loopback_devices = manager.list_loopback_devices()
    if loopback_devices:
        print("\nWASAPI Loopback 设备（系统音频）：")
        for i, device in enumerate(loopback_devices):
            print(f"  {i+1}. [{device['index']}] {device['name']}")
        
        # 尝试获取默认 WASAPI Loopback
        default_loopback = manager.get_default_wasapi_loopback()
        default_choice = 1
        if default_loopback:
            for i, device in enumerate(loopback_devices):
                if device['index'] == default_loopback['index']:
                    default_choice = i + 1
                    break
            print(f"\n推荐使用默认设备: [{default_loopback['index']}] {default_loopback['name']}")
        
        choice = input(f"\n选择 WASAPI Loopback 设备 (1-{len(loopback_devices)}, 默认 {default_choice}, 或按回车跳过): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(loopback_devices):
            loopback_device = loopback_devices[int(choice) - 1]['index']
        elif not choice and default_loopback:
            loopback_device = default_loopback['index']
            print(f"使用默认 WASAPI Loopback: {default_loopback['name']}")
        else:
            loopback_device = None
            print("跳过 WASAPI Loopback 设备")
    else:
        print("\n⚠️  未检测到 WASAPI Loopback 设备")
        print("请确保您的系统支持 WASAPI，且 pyaudiowpatch 库已正确安装")
        loopback_device = None
    
    # 选择麦克风设备
    input_devices = manager.list_input_devices()
    if input_devices:
        print("\n麦克风设备：")
        for i, device in enumerate(input_devices):
            print(f"  {i+1}. [{device['index']}] {device['name']}")
        
        default_input = manager.get_default_input_device()
        default_choice = 1
        if default_input:
            for i, device in enumerate(input_devices):
                if device['index'] == default_input['index']:
                    default_choice = i + 1
                    break
        
        choice = input(f"\n选择麦克风设备 (1-{len(input_devices)}, 默认 {default_choice}, 或按回车跳过): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(input_devices):
            microphone_device = input_devices[int(choice) - 1]['index']
        elif not choice and default_input:
            microphone_device = default_input['index']
            print(f"使用默认麦克风: {default_input['name']}")
        else:
            microphone_device = None
            print("跳过麦克风设备")
    else:
        print("\n未检测到麦克风设备")
        microphone_device = None
    
    if loopback_device is None and microphone_device is None:
        print("\n错误：至少需要选择一个设备")
        return
    
    # 创建采集器
    print(f"\n创建音频采集器...")
    capturer = AudioCapturer(
        loopback_device=loopback_device,
        microphone_device=microphone_device,
        samplerate=16000,
        channels=1,
        chunk_size=480  # 30ms @ 16kHz
    )
    
    # 设置回调函数
    capturer.set_loopback_callback(loopback_callback)
    capturer.set_microphone_callback(microphone_callback)
    
    # 开始采集
    print(f"\n开始采集音频，持续 {duration} 秒...")
    print("提示：")
    if loopback_device is not None:
        print("  - 播放一些音频（如音乐、视频、VRChat 语音）以测试 WASAPI Loopback 采集")
    if microphone_device is not None:
        print("  - 对着麦克风说话以测试麦克风采集")
    print()
    
    try:
        capturer.start()
        
        # 等待指定时间
        time.sleep(duration)
        
        # 停止采集
        capturer.stop()
        
        # 显示统计信息
        stats = capturer.get_statistics()
        print("\n" + "="*60)
        print("采集统计信息：")
        print("="*60)
        print(f"回环音频帧数: {stats['loopback_frames_captured']}")
        print(f"麦克风音频帧数: {stats['microphone_frames_captured']}")
        print(f"回环溢出次数: {stats['loopback_overflows']}")
        print(f"麦克风溢出次数: {stats['microphone_overflows']}")
        print(f"回环队列剩余: {stats['loopback_queue_size']}")
        print(f"麦克风队列剩余: {stats['microphone_queue_size']}")
        
        if stats['loopback_frames_captured'] > 0 or stats['microphone_frames_captured'] > 0:
            print("\n✓ 音频采集成功！")
        else:
            print("\n⚠️  未采集到音频数据，请检查设备设置")
        
    except KeyboardInterrupt:
        print("\n用户中断")
        capturer.stop()
    except Exception as e:
        logger.error(f"采集过程出错: {e}", exc_info=True)
        capturer.stop()


def main():
    """主函数"""
    print("="*60)
    print("VRChat 社交助手 - 音频采集模块测试")
    print("="*60)
    
    # 测试设备管理器
    manager = test_device_manager()
    
    # 等待用户确认
    input("\n按回车键继续测试音频采集...")
    
    # 测试音频采集
    test_audio_capture(manager, duration=10)
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()

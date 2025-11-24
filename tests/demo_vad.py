"""
VAD 模块快速演示脚本

演示 VAD 模块的基本功能
"""

import sys
import numpy as np
import time
import logging
from pathlib import Path

# 添加 src 目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vad import VADDetector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def generate_test_audio():
    """
    生成测试音频序列：静音-语音-静音-语音-静音
    
    Returns:
        list: 音频帧列表，每个元素是 (audio_data, timestamp)
    """
    sample_rate = 16000
    frame_size = 480  # 30ms @ 16kHz
    
    frames = []
    t_offset = 0
    timestamp = time.time()
    
    # 1. 静音 500ms
    logger.info("生成静音片段 (500ms)...")
    for _ in range(17):  # 17 * 30ms ≈ 500ms
        silence = np.zeros(frame_size, dtype=np.float32)
        frames.append((silence, timestamp + t_offset))
        t_offset += 0.03
    
    # 2. 语音 1 秒
    logger.info("生成语音片段 (1000ms)...")
    for i in range(33):  # 33 * 30ms ≈ 1000ms
        t = np.linspace(0, 0.03, frame_size)
        # 混合多个频率的正弦波模拟语音
        speech = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 音符
            0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 音符
            0.1 * np.sin(2 * np.pi * 220 * t)    # A3 音符
        ).astype(np.float32)
        frames.append((speech, timestamp + t_offset))
        t_offset += 0.03
    
    # 3. 静音 500ms
    logger.info("生成静音片段 (500ms)...")
    for _ in range(17):
        silence = np.zeros(frame_size, dtype=np.float32)
        frames.append((silence, timestamp + t_offset))
        t_offset += 0.03
    
    # 4. 语音 800ms
    logger.info("生成语音片段 (800ms)...")
    for i in range(27):  # 27 * 30ms ≈ 800ms
        t = np.linspace(0, 0.03, frame_size)
        # 不同频率
        speech = (
            0.3 * np.sin(2 * np.pi * 523 * t) +  # C5 音符
            0.2 * np.sin(2 * np.pi * 659 * t)    # E5 音符
        ).astype(np.float32)
        frames.append((speech, timestamp + t_offset))
        t_offset += 0.03
    
    # 5. 静音 500ms（触发最后一个片段输出）
    logger.info("生成静音片段 (500ms)...")
    for _ in range(17):
        silence = np.zeros(frame_size, dtype=np.float32)
        frames.append((silence, timestamp + t_offset))
        t_offset += 0.03
    
    logger.info(f"总共生成 {len(frames)} 帧音频，总时长约 {len(frames) * 0.03:.2f} 秒")
    
    return frames


def main():
    """主函数"""
    print("=" * 70)
    print("VAD 模块快速演示")
    print("=" * 70)
    
    # 创建 VAD 检测器
    print("\n1. 初始化 VAD 检测器...")
    detector = VADDetector(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        max_speech_duration_ms=10000,
        min_silence_duration_ms=300,
        debug=False
    )
    
    # 记录检测到的语音片段
    detected_segments = []
    
    def speech_callback(segment, metadata):
        detected_segments.append(metadata)
        print(f"\n  ✓ 检测到语音片段 #{len(detected_segments)}:")
        print(f"    - 时长: {metadata['duration']:.2f} 秒")
        print(f"    - 置信度: {metadata['avg_confidence']:.3f}")
        print(f"    - 样本数: {metadata['num_samples']}")
        print(f"    - 起始时间: {metadata['start_time']:.3f}")
        print(f"    - 结束时间: {metadata['end_time']:.3f}")
    
    detector.set_callback(speech_callback)
    
    # 生成测试音频
    print("\n2. 生成测试音频序列...")
    test_frames = generate_test_audio()
    
    # 处理音频
    print("\n3. 处理音频流...")
    print("   预期结果: 检测到 2 个语音片段")
    print("   - 片段 1: 约 1.0 秒")
    print("   - 片段 2: 约 0.8 秒")
    print("\n   开始处理:")
    
    for i, (audio_data, timestamp) in enumerate(test_frames):
        detector.process_audio(audio_data, timestamp)
        
        # 每 10 帧打印一次进度
        if (i + 1) % 10 == 0:
            print(f"   处理进度: {i + 1}/{len(test_frames)} 帧", end='\r')
    
    print(f"\n   ✓ 处理完成: {len(test_frames)} 帧")
    
    # 打印统计信息
    print("\n4. VAD 检测统计:")
    stats = detector.get_statistics()
    
    print(f"   - 处理帧数: {stats['total_frames_processed']}")
    print(f"   - 检测片段数: {stats['speech_segments_detected']}")
    print(f"   - 总语音时长: {stats['total_speech_duration']:.2f} 秒")
    print(f"   - 平均片段时长: {stats['avg_speech_duration']:.2f} 秒")
    print(f"   - 平均处理时间: {stats['avg_processing_time_ms']:.2f} ms")
    print(f"   - 丢帧数: {stats['frames_dropped']}")
    
    # 性能评估
    print("\n5. 性能评估:")
    if stats['avg_processing_time_ms'] < 30:
        print(f"   ✓ 处理延迟: {stats['avg_processing_time_ms']:.2f} ms (目标: <30ms) - 优秀")
    elif stats['avg_processing_time_ms'] < 50:
        print(f"   ⚠ 处理延迟: {stats['avg_processing_time_ms']:.2f} ms (目标: <30ms) - 可接受")
    else:
        print(f"   ✗ 处理延迟: {stats['avg_processing_time_ms']:.2f} ms (目标: <30ms) - 需要优化")
    
    if stats['frames_dropped'] == 0:
        print(f"   ✓ 丢帧率: 0% - 完美")
    else:
        drop_rate = stats['frames_dropped'] / max(stats['total_frames_processed'], 1) * 100
        print(f"   ⚠ 丢帧率: {drop_rate:.2f}%")
    
    # 验证结果
    print("\n6. 结果验证:")
    if len(detected_segments) >= 1:
        print(f"   ✓ 成功检测到 {len(detected_segments)} 个语音片段")
        
        # 检查片段时长
        for i, seg in enumerate(detected_segments, 1):
            duration = seg['duration']
            confidence = seg['avg_confidence']
            
            if duration >= 0.25:  # 满足最小时长要求
                print(f"   ✓ 片段 {i}: 时长={duration:.2f}s, 置信度={confidence:.3f} - 有效")
            else:
                print(f"   ⚠ 片段 {i}: 时长={duration:.2f}s, 置信度={confidence:.3f} - 过短")
    else:
        print(f"   ⚠ 未检测到语音片段（可能是阈值设置问题）")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        sys.exit(1)

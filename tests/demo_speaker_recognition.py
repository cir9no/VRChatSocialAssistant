"""
说话人识别模块演示脚本

演示如何使用说话人识别模块进行声纹注册和识别
"""

import sys
import os
import logging
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from speaker_recognition import SpeakerRecognizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_audio(duration: float = 2.0, sample_rate: int = 16000, seed: int = None) -> np.ndarray:
    """
    生成合成音频用于演示
    
    Args:
        duration: 音频时长（秒）
        sample_rate: 采样率
        seed: 随机种子（用于生成可复现的音频）
    
    Returns:
        音频数据（numpy array）
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_samples = int(duration * sample_rate)
    
    # 生成简单的正弦波 + 噪声模拟语音
    t = np.linspace(0, duration, num_samples)
    
    # 基频（模拟不同说话人的音高）
    f0 = 150 + (seed or 0) * 20  # 不同种子不同音高
    
    # 生成信号
    signal = np.sin(2 * np.pi * f0 * t)
    signal += 0.5 * np.sin(2 * np.pi * f0 * 2 * t)  # 二次谐波
    signal += 0.3 * np.random.randn(num_samples)  # 噪声
    
    # 归一化
    signal = signal / np.max(np.abs(signal)) * 0.3
    
    return signal.astype(np.float32)


def demo_registration():
    """演示声纹注册流程"""
    print("\n" + "="*70)
    print("演示 1: 声纹注册")
    print("="*70)
    
    # 创建识别器
    recognizer = SpeakerRecognizer()
    
    # 模拟注册好友
    friend_id = "demo_friend_001"
    friend_name = "张三"
    
    print(f"\n正在注册好友: {friend_name} (ID: {friend_id})")
    print("生成 3 段音频样本...")
    
    # 生成3段音频样本（使用相同种子确保一致性）
    audio_segments = []
    for i in range(3):
        audio = generate_synthetic_audio(duration=2.5, seed=42 + i)
        audio_segments.append(audio)
        print(f"  样本 {i+1}: {len(audio)/16000:.2f}秒, "
              f"RMS={np.sqrt(np.mean(audio**2)):.4f}")
    
    # 注册
    success = recognizer.register_speaker(
        friend_id=friend_id,
        name=friend_name,
        audio_segments=audio_segments,
        sample_rate=16000
    )
    
    if success:
        print(f"\n✅ {friend_name} 注册成功！")
    else:
        print(f"\n❌ {friend_name} 注册失败")
        return None
    
    # 显示已注册说话人
    registered = recognizer.get_registered_speakers()
    print(f"\n已注册说话人: {len(registered)} 人")
    for speaker_id in registered:
        info = recognizer.get_speaker_info(speaker_id)
        if info:
            print(f"  - {info.name} (ID: {speaker_id})")
            print(f"    样本数: {info.sample_count}, 平均时长: {info.avg_duration:.2f}s")
    
    return recognizer


def demo_recognition(recognizer: SpeakerRecognizer):
    """演示声纹识别流程"""
    print("\n" + "="*70)
    print("演示 2: 声纹识别")
    print("="*70)
    
    # 生成测试音频（使用相似的种子模拟同一人）
    print("\n测试 1: 识别已注册的说话人")
    test_audio1 = generate_synthetic_audio(duration=2.0, seed=45)  # 相近种子
    
    result = recognizer.recognize(
        audio_segment=test_audio1,
        timestamp=1.0,
        sample_rate=16000
    )
    
    print(f"识别结果:")
    print(f"  - 是否匹配: {'是' if result.matched else '否'}")
    if result.matched:
        info = recognizer.get_speaker_info(result.speaker_id)
        print(f"  - 说话人: {info.name if info else result.speaker_id}")
        print(f"  - 置信度: {result.confidence:.3f}")
    print(f"  - 处理时间: {result.processing_time:.2f}ms")
    
    if result.similarity_scores:
        print(f"  - 相似度分数:")
        for speaker_id, score in result.similarity_scores.items():
            info = recognizer.get_speaker_info(speaker_id)
            name = info.name if info else speaker_id
            print(f"    {name}: {score:.3f}")
    
    # 测试未注册的说话人
    print("\n测试 2: 识别未注册的说话人")
    test_audio2 = generate_synthetic_audio(duration=2.0, seed=999)  # 完全不同
    
    result = recognizer.recognize(
        audio_segment=test_audio2,
        timestamp=2.0,
        sample_rate=16000
    )
    
    print(f"识别结果:")
    print(f"  - 是否匹配: {'是' if result.matched else '否'}")
    if not result.matched:
        print(f"  - 未能匹配到已注册说话人")
        if result.similarity_scores:
            max_score = max(result.similarity_scores.values())
            print(f"  - 最高相似度: {max_score:.3f}")


def demo_statistics(recognizer: SpeakerRecognizer):
    """演示统计信息"""
    print("\n" + "="*70)
    print("演示 3: 统计信息")
    print("="*70)
    
    stats = recognizer.get_statistics()
    
    print(f"\n识别器状态: {stats['state']}")
    print(f"\n识别统计:")
    print(f"  - 总识别次数: {stats['total_recognitions']}")
    print(f"  - 成功匹配: {stats['successful_matches']}")
    print(f"  - 失败匹配: {stats['failed_matches']}")
    print(f"  - 成功率: {stats['success_rate']:.1%}")
    
    print(f"\n数据库统计:")
    print(f"  - 档案数量: {stats['database']['total_profiles']}")
    print(f"  - 缓存大小: {stats['database']['cache_size']}")
    print(f"  - 存储目录: {stats['database']['profiles_dir']}")
    
    print(f"\n匹配引擎统计:")
    print(f"  - 当前阈值: {stats['matching']['current_threshold']:.3f}")
    print(f"  - 差值阈值: {stats['matching']['difference_threshold']:.3f}")
    print(f"  - 自适应启用: {'是' if stats['matching']['adaptive_enabled'] else '否'}")


def demo_multiple_speakers():
    """演示多说话人场景"""
    print("\n" + "="*70)
    print("演示 4: 多说话人注册与识别")
    print("="*70)
    
    recognizer = SpeakerRecognizer()
    
    # 注册多个说话人
    friends = [
        ("friend_001", "张三", 42),
        ("friend_002", "李四", 100),
        ("friend_003", "王五", 200),
    ]
    
    print("\n注册多个说话人...")
    for friend_id, name, seed in friends:
        audio_segments = [
            generate_synthetic_audio(duration=2.5, seed=seed + i)
            for i in range(3)
        ]
        
        success = recognizer.register_speaker(
            friend_id=friend_id,
            name=name,
            audio_segments=audio_segments,
            sample_rate=16000
        )
        
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    # 测试识别
    print("\n测试识别各个说话人...")
    for friend_id, name, seed in friends:
        test_audio = generate_synthetic_audio(duration=2.0, seed=seed + 5)
        result = recognizer.recognize(test_audio, sample_rate=16000)
        
        if result.matched:
            matched_info = recognizer.get_speaker_info(result.speaker_id)
            matched_name = matched_info.name if matched_info else result.speaker_id
            status = "✅" if matched_name == name else "⚠️"
            print(f"  {status} 期望: {name}, 识别为: {matched_name} "
                  f"(置信度: {result.confidence:.3f})")
        else:
            print(f"  ❌ 期望: {name}, 未能识别")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("说话人识别模块演示")
    print("="*70)
    print("\n⚠️ 注意: 此演示使用合成音频，仅用于展示功能流程")
    print("   实际使用时应使用真实的人声录音")
    
    try:
        # 演示1: 注册
        recognizer = demo_registration()
        
        if recognizer:
            # 演示2: 识别
            demo_recognition(recognizer)
            
            # 演示3: 统计
            demo_statistics(recognizer)
        
        # 演示4: 多说话人
        demo_multiple_speakers()
        
        print("\n" + "="*70)
        print("🎉 演示完成")
        print("="*70)
        print("\n提示:")
        print("  1. 查看生成的声纹档案: data/speaker_profiles/")
        print("  2. 运行单元测试: python tests/test_speaker_recognition.py")
        print("  3. 安装真实模型: pip install pyannote.audio")
        
    except Exception as e:
        logger.error(f"演示失败: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

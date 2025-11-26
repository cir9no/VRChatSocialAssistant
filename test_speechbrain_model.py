"""
测试 SpeechBrain ECAPA-TDNN 模型加载
"""
import sys
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_model_load():
    """测试模型加载"""
    try:
        from src.speaker_recognition.embedding_engine import EmbeddingEngine
        
        logger.info("=" * 60)
        logger.info("开始测试 SpeechBrain ECAPA-TDNN 模型")
        logger.info("=" * 60)
        
        # 创建引擎
        logger.info("\n1. 初始化 EmbeddingEngine...")
        engine = EmbeddingEngine(
            model_path="models/speaker_recognition/speechbrain",
            device="cpu",  # 使用 CPU 测试
            auto_download=True
        )
        
        # 获取模型信息
        logger.info("\n2. 模型信息:")
        model_info = engine.get_model_info()
        for key, value in model_info.items():
            logger.info(f"   {key}: {value}")
        
        # 测试音频提取
        logger.info("\n3. 测试声纹提取...")
        import numpy as np
        
        # 创建测试音频 (3秒, 16kHz)
        test_audio = np.random.randn(16000 * 3).astype(np.float32) * 0.1
        
        embedding = engine.extract_embedding(test_audio, sample_rate=16000)
        
        logger.info(f"   提取的嵌入向量形状: {embedding.shape}")
        logger.info(f"   嵌入维度: {embedding.shape[0]}")
        logger.info(f"   嵌入范围: [{embedding.min():.4f}, {embedding.max():.4f}]")
        logger.info(f"   嵌入范数: {np.linalg.norm(embedding):.4f}")
        
        # 验证嵌入维度
        assert embedding.shape[0] == 192, f"嵌入维度应该是192，但得到 {embedding.shape[0]}"
        
        logger.info("\n" + "=" * 60)
        logger.info("✓ 所有测试通过!")
        logger.info("✓ SpeechBrain ECAPA-TDNN 模型工作正常")
        logger.info("=" * 60)
        
        return True
        
    except ImportError as e:
        logger.error(f"\n✗ 导入错误: {e}")
        logger.error("请确保已安装 speechbrain:")
        logger.error("  pip install speechbrain")
        return False
        
    except Exception as e:
        logger.error(f"\n✗ 测试失败: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = test_model_load()
    sys.exit(0 if success else 1)

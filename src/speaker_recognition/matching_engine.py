"""
声纹匹配引擎

负责计算声纹相似度和执行匹配决策
"""

import logging
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta

from .models import MatchingConfig, SpeakerResult

logger = logging.getLogger(__name__)


class MatchingEngine:
    """
    声纹匹配引擎
    
    计算声纹相似度并执行匹配决策
    """
    
    def __init__(self, config: Optional[MatchingConfig] = None):
        """
        初始化匹配引擎
        
        Args:
            config: 匹配配置对象
        """
        self.config = config or MatchingConfig()
        self.config.validate()
        
        # 自适应阈值相关
        self.adaptive_enabled = self.config.enable_adaptive_threshold
        self.current_threshold = self.config.base_threshold
        
        # 统计信息（用于自适应调整）
        self.total_matches = 0
        self.false_rejects = 0  # 误拒（需要人工标注）
        self.false_accepts = 0  # 误识（需要人工标注）
        self.ambiguous_matches = 0  # 歧义匹配
        
        self.last_adapt_time = datetime.now()
        self.adapt_interval = timedelta(minutes=10)
        
        logger.info(f"MatchingEngine 初始化: method={self.config.similarity_method}, "
                   f"threshold={self.current_threshold}")
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算两个嵌入向量的相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
        
        Returns:
            相似度分数 [0, 1]（余弦相似度）或 [0, +∞)（欧氏距离）
        """
        if self.config.similarity_method == "cosine":
            # 余弦相似度
            similarity = self._cosine_similarity(embedding1, embedding2)
        elif self.config.similarity_method == "euclidean":
            # 欧氏距离（转换为相似度）
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1.0 / (1.0 + distance)
        else:
            raise ValueError(f"不支持的相似度方法: {self.config.similarity_method}")
        
        return float(similarity)
    
    def _cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        计算余弦相似度
        
        Args:
            embedding1: 第一个嵌入向量
            embedding2: 第二个嵌入向量
        
        Returns:
            余弦相似度 [0, 1]
        """
        # 归一化
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度
        dot_product = np.dot(embedding1, embedding2)
        cosine_sim = dot_product / (norm1 * norm2)
        
        # 将[-1, 1]映射到[0, 1]
        similarity = (cosine_sim + 1.0) / 2.0
        
        return float(np.clip(similarity, 0.0, 1.0))
    
    def match(
        self,
        test_embedding: np.ndarray,
        registered_embeddings: Dict[str, np.ndarray],
        timestamp: float = 0.0
    ) -> SpeakerResult:
        """
        匹配测试嵌入与已注册的声纹
        
        Args:
            test_embedding: 测试音频的嵌入向量
            registered_embeddings: 已注册的声纹字典 {friend_id: embedding}
            timestamp: 时间戳
        
        Returns:
            匹配结果
        """
        import time
        start_time = time.time()
        
        # 如果没有注册声纹，直接返回未匹配
        if not registered_embeddings:
            logger.debug("没有已注册声纹，跳过匹配")
            return SpeakerResult(
                matched=False,
                timestamp=timestamp,
                processing_time=(time.time() - start_time) * 1000
            )
        
        # 计算与所有注册声纹的相似度
        similarity_scores = {}
        for friend_id, registered_embedding in registered_embeddings.items():
            similarity = self.compute_similarity(test_embedding, registered_embedding)
            similarity_scores[friend_id] = similarity
        
        # 找出最高和次高相似度
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        best_match_id, best_score = sorted_scores[0]
        second_best_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        
        # 应用匹配决策逻辑
        matched = False
        final_speaker_id = None
        
        # 1. 检查是否超过基础阈值
        if best_score < self.current_threshold:
            logger.debug(f"相似度低于阈值: {best_score:.3f} < {self.current_threshold:.3f}")
        # 2. 检查是否低于拒绝阈值
        elif best_score < self.config.reject_threshold:
            logger.debug(f"相似度低于拒绝阈值: {best_score:.3f} < {self.config.reject_threshold:.3f}")
        # 3. 检查是否存在歧义（两个候选相似度接近）
        elif (best_score - second_best_score) < self.config.difference_threshold:
            logger.debug(f"歧义匹配: diff={best_score - second_best_score:.3f} < "
                        f"{self.config.difference_threshold:.3f}")
            self.ambiguous_matches += 1
        else:
            # 匹配成功
            matched = True
            final_speaker_id = best_match_id
            logger.debug(f"匹配成功: speaker_id={final_speaker_id}, "
                        f"confidence={best_score:.3f}")
        
        # 更新统计
        self.total_matches += 1
        
        # 检查是否需要自适应调整阈值
        if self.adaptive_enabled:
            self._maybe_adapt_threshold()
        
        # 构造结果
        result = SpeakerResult(
            speaker_id=final_speaker_id,
            confidence=best_score,
            matched=matched,
            similarity_scores=similarity_scores,
            timestamp=timestamp,
            processing_time=(time.time() - start_time) * 1000
        )
        
        return result
    
    def _maybe_adapt_threshold(self):
        """
        自适应阈值调整
        
        根据统计信息动态调整阈值
        """
        # 检查是否满足调整条件
        time_elapsed = datetime.now() - self.last_adapt_time
        
        if self.total_matches < 100 and time_elapsed < self.adapt_interval:
            return  # 样本不足或时间未到
        
        # 计算统计指标
        if self.total_matches == 0:
            return
        
        # 注意：误拒率和误识率需要人工标注，这里仅作示例
        # 实际应用中可以通过用户反馈收集数据
        ambiguous_rate = self.ambiguous_matches / self.total_matches
        
        old_threshold = self.current_threshold
        
        # 根据歧义率调整差值阈值
        if ambiguous_rate > 0.15:
            self.config.difference_threshold = min(
                self.config.difference_threshold + 0.01,
                0.20
            )
            logger.info(f"歧义率过高({ambiguous_rate:.2%})，提高差值阈值至 "
                       f"{self.config.difference_threshold:.3f}")
        
        # 重置统计
        self.last_adapt_time = datetime.now()
        # 不重置计数器，保留累计统计
        
        if old_threshold != self.current_threshold:
            logger.info(f"自适应阈值调整: {old_threshold:.3f} -> {self.current_threshold:.3f}")
    
    def update_threshold(self, threshold: float):
        """
        手动更新基础阈值
        
        Args:
            threshold: 新的阈值 [0.5, 0.95]
        """
        if not (0.5 <= threshold <= 0.95):
            raise ValueError(f"阈值必须在[0.5, 0.95]范围内，当前值：{threshold}")
        
        old_threshold = self.current_threshold
        self.current_threshold = threshold
        self.config.base_threshold = threshold
        
        logger.info(f"手动更新阈值: {old_threshold:.3f} -> {threshold:.3f}")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_matches = 0
        self.false_rejects = 0
        self.false_accepts = 0
        self.ambiguous_matches = 0
        self.last_adapt_time = datetime.now()
        logger.info("重置匹配统计信息")
    
    def get_statistics(self) -> dict:
        """
        获取匹配统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_matches': self.total_matches,
            'false_rejects': self.false_rejects,
            'false_accepts': self.false_accepts,
            'ambiguous_matches': self.ambiguous_matches,
            'current_threshold': self.current_threshold,
            'difference_threshold': self.config.difference_threshold,
            'adaptive_enabled': self.adaptive_enabled,
        }
        
        # 计算比率
        if self.total_matches > 0:
            stats['ambiguous_rate'] = self.ambiguous_matches / self.total_matches
            if self.false_rejects + self.false_accepts > 0:
                stats['false_reject_rate'] = self.false_rejects / self.total_matches
                stats['false_accept_rate'] = self.false_accepts / self.total_matches
        
        return stats

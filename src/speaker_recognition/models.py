"""
数据模型定义

定义说话人识别模块使用的数据结构
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import numpy as np


@dataclass
class SpeakerResult:
    """
    说话人识别结果
    """
    speaker_id: Optional[str] = None  # 匹配到的说话人ID（UUID）
    confidence: float = 0.0  # 匹配置信度 [0, 1]
    matched: bool = False  # 是否匹配成功
    similarity_scores: Dict[str, float] = field(default_factory=dict)  # 与所有候选的相似度
    timestamp: float = 0.0  # 时间戳
    processing_time: float = 0.0  # 处理耗时（毫秒）
    
    def __str__(self):
        if self.matched:
            return f"SpeakerResult(speaker={self.speaker_id}, confidence={self.confidence:.3f})"
        else:
            return "SpeakerResult(no_match)"


@dataclass
class ProfileData:
    """
    声纹档案数据
    """
    friend_id: str  # 好友唯一标识（UUID）
    name: str  # 好友昵称
    embedding: np.ndarray  # 声纹嵌入向量 (192,)
    embedding_version: int = 1  # 声纹版本号
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())  # 注册时间
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())  # 最后更新时间
    sample_count: int = 1  # 注册时使用的音频样本数
    avg_duration: float = 0.0  # 平均样本时长（秒）
    model_version: str = "ecapa-tdnn-v1"  # 使用的模型版本
    embedding_file: str = ""  # 嵌入向量文件名
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.embedding_file:
            self.embedding_file = f"{self.friend_id}.npy"
    
    def to_dict(self) -> dict:
        """转换为字典（不包含embedding数组）"""
        return {
            'friend_id': self.friend_id,
            'name': self.name,
            'embedding_version': self.embedding_version,
            'registered_at': self.registered_at,
            'updated_at': self.updated_at,
            'sample_count': self.sample_count,
            'avg_duration': self.avg_duration,
            'model_version': self.model_version,
            'embedding_file': self.embedding_file,
        }
    
    @classmethod
    def from_dict(cls, data: dict, embedding: np.ndarray) -> 'ProfileData':
        """从字典创建实例"""
        return cls(
            friend_id=data['friend_id'],
            name=data['name'],
            embedding=embedding,
            embedding_version=data.get('embedding_version', 1),
            registered_at=data.get('registered_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            sample_count=data.get('sample_count', 1),
            avg_duration=data.get('avg_duration', 0.0),
            model_version=data.get('model_version', 'ecapa-tdnn-v1'),
            embedding_file=data.get('embedding_file', ''),
        )


@dataclass
class AudioSegment:
    """
    音频片段数据
    """
    audio_data: np.ndarray  # 音频数据 (float32)
    sample_rate: int  # 采样率
    timestamp: float  # 时间戳
    duration: float = 0.0  # 时长（秒）
    
    def __post_init__(self):
        """计算时长"""
        if self.duration == 0.0:
            self.duration = len(self.audio_data) / self.sample_rate


@dataclass
class MatchingConfig:
    """
    匹配配置
    """
    similarity_method: str = "cosine"  # 相似度计算方法
    base_threshold: float = 0.75  # 基础阈值
    difference_threshold: float = 0.10  # 差值阈值
    reject_threshold: float = 0.50  # 拒绝阈值
    enable_adaptive_threshold: bool = True  # 是否启用自适应阈值
    
    def validate(self):
        """验证配置有效性"""
        if not (0.5 <= self.base_threshold <= 0.95):
            raise ValueError(f"base_threshold must be in [0.5, 0.95], got {self.base_threshold}")
        if not (0.05 <= self.difference_threshold <= 0.20):
            raise ValueError(f"difference_threshold must be in [0.05, 0.20], got {self.difference_threshold}")
        if self.similarity_method not in ["cosine", "euclidean"]:
            raise ValueError(f"similarity_method must be 'cosine' or 'euclidean', got {self.similarity_method}")

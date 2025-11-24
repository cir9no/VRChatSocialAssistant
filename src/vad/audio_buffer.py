"""
音频缓冲管理

管理音频数据的缓冲和窗口切分
"""

import numpy as np
from collections import deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioBuffer:
    """
    音频缓冲区管理类
    
    提供音频数据的缓冲、窗口切分和管理功能
    """
    
    def __init__(
        self,
        window_size: int = 512,
        max_buffer_size: int = 16000 * 30  # 默认最大 30 秒缓冲
    ):
        """
        初始化音频缓冲区
        
        Args:
            window_size: 处理窗口大小（样本数），默认 512 (32ms @ 16kHz)
            max_buffer_size: 最大缓冲区大小（样本数），防止内存溢出
        """
        self.window_size = window_size
        self.max_buffer_size = max_buffer_size
        
        # 使用 deque 作为环形缓冲区
        self.buffer = deque(maxlen=max_buffer_size)
        
        # 统计信息
        self.total_samples_added = 0
        self.total_samples_dropped = 0
        
        logger.debug(f"AudioBuffer 初始化: window_size={window_size}, max_buffer_size={max_buffer_size}")
    
    def append(self, audio_data: np.ndarray):
        """
        追加音频数据到缓冲区
        
        Args:
            audio_data: 音频数据，numpy array，float32，形状为 (n_samples,)
        """
        if not isinstance(audio_data, np.ndarray):
            raise TypeError("audio_data 必须是 numpy.ndarray 类型")
        
        # 扁平化数组
        if audio_data.ndim > 1:
            audio_data = audio_data.flatten()
        
        # 检查缓冲区是否会溢出
        current_size = len(self.buffer)
        new_samples = len(audio_data)
        
        if current_size + new_samples > self.max_buffer_size:
            # 计算将被丢弃的样本数
            dropped = current_size + new_samples - self.max_buffer_size
            self.total_samples_dropped += dropped
            logger.warning(f"缓冲区溢出，将丢弃 {dropped} 个旧样本")
        
        # 追加数据（deque 会自动丢弃旧数据）
        self.buffer.extend(audio_data)
        self.total_samples_added += new_samples
    
    def get_window(self) -> Optional[np.ndarray]:
        """
        获取一个处理窗口的数据
        
        Returns:
            窗口数据，numpy array，形状为 (window_size,)
            如果缓冲区数据不足，返回 None
        """
        if len(self.buffer) < self.window_size:
            return None
        
        # 提取窗口数据（不从缓冲区移除）
        window_data = np.array(list(self.buffer)[:self.window_size], dtype=np.float32)
        
        return window_data
    
    def consume(self, n_samples: int):
        """
        从缓冲区消费（移除）指定数量的样本
        
        Args:
            n_samples: 要消费的样本数
        """
        if n_samples <= 0:
            return
        
        # 从左侧移除样本
        for _ in range(min(n_samples, len(self.buffer))):
            self.buffer.popleft()
    
    def get_all(self) -> np.ndarray:
        """
        获取缓冲区中的所有数据（不移除）
        
        Returns:
            所有缓冲数据，numpy array
        """
        if len(self.buffer) == 0:
            return np.array([], dtype=np.float32)
        
        return np.array(list(self.buffer), dtype=np.float32)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        logger.debug("缓冲区已清空")
    
    def size(self) -> int:
        """
        获取当前缓冲区大小（样本数）
        
        Returns:
            缓冲区中的样本数
        """
        return len(self.buffer)
    
    def get_statistics(self) -> dict:
        """
        获取缓冲区统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'current_size': len(self.buffer),
            'window_size': self.window_size,
            'max_buffer_size': self.max_buffer_size,
            'total_samples_added': self.total_samples_added,
            'total_samples_dropped': self.total_samples_dropped,
            'utilization': len(self.buffer) / self.max_buffer_size if self.max_buffer_size > 0 else 0
        }

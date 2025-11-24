"""
Silero VAD 模型封装

封装 Silero VAD PyTorch 模型，提供简洁的推理接口
"""

import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SileroVAD:
    """
    Silero VAD 模型封装类
    
    提供流式语音活动检测功能，支持 CPU 和 GPU 推理
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        device: str = "cpu",
        force_reload: bool = False
    ):
        """
        初始化 Silero VAD 模型
        
        Args:
            sample_rate: 音频采样率，支持 8000/16000 Hz
            device: 推理设备，'cpu' 或 'cuda'
            force_reload: 是否强制重新下载模型
        """
        self.sample_rate = sample_rate
        self.device = device
        self.force_reload = force_reload
        
        # 验证采样率
        if sample_rate not in [8000, 16000]:
            raise ValueError(f"不支持的采样率: {sample_rate}，仅支持 8000 或 16000 Hz")
        
        # 验证设备
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA 不可用，回退到 CPU")
            self.device = "cpu"
        
        # 加载模型
        self.model = None
        self._state = None
        self._load_model()
        
        logger.info(f"Silero VAD 已初始化: sample_rate={sample_rate}, device={self.device}")
    
    def _load_model(self):
        """从 torch hub 加载 Silero VAD 模型"""
        try:
            logger.info("正在加载 Silero VAD 模型...")
            
            # 从 torch hub 加载模型
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=self.force_reload,
                onnx=False,
                trust_repo=True
            )
            
            self.model = model.to(self.device)
            self.model.eval()
            
            # 提取工具函数（虽然我们不直接使用，但保留以备后用）
            self._utils = utils
            
            logger.info("Silero VAD 模型加载成功")
            
        except Exception as e:
            logger.error(f"加载 Silero VAD 模型失败: {e}", exc_info=True)
            raise RuntimeError(f"Silero VAD 模型加载失败: {e}")
    
    def predict(self, audio_chunk: np.ndarray) -> float:
        """
        预测音频片段的语音概率
        
        Args:
            audio_chunk: 音频数据，numpy array，float32，形状为 (n_samples,)
                        推荐长度: 512 或 1024 样本
        
        Returns:
            语音概率值，范围 [0.0, 1.0]
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        # 转换为 torch tensor
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).float()
        else:
            audio_tensor = audio_chunk.float()
        
        # 移动到指定设备
        audio_tensor = audio_tensor.to(self.device)
        
        # 确保是 1D tensor
        if audio_tensor.dim() > 1:
            audio_tensor = audio_tensor.squeeze()
        
        try:
            with torch.no_grad():
                # Silero VAD 模型需要输入形状为 (batch_size, samples)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # 推理
                speech_prob = self.model(audio_tensor, self.sample_rate)
                
                # 提取概率值
                if isinstance(speech_prob, torch.Tensor):
                    prob_value = speech_prob.item()
                else:
                    prob_value = float(speech_prob)
                
                return prob_value
                
        except Exception as e:
            logger.error(f"VAD 推理失败: {e}", exc_info=True)
            # 返回中性值，避免中断处理流程
            return 0.5
    
    def reset_states(self):
        """
        重置模型内部状态
        
        用于处理新的音频流或重新开始检测
        """
        self._state = None
        
        # 如果模型有 reset_states 方法，调用它
        if hasattr(self.model, 'reset_states'):
            try:
                self.model.reset_states()
                logger.debug("模型内部状态已重置")
            except Exception as e:
                logger.warning(f"重置模型状态失败: {e}")
    
    def __del__(self):
        """析构函数，释放资源"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

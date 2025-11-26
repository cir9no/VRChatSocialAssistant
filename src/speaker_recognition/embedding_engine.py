"""
声纹提取引擎

负责加载模型并从音频中提取声纹嵌入向量
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
import torchaudio

logger = logging.getLogger(__name__)


class EmbeddingEngine:
    """
    声纹提取引擎
    
    使用 SpeechBrain ECAPA-TDNN 模型提取声纹特征
    """
    
    def __init__(
        self,
        model_path: str = "models/speaker_recognition/speechbrain",
        sample_rate: int = 16000,
        device: str = "auto",
        auto_download: bool = True,
        min_audio_length: float = 0.5,
        max_audio_length: float = 10.0,
    ):
        """
        初始化声纹提取引擎
        
        Args:
            model_path: 模型存储路径（SpeechBrain 模型保存目录）
            sample_rate: 音频采样率
            device: 推理设备 (auto/cpu/cuda)
            auto_download: 是否自动下载模型
            min_audio_length: 最短音频时长（秒）
            max_audio_length: 最长音频时长（秒）
        """
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.min_audio_length = min_audio_length
        self.max_audio_length = max_audio_length
        self.embedding_dim = 192  # ECAPA-TDNN 嵌入维度
        
        # 确定设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"EmbeddingEngine 初始化: device={self.device}, "
                   f"sample_rate={sample_rate}, model=SpeechBrain ECAPA-TDNN")
        
        # 初始化模型
        self.model = None
        self.classifier = None
        self._load_model(auto_download)
    
    def _load_model(self, auto_download: bool = True):
        """
        加载 SpeechBrain ECAPA-TDNN 模型
        
        Args:
            auto_download: 是否自动下载模型
        """
        try:
            logger.info("开始加载 SpeechBrain ECAPA-TDNN 模型...")
            self._load_speechbrain_model()
            logger.info("✓ SpeechBrain ECAPA-TDNN 模型加载成功")
            
        except ImportError as e:
            logger.error(f"SpeechBrain 未安装: {e}")
            logger.info("请安装 SpeechBrain: pip install speechbrain")
            raise
        except Exception as e:
            logger.error(f"加载 SpeechBrain 模型失败: {e}", exc_info=True)
            raise
    
    def _load_speechbrain_model(self):
        """
        加载 SpeechBrain ECAPA-TDNN 模型
        
        使用 speechbrain/spkrec-ecapa-voxceleb 预训练模型
        """
        try:
            from speechbrain.inference.speaker import EncoderClassifier
            
            logger.info("从 HuggingFace 加载 SpeechBrain ECAPA-TDNN 模型...")
            logger.info("模型: speechbrain/spkrec-ecapa-voxceleb")
            
            # 确保保存目录存在
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # 加载预训练模型
            self.classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(self.model_path),
                run_opts={"device": str(self.device)}
            )
            
            # 创建包装器以保持接口一致性
            class SpeechBrainWrapper:
                def __init__(self, classifier, device, embedding_dim=192):
                    self.classifier = classifier
                    self.device = device
                    self.embedding_dim = embedding_dim
                
                def to(self, device):
                    self.device = device
                    return self
                
                def eval(self):
                    return self
                
                def __call__(self, waveform):
                    import torch
                    import numpy as np
                    
                    # 转换为正确格式
                    if isinstance(waveform, np.ndarray):
                        waveform = torch.from_numpy(waveform).float()
                    
                    # SpeechBrain 期望 [batch, time] 格式
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    elif waveform.ndim == 2 and waveform.shape[0] == 1:
                        # 如果是 [1, time] 格式，保持不变
                        pass
                    elif waveform.ndim == 2:
                        # 如果是 [channels, time] 且 channels > 1，取第一个声道
                        waveform = waveform[0:1, :]
                    
                    waveform = waveform.to(self.device)
                    
                    # 提取嵌入
                    with torch.no_grad():
                        embedding = self.classifier.encode_batch(waveform)
                    
                    return embedding
            
            self.model = SpeechBrainWrapper(self.classifier, self.device, self.embedding_dim)
            logger.info(f"✓ 模型加载成功，设备: {self.device}")
            logger.info(f"✓ 嵌入维度: {self.embedding_dim}")
            
        except ImportError:
            logger.error("SpeechBrain 未安装")
            logger.info("请运行: pip install speechbrain")
            raise
        except Exception as e:
            logger.error(f"加载 SpeechBrain 模型失败: {e}")
            raise
    
    def preprocess_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        音频预处理
        
        Args:
            audio_data: 音频数据 (numpy array, float32)
            sample_rate: 原始采样率（如果与目标不同则重采样）
        
        Returns:
            预处理后的音频张量
        """
        # 确保是 numpy array
        if not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # 确保是 float32
        audio_data = audio_data.astype(np.float32)
        
        # 归一化到 [-1, 1]
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / 32768.0
        
        # 转换为 torch tensor
        waveform = torch.from_numpy(audio_data).float()
        
        # 确保是2D张量 [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # 重采样（如果需要）
        if sample_rate is not None and sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # 填充或截断到合适长度
        num_samples = waveform.shape[1]
        target_samples = int(self.sample_rate * 3.0)  # 默认3秒
        
        if num_samples < target_samples:
            # 填充
            padding = target_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif num_samples > target_samples:
            # 截断
            waveform = waveform[:, :target_samples]
        
        return waveform
    
    def extract_embedding(
        self,
        audio_data: Union[np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        从音频中提取声纹嵌入向量
        
        Args:
            audio_data: 音频数据 (numpy array 或 torch tensor)
            sample_rate: 音频采样率
        
        Returns:
            声纹嵌入向量 (numpy array, shape: (192,))
        """
        if self.model is None:
            raise RuntimeError("模型未加载")
        
        try:
            # 预处理音频
            if isinstance(audio_data, np.ndarray):
                waveform = self.preprocess_audio(audio_data, sample_rate)
            else:
                waveform = audio_data
            
            # 移动到设备
            waveform = waveform.to(self.device)
            
            # 提取嵌入
            with torch.no_grad():
                embedding = self.model(waveform)
            
            # 转换为 numpy array
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            # 确保是1D数组
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            
            # 归一化
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            logger.debug(f"提取声纹嵌入: shape={embedding.shape}, "
                        f"norm={np.linalg.norm(embedding):.3f}")
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"提取声纹嵌入失败: {e}", exc_info=True)
            raise
    
    def validate_audio(
        self,
        audio_data: np.ndarray,
        sample_rate: int
    ) -> tuple[bool, str]:
        """
        验证音频质量是否满足要求
        
        Args:
            audio_data: 音频数据
            sample_rate: 采样率
        
        Returns:
            (是否合格, 错误信息)
        """
        # 检查时长
        duration = len(audio_data) / sample_rate
        if duration < self.min_audio_length:
            return False, f"音频时长过短: {duration:.2f}s < {self.min_audio_length}s"
        if duration > self.max_audio_length:
            return False, f"音频时长过长: {duration:.2f}s > {self.max_audio_length}s"
        
        # 检查音量
        rms = np.sqrt(np.mean(audio_data ** 2))
        if rms < 0.001:  # 约 -60dB
            return False, f"音频音量过小: RMS={rms:.6f}"
        
        # 检查是否全为静音
        if np.max(np.abs(audio_data)) < 0.001:
            return False, "音频数据接近静音"
        
        return True, ""
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_path': str(self.model_path),
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'min_audio_length': self.min_audio_length,
            'max_audio_length': self.max_audio_length,
            'model_loaded': self.model is not None,
            'embedding_dim': self.embedding_dim,
        }

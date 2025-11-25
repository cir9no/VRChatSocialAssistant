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
    
    使用 pyannote.audio 的 ECAPA-TDNN 模型提取声纹特征
    """
    
    def __init__(
        self,
        model_path: str = "models/speaker_recognition/ecapa-tdnn/",
        sample_rate: int = 16000,
        device: str = "auto",
        auto_download: bool = True,
        min_audio_length: float = 0.5,
        max_audio_length: float = 10.0,
    ):
        """
        初始化声纹提取引擎
        
        Args:
            model_path: 模型存储路径
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
        
        # 确定设备
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"EmbeddingEngine 初始化: device={self.device}, "
                   f"sample_rate={sample_rate}")
        
        # 初始化模型
        self.model = None
        self._load_model(auto_download)
    
    def _load_model(self, auto_download: bool = True):
        """
        加载声纹提取模型
        
        Args:
            auto_download: 是否自动下载模型
        """
        try:
            # 尝试从本地加载
            if self._check_model_exists():
                self._load_from_local()
            elif auto_download:
                logger.info("本地模型不存在，开始自动下载...")
                self._download_model()
                self._load_from_local()
            else:
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            logger.info("声纹提取模型加载成功")
            
        except Exception as e:
            logger.error(f"加载声纹提取模型失败: {e}", exc_info=True)
            raise
    
    def _check_model_exists(self) -> bool:
        """检查模型文件是否存在"""
        # 检查是否存在模型文件（简化版检查）
        if not self.model_path.exists():
            return False
        
        # 检查关键文件
        config_file = self.model_path / "config.yaml"
        return config_file.exists()
    
    def _download_model(self):
        """
        从 ModelScope 下载模型
        """
        try:
            from modelscope.hub.snapshot_download import snapshot_download
            
            logger.info("从 ModelScope 下载 pyannote.audio ECAPA-TDNN 模型...")
            
            # 创建目录
            self.model_path.mkdir(parents=True, exist_ok=True)
            
            # 下载模型
            # 注意：这里使用的是示例路径，实际需要找到正确的ModelScope模型ID
            model_dir = snapshot_download(
                'iic/speech_ecapa-tdnn_sv_en_voxceleb_16k',
                cache_dir=str(self.model_path.parent)
            )
            
            logger.info(f"模型下载成功: {model_dir}")
            
        except ImportError:
            logger.error("未安装 modelscope，无法自动下载模型")
            logger.info("请手动安装: pip install modelscope")
            raise
        except Exception as e:
            logger.error(f"下载模型失败: {e}", exc_info=True)
            raise
    
    def _load_from_local(self):
        """
        从本地加载模型
        """
        try:
            # 使用 pyannote.audio 加载模型
            from pyannote.audio import Model
            
            model_file = self.model_path / "pytorch_model.bin"
            if not model_file.exists():
                # 尝试从目录加载
                self.model = Model.from_pretrained(str(self.model_path))
            else:
                self.model = Model.from_pretrained(str(model_file))
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"从本地加载模型: {self.model_path}")
            
        except ImportError:
            logger.warning("pyannote.audio 未安装，尝试使用替代方案...")
            self._load_alternative_model()
        except Exception as e:
            logger.error(f"从本地加载模型失败: {e}", exc_info=True)
            raise
    
    def _load_alternative_model(self):
        """
        加载替代模型（简化实现，用于演示）
        
        注意：这是一个占位实现，实际应用中应使用真实的声纹模型
        """
        logger.warning("使用简化的替代模型（仅用于演示）")
        
        # 创建一个简单的随机嵌入提取器（仅用于测试）
        class DummyEmbeddingModel:
            def __init__(self, embedding_dim=192):
                self.embedding_dim = embedding_dim
            
            def to(self, device):
                return self
            
            def eval(self):
                return self
            
            def __call__(self, waveform):
                # 返回随机嵌入（仅用于测试）
                batch_size = waveform.shape[0] if len(waveform.shape) > 1 else 1
                # 使用音频数据的统计特征生成"嵌入"
                if len(waveform.shape) == 1:
                    waveform = waveform.unsqueeze(0)
                
                # 计算简单的音频特征
                mean = torch.mean(waveform, dim=-1)
                std = torch.std(waveform, dim=-1)
                max_val = torch.max(waveform, dim=-1)[0]
                min_val = torch.min(waveform, dim=-1)[0]
                
                # 重复特征以匹配嵌入维度
                features = torch.stack([mean, std, max_val, min_val], dim=-1)
                embedding = features.repeat(1, self.embedding_dim // 4)
                
                return embedding
        
        self.model = DummyEmbeddingModel(embedding_dim=192)
        logger.warning("⚠️ 当前使用的是演示用的简化模型，不具备真实的声纹识别能力")
        logger.warning("⚠️ 请安装 pyannote.audio 以使用真实模型: pip install pyannote.audio")
    
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
        }

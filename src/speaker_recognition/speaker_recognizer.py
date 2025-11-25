"""
说话人识别协调器

统一的模块入口，协调声纹提取、匹配、数据库等子组件
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional, List, Callable
import yaml
import numpy as np

from .models import SpeakerResult, ProfileData, MatchingConfig, AudioSegment
from .embedding_engine import EmbeddingEngine
from .matching_engine import MatchingEngine
from .profile_database import ProfileDatabase

logger = logging.getLogger(__name__)


class RecognizerState(Enum):
    """识别器状态"""
    IDLE = "idle"
    READY = "ready"
    RECOGNIZING = "recognizing"
    ERROR = "error"


class SpeakerRecognizer:
    """
    说话人识别协调器
    
    提供声纹注册、识别、更新的统一接口
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化说话人识别器
        
        Args:
            config_path: 配置文件路径
        """
        self.state = RecognizerState.IDLE
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 初始化子组件
        self.embedding_engine = None
        self.matching_engine = None
        self.profile_database = None
        
        # 统计信息
        self.total_recognitions = 0
        self.successful_matches = 0
        self.failed_matches = 0
        
        # 初始化
        self._initialize()
        
        logger.info("SpeakerRecognizer 初始化完成")
    
    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            配置字典
        """
        if config_path is None:
            config_path = "config/speaker_recognition_config.yaml"
        
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"加载配置文件: {config_path}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'embedding': {
                'model_path': 'models/speaker_recognition/iic/speech_ecapa-tdnn_sv_en_voxceleb_16k',
                'device': 'auto',
                'auto_download': True,
                'sample_rate': 16000,
            },
            'preprocessing': {
                'min_audio_length': 0.5,
                'max_audio_length': 10.0,
                'normalize': True,
            },
            'matching': {
                'similarity_method': 'cosine',
                'base_threshold': 0.75,
                'difference_threshold': 0.10,
                'reject_threshold': 0.50,
                'enable_adaptive_threshold': True,
            },
            'registration': {
                'min_samples': 3,
                'max_samples': 5,
                'min_sample_duration': 2.0,
            },
            'storage': {
                'profiles_dir': 'data/speaker_profiles/',
            },
        }
    
    def _initialize(self):
        """初始化各个子组件"""
        try:
            # 初始化声纹提取引擎
            embedding_config = self.config.get('embedding', {})
            preprocessing_config = self.config.get('preprocessing', {})
            
            self.embedding_engine = EmbeddingEngine(
                model_path=embedding_config.get('model_path', 'models/speaker_recognition/ecapa-tdnn/'),
                sample_rate=embedding_config.get('sample_rate', 16000),
                device=embedding_config.get('device', 'auto'),
                auto_download=embedding_config.get('auto_download', True),
                min_audio_length=preprocessing_config.get('min_audio_length', 0.5),
                max_audio_length=preprocessing_config.get('max_audio_length', 10.0),
            )
            
            # 初始化匹配引擎
            matching_config = self.config.get('matching', {})
            self.matching_engine = MatchingEngine(
                config=MatchingConfig(
                    similarity_method=matching_config.get('similarity_method', 'cosine'),
                    base_threshold=matching_config.get('base_threshold', 0.75),
                    difference_threshold=matching_config.get('difference_threshold', 0.10),
                    reject_threshold=matching_config.get('reject_threshold', 0.50),
                    enable_adaptive_threshold=matching_config.get('enable_adaptive_threshold', True),
                )
            )
            
            # 初始化声纹数据库
            storage_config = self.config.get('storage', {})
            self.profile_database = ProfileDatabase(
                profiles_dir=storage_config.get('profiles_dir', 'data/speaker_profiles/')
            )
            
            self.state = RecognizerState.READY
            logger.info("所有子组件初始化成功")
            
        except Exception as e:
            logger.error(f"初始化失败: {e}", exc_info=True)
            self.state = RecognizerState.ERROR
            raise
    
    def register_speaker(
        self,
        friend_id: str,
        name: str,
        audio_segments: List[np.ndarray],
        sample_rate: int = 16000
    ) -> bool:
        """
        注册新的说话人声纹
        
        Args:
            friend_id: 好友ID（UUID）
            name: 好友昵称
            audio_segments: 音频片段列表（每个为numpy array）
            sample_rate: 音频采样率
        
        Returns:
            是否注册成功
        """
        if self.state != RecognizerState.READY:
            logger.error(f"识别器状态不正确: {self.state}")
            return False
        
        try:
            logger.info(f"开始注册说话人: friend_id={friend_id}, name={name}, "
                       f"samples={len(audio_segments)}")
            
            # 验证样本数量
            min_samples = self.config.get('registration', {}).get('min_samples', 3)
            if len(audio_segments) < min_samples:
                logger.error(f"音频样本数量不足: {len(audio_segments)} < {min_samples}")
                return False
            
            # 提取每个样本的嵌入向量
            embeddings = []
            total_duration = 0.0
            
            for i, audio_data in enumerate(audio_segments):
                # 验证音频质量
                valid, error_msg = self.embedding_engine.validate_audio(audio_data, sample_rate)
                if not valid:
                    logger.warning(f"样本 {i+1} 质量不合格: {error_msg}")
                    continue
                
                # 提取嵌入
                embedding = self.embedding_engine.extract_embedding(audio_data, sample_rate)
                embeddings.append(embedding)
                
                # 统计时长
                duration = len(audio_data) / sample_rate
                total_duration += duration
                
                logger.debug(f"样本 {i+1}/{len(audio_segments)} 嵌入提取成功: "
                           f"shape={embedding.shape}, duration={duration:.2f}s")
            
            if len(embeddings) < min_samples:
                logger.error(f"有效样本数量不足: {len(embeddings)} < {min_samples}")
                return False
            
            # 计算平均嵌入向量
            avg_embedding = np.mean(embeddings, axis=0)
            
            # 归一化
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            
            # 保存到数据库
            metadata = {
                'name': name,
                'sample_count': len(embeddings),
                'avg_duration': total_duration / len(embeddings),
                'model_version': 'ecapa-tdnn-v1',
            }
            
            success = self.profile_database.save_profile(friend_id, avg_embedding, metadata)
            
            if success:
                logger.info(f"说话人注册成功: friend_id={friend_id}, "
                           f"embeddings={len(embeddings)}, "
                           f"avg_shape={avg_embedding.shape}")
            
            return success
            
        except Exception as e:
            logger.error(f"注册说话人失败: {e}", exc_info=True)
            return False
    
    def recognize(
        self,
        audio_segment: np.ndarray,
        timestamp: float = 0.0,
        sample_rate: int = 16000
    ) -> SpeakerResult:
        """
        识别音频片段的说话人
        
        Args:
            audio_segment: 音频片段（numpy array）
            timestamp: 时间戳
            sample_rate: 音频采样率
        
        Returns:
            识别结果
        """
        if self.state != RecognizerState.READY:
            logger.error(f"识别器状态不正确: {self.state}")
            return SpeakerResult(matched=False, timestamp=timestamp)
        
        start_time = time.time()
        self.total_recognitions += 1
        
        try:
            # 验证音频质量
            valid, error_msg = self.embedding_engine.validate_audio(audio_segment, sample_rate)
            if not valid:
                logger.debug(f"音频质量不合格: {error_msg}")
                self.failed_matches += 1
                return SpeakerResult(
                    matched=False,
                    timestamp=timestamp,
                    processing_time=(time.time() - start_time) * 1000
                )
            
            # 提取嵌入向量
            test_embedding = self.embedding_engine.extract_embedding(audio_segment, sample_rate)
            
            # 加载所有已注册声纹
            profiles = self.profile_database.load_all_profiles()
            if not profiles:
                logger.debug("没有已注册声纹，跳过匹配")
                self.failed_matches += 1
                return SpeakerResult(
                    matched=False,
                    timestamp=timestamp,
                    processing_time=(time.time() - start_time) * 1000
                )
            
            # 提取嵌入向量字典
            registered_embeddings = {
                friend_id: profile.embedding
                for friend_id, profile in profiles.items()
            }
            
            # 执行匹配
            result = self.matching_engine.match(
                test_embedding,
                registered_embeddings,
                timestamp
            )
            
            # 更新统计
            if result.matched:
                self.successful_matches += 1
                logger.debug(f"识别成功: speaker={result.speaker_id}, "
                           f"confidence={result.confidence:.3f}")
            else:
                self.failed_matches += 1
            
            return result
            
        except Exception as e:
            logger.error(f"识别失败: {e}", exc_info=True)
            self.failed_matches += 1
            return SpeakerResult(
                matched=False,
                timestamp=timestamp,
                processing_time=(time.time() - start_time) * 1000
            )
    
    def update_speaker_profile(
        self,
        friend_id: str,
        audio_segment: np.ndarray,
        sample_rate: int = 16000,
        weight: float = 0.2
    ) -> bool:
        """
        更新已注册的说话人声纹（持续学习）
        
        Args:
            friend_id: 好友ID
            audio_segment: 新的音频片段
            sample_rate: 音频采样率
            weight: 新样本权重（0.0-1.0）
        
        Returns:
            是否更新成功
        """
        try:
            # 加载现有档案
            profile = self.profile_database.load_profile(friend_id)
            if profile is None:
                logger.error(f"档案不存在: friend_id={friend_id}")
                return False
            
            # 提取新嵌入
            new_embedding = self.embedding_engine.extract_embedding(audio_segment, sample_rate)
            
            # 计算加权平均
            old_weight = 1.0 - weight
            updated_embedding = old_weight * profile.embedding + weight * new_embedding
            
            # 归一化
            updated_embedding = updated_embedding / (np.linalg.norm(updated_embedding) + 1e-8)
            
            # 更新数据库
            success = self.profile_database.update_profile(
                friend_id,
                updated_embedding,
                {'sample_count': profile.sample_count + 1}
            )
            
            if success:
                logger.info(f"声纹档案更新成功: friend_id={friend_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"更新声纹档案失败: {e}", exc_info=True)
            return False
    
    def remove_speaker(self, friend_id: str) -> bool:
        """
        删除已注册的说话人
        
        Args:
            friend_id: 好友ID
        
        Returns:
            是否删除成功
        """
        return self.profile_database.delete_profile(friend_id)
    
    def get_registered_speakers(self) -> List[str]:
        """
        获取所有已注册的说话人ID列表
        
        Returns:
            好友ID列表
        """
        return self.profile_database.list_profiles()
    
    def get_speaker_info(self, friend_id: str) -> Optional[ProfileData]:
        """
        获取说话人详细信息
        
        Args:
            friend_id: 好友ID
        
        Returns:
            ProfileData对象或None
        """
        return self.profile_database.load_profile(friend_id)
    
    def get_statistics(self) -> dict:
        """
        获取识别统计信息
        
        Returns:
            统计信息字典
        """
        db_stats = self.profile_database.get_statistics()
        matching_stats = self.matching_engine.get_statistics()
        
        stats = {
            'state': self.state.value,
            'total_recognitions': self.total_recognitions,
            'successful_matches': self.successful_matches,
            'failed_matches': self.failed_matches,
            'success_rate': self.successful_matches / self.total_recognitions if self.total_recognitions > 0 else 0.0,
            'database': db_stats,
            'matching': matching_stats,
        }
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_recognitions = 0
        self.successful_matches = 0
        self.failed_matches = 0
        self.matching_engine.reset_statistics()
        logger.info("统计信息已重置")

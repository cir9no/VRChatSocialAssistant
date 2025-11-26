"""
声纹数据库

负责存储和管理已注册的好友声纹数据
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np

from .models import ProfileData

logger = logging.getLogger(__name__)


class ProfileDatabase:
    """
    声纹档案数据库
    
    使用JSON+NPY文件存储声纹数据
    """
    
    def __init__(self, profiles_dir: str = "data/speaker_profiles/"):
        """
        初始化声纹数据库
        
        Args:
            profiles_dir: 声纹档案存储目录
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        # 缓存已加载的档案
        self._cache: Dict[str, ProfileData] = {}
        self._cache_enabled = True
        
        logger.info(f"ProfileDatabase 初始化: dir={self.profiles_dir}")
    
    def save_profile(
        self,
        friend_id: str,
        embedding: np.ndarray,
        metadata: dict
    ) -> bool:
        """
        保存声纹档案
        
        Args:
            friend_id: 好友ID
            embedding: 声纹嵌入向量
            metadata: 元数据字典
        
        Returns:
            是否保存成功
        """
        try:
            # 创建ProfileData对象
            profile = ProfileData(
                friend_id=friend_id,
                name=metadata.get('name', friend_id),
                embedding=embedding,
                sample_count=metadata.get('sample_count', 1),
                avg_duration=metadata.get('avg_duration', 0.0),
                model_version=metadata.get('model_version', 'speechbrain-ecapa-tdnn-v1'),
            )
            
            # 保存嵌入向量（NPY文件）
            embedding_path = self.profiles_dir / f"{friend_id}.npy"
            np.save(embedding_path, embedding.astype(np.float32))
            
            # 保存元数据（JSON文件）
            metadata_path = self.profiles_dir / f"{friend_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            if self._cache_enabled:
                self._cache[friend_id] = profile
            
            logger.info(f"保存声纹档案成功: friend_id={friend_id}, "
                       f"embedding_shape={embedding.shape}")
            return True
            
        except Exception as e:
            logger.error(f"保存声纹档案失败: friend_id={friend_id}, error={e}",
                        exc_info=True)
            return False
    
    def load_profile(self, friend_id: str) -> Optional[ProfileData]:
        """
        加载声纹档案
        
        Args:
            friend_id: 好友ID
        
        Returns:
            ProfileData对象，如果不存在返回None
        """
        # 检查缓存
        if self._cache_enabled and friend_id in self._cache:
            return self._cache[friend_id]
        
        try:
            # 检查文件是否存在
            metadata_path = self.profiles_dir / f"{friend_id}.json"
            embedding_path = self.profiles_dir / f"{friend_id}.npy"
            
            if not metadata_path.exists() or not embedding_path.exists():
                logger.warning(f"声纹档案不存在: friend_id={friend_id}")
                return None
            
            # 加载元数据
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 加载嵌入向量
            embedding = np.load(embedding_path).astype(np.float32)
            
            # 创建ProfileData对象
            profile = ProfileData.from_dict(metadata, embedding)
            
            # 更新缓存
            if self._cache_enabled:
                self._cache[friend_id] = profile
            
            logger.debug(f"加载声纹档案成功: friend_id={friend_id}")
            return profile
            
        except Exception as e:
            logger.error(f"加载声纹档案失败: friend_id={friend_id}, error={e}",
                        exc_info=True)
            return None
    
    def update_profile(
        self,
        friend_id: str,
        embedding: np.ndarray,
        update_metadata: Optional[dict] = None
    ) -> bool:
        """
        更新声纹档案（持续学习）
        
        Args:
            friend_id: 好友ID
            embedding: 新的声纹嵌入向量
            update_metadata: 需要更新的元数据
        
        Returns:
            是否更新成功
        """
        try:
            # 加载现有档案
            existing_profile = self.load_profile(friend_id)
            if existing_profile is None:
                logger.warning(f"档案不存在，无法更新: friend_id={friend_id}")
                return False
            
            # 更新嵌入向量
            embedding_path = self.profiles_dir / f"{friend_id}.npy"
            np.save(embedding_path, embedding.astype(np.float32))
            
            # 更新元数据
            existing_profile.embedding = embedding
            existing_profile.updated_at = datetime.now().isoformat()
            existing_profile.embedding_version += 1
            
            if update_metadata:
                for key, value in update_metadata.items():
                    if hasattr(existing_profile, key):
                        setattr(existing_profile, key, value)
            
            # 保存元数据
            metadata_path = self.profiles_dir / f"{friend_id}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing_profile.to_dict(), f, ensure_ascii=False, indent=2)
            
            # 更新缓存
            if self._cache_enabled:
                self._cache[friend_id] = existing_profile
            
            logger.info(f"更新声纹档案成功: friend_id={friend_id}, "
                       f"version={existing_profile.embedding_version}")
            return True
            
        except Exception as e:
            logger.error(f"更新声纹档案失败: friend_id={friend_id}, error={e}",
                        exc_info=True)
            return False
    
    def delete_profile(self, friend_id: str) -> bool:
        """
        删除声纹档案
        
        Args:
            friend_id: 好友ID
        
        Returns:
            是否删除成功
        """
        try:
            metadata_path = self.profiles_dir / f"{friend_id}.json"
            embedding_path = self.profiles_dir / f"{friend_id}.npy"
            
            # 删除文件
            if metadata_path.exists():
                metadata_path.unlink()
            if embedding_path.exists():
                embedding_path.unlink()
            
            # 清除缓存
            if friend_id in self._cache:
                del self._cache[friend_id]
            
            logger.info(f"删除声纹档案成功: friend_id={friend_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除声纹档案失败: friend_id={friend_id}, error={e}",
                        exc_info=True)
            return False
    
    def list_profiles(self) -> List[str]:
        """
        列出所有已注册的好友ID
        
        Returns:
            好友ID列表
        """
        try:
            # 查找所有JSON元数据文件
            json_files = list(self.profiles_dir.glob("*.json"))
            friend_ids = [f.stem for f in json_files]
            
            logger.debug(f"列出声纹档案: count={len(friend_ids)}")
            return friend_ids
            
        except Exception as e:
            logger.error(f"列出声纹档案失败: error={e}", exc_info=True)
            return []
    
    def load_all_profiles(self) -> Dict[str, ProfileData]:
        """
        加载所有声纹档案
        
        Returns:
            friend_id -> ProfileData的字典
        """
        profiles = {}
        friend_ids = self.list_profiles()
        
        for friend_id in friend_ids:
            profile = self.load_profile(friend_id)
            if profile is not None:
                profiles[friend_id] = profile
        
        logger.info(f"加载所有声纹档案: count={len(profiles)}")
        return profiles
    
    def clear_cache(self):
        """清空缓存"""
        self._cache.clear()
        logger.debug("清空声纹档案缓存")
    
    def enable_cache(self, enabled: bool = True):
        """
        启用/禁用缓存
        
        Args:
            enabled: 是否启用缓存
        """
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()
        logger.info(f"声纹档案缓存: {'启用' if enabled else '禁用'}")
    
    def get_statistics(self) -> dict:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        friend_ids = self.list_profiles()
        total_size = 0
        
        for friend_id in friend_ids:
            json_path = self.profiles_dir / f"{friend_id}.json"
            npy_path = self.profiles_dir / f"{friend_id}.npy"
            if json_path.exists():
                total_size += json_path.stat().st_size
            if npy_path.exists():
                total_size += npy_path.stat().st_size
        
        return {
            'total_profiles': len(friend_ids),
            'cache_size': len(self._cache),
            'cache_enabled': self._cache_enabled,
            'total_disk_size_bytes': total_size,
            'profiles_dir': str(self.profiles_dir),
        }

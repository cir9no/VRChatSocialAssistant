"""
音频设备管理模块

负责列举、选择和管理系统音频输入/输出设备（支持 WASAPI Loopback）
"""

import pyaudiowpatch as pyaudio
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """音频设备管理器（支持 WASAPI Loopback）"""
    
    def __init__(self):
        """初始化设备管理器"""
        self.p = pyaudio.PyAudio()
        self._refresh_devices()
    
    def _refresh_devices(self):
        """刷新设备列表"""
        self.device_count = self.p.get_device_count()
        logger.info(f"检测到 {self.device_count} 个音频设备")
    
    def list_devices(self) -> List[Dict]:
        """
        列出所有音频设备
        
        Returns:
            设备列表，每个设备包含详细信息
        """
        devices = []
        for idx in range(self.device_count):
            try:
                info = self.p.get_device_info_by_index(idx)
                devices.append({
                    'index': idx,
                    'name': info['name'],
                    'max_input_channels': info['maxInputChannels'],
                    'max_output_channels': info['maxOutputChannels'],
                    'default_samplerate': int(info['defaultSampleRate']),
                    'hostapi': info['hostApi'],
                    'hostapi_name': self.p.get_host_api_info_by_index(info['hostApi'])['name'],
                    'is_loopback': info.get('isLoopbackDevice', False)
                })
            except Exception as e:
                logger.warning(f"获取设备 {idx} 信息失败: {e}")
        return devices
    
    def list_input_devices(self) -> List[Dict]:
        """
        列出所有输入设备（麦克风）
        
        Returns:
            输入设备列表
        """
        return [d for d in self.list_devices() if d['max_input_channels'] > 0]
    
    def list_output_devices(self) -> List[Dict]:
        """
        列出所有输出设备（扬声器）
        
        Returns:
            输出设备列表
        """
        return [d for d in self.list_devices() if d['max_output_channels'] > 0]
    
    def list_loopback_devices(self) -> List[Dict]:
        """
        列出所有 WASAPI Loopback 设备
        
        Returns:
            Loopback 设备列表（系统输出设备的回环版本）
        """
        loopback_devices = []
        for device in self.list_devices():
            # pyaudiowpatch 会自动为 WASAPI 输出设备创建对应的 loopback 设备
            if device.get('is_loopback', False):
                loopback_devices.append(device)
        
        return loopback_devices
    
    def get_default_input_device(self) -> Optional[Dict]:
        """
        获取默认输入设备
        
        Returns:
            默认输入设备信息，如果没有则返回 None
        """
        try:
            default_info = self.p.get_default_input_device_info()
            if default_info:
                return self.get_device_info(default_info['index'])
        except Exception as e:
            logger.error(f"获取默认输入设备失败: {e}")
        return None
    
    def get_default_output_device(self) -> Optional[Dict]:
        """
        获取默认输出设备
        
        Returns:
            默认输出设备信息，如果没有则返回 None
        """
        try:
            default_info = self.p.get_default_output_device_info()
            if default_info:
                return self.get_device_info(default_info['index'])
        except Exception as e:
            logger.error(f"获取默认输出设备失败: {e}")
        return None
    
    def get_default_wasapi_loopback(self) -> Optional[Dict]:
        """
        获取默认 WASAPI Loopback 设备（默认扬声器的 loopback）
        
        Returns:
            默认 Loopback 设备信息，如果没有则返回 None
        """
        try:
            # 获取默认 WASAPI loopback 设备
            wasapi_info = self.p.get_default_wasapi_loopback()
            if wasapi_info:
                return self.get_device_info(wasapi_info['index'])
        except Exception as e:
            logger.warning(f"获取默认 WASAPI Loopback 失败: {e}")
            # 如果失败，尝试从列表中查找第一个 loopback 设备
            loopback_devices = self.list_loopback_devices()
            if loopback_devices:
                return loopback_devices[0]
        return None
    
    def find_device_by_name(self, name: str, device_type: str = 'all') -> Optional[Dict]:
        """
        根据名称查找设备
        
        Args:
            name: 设备名称（支持部分匹配）
            device_type: 设备类型 ('input', 'output', 'loopback', 'all')
        
        Returns:
            匹配的设备信息，如果没有找到则返回 None
        """
        if device_type == 'input':
            devices = self.list_input_devices()
        elif device_type == 'output':
            devices = self.list_output_devices()
        elif device_type == 'loopback':
            devices = self.list_loopback_devices()
        else:
            devices = self.list_devices()
        
        name_lower = name.lower()
        for device in devices:
            if name_lower in device['name'].lower():
                return device
        
        return None
    
    def get_device_info(self, device_index: int) -> Optional[Dict]:
        """
        获取指定索引的设备信息
        
        Args:
            device_index: 设备索引
        
        Returns:
            设备信息，如果索引无效则返回 None
        """
        try:
            info = self.p.get_device_info_by_index(device_index)
            return {
                'index': device_index,
                'name': info['name'],
                'max_input_channels': info['maxInputChannels'],
                'max_output_channels': info['maxOutputChannels'],
                'default_samplerate': int(info['defaultSampleRate']),
                'hostapi': info['hostApi'],
                'hostapi_name': self.p.get_host_api_info_by_index(info['hostApi'])['name'],
                'is_loopback': info.get('isLoopbackDevice', False)
            }
        except Exception as e:
            logger.error(f"获取设备 {device_index} 信息失败: {e}")
        return None
    
    def check_device_support(self, device_index: int, samplerate: int, 
                            channels: int, is_input: bool = True) -> bool:
        """
        检查设备是否支持指定的采样率和声道数
        
        Args:
            device_index: 设备索引
            samplerate: 采样率
            channels: 声道数
            is_input: 是否为输入设备
        
        Returns:
            是否支持
        """
        try:
            device_info = self.p.get_device_info_by_index(device_index)
            if is_input:
                supported = self.p.is_format_supported(
                    samplerate,
                    input_device=device_index,
                    input_channels=channels,
                    input_format=pyaudio.paInt16
                )
            else:
                supported = self.p.is_format_supported(
                    samplerate,
                    output_device=device_index,
                    output_channels=channels,
                    output_format=pyaudio.paInt16
                )
            return supported
        except Exception as e:
            logger.warning(f"设备 {device_index} 不支持参数 (sr={samplerate}, ch={channels}): {e}")
            return False
    
    def print_device_list(self):
        """打印所有设备信息（用于调试）"""
        print("\n=== 所有音频设备 ===")
        for device in self.list_devices():
            loopback_mark = " [LOOPBACK]" if device.get('is_loopback', False) else ""
            print(f"[{device['index']}] {device['name']}{loopback_mark}")
            print(f"    输入通道: {device['max_input_channels']}, "
                  f"输出通道: {device['max_output_channels']}")
            print(f"    默认采样率: {device['default_samplerate']}")
            print(f"    Host API: {device['hostapi_name']}\n")
        
        print("\n=== 输入设备（麦克风）===")
        for device in self.list_input_devices():
            print(f"[{device['index']}] {device['name']}")
        
        print("\n=== WASAPI Loopback 设备（系统音频）===")
        loopback = self.list_loopback_devices()
        if loopback:
            for device in loopback:
                print(f"[{device['index']}] {device['name']}")
        else:
            print("未检测到 WASAPI Loopback 设备")
        
        print("\n=== 默认设备 ===")
        default_input = self.get_default_input_device()
        default_output = self.get_default_output_device()
        default_loopback = self.get_default_wasapi_loopback()
        
        if default_input:
            print(f"默认输入: [{default_input['index']}] {default_input['name']}")
        if default_output:
            print(f"默认输出: [{default_output['index']}] {default_output['name']}")
        if default_loopback:
            print(f"默认 Loopback: [{default_loopback['index']}] {default_loopback['name']}")
    
    def close(self):
        """关闭设备管理器，释放资源"""
        if hasattr(self, 'p'):
            self.p.terminate()
            logger.info("设备管理器已关闭")
    
    def __del__(self):
        """析构函数"""
        self.close()

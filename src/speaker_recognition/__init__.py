"""
说话人识别模块

基于声纹注册的说话人验证系统，用于从音频流中识别目标好友的声音。
"""

from .speaker_recognizer import SpeakerRecognizer
from .models import SpeakerResult, ProfileData

__all__ = [
    'SpeakerRecognizer',
    'SpeakerResult',
    'ProfileData',
]

__version__ = '1.0.0'

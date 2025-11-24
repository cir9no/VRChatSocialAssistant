"""
VAD (Voice Activity Detection) 模块

提供语音活动检测功能，用于从连续音频流中检测和切分语音片段。
"""

from .vad_detector import VADDetector
from .silero_vad import SileroVAD
from .audio_buffer import AudioBuffer

__all__ = ['VADDetector', 'SileroVAD', 'AudioBuffer']

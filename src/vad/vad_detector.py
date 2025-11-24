"""
VAD 检测器主类

实现语音活动检测的主要逻辑，包括状态机管理和语音片段切分
"""

import numpy as np
import time
import uuid
from enum import Enum
from typing import Optional, Callable, Dict, List
import logging

from .silero_vad import SileroVAD
from .audio_buffer import AudioBuffer

logger = logging.getLogger(__name__)


class VADState(Enum):
    """VAD 状态枚举"""
    IDLE = "idle"                      # 空闲状态（静音）
    SPEECH_START = "speech_start"      # 语音开始
    SPEECH_ONGOING = "speech_ongoing"  # 语音进行中
    SPEECH_END = "speech_end"          # 语音结束（等待确认）


class VADDetector:
    """
    VAD 检测器主类
    
    管理音频流处理、语音片段检测和输出
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        max_speech_duration_ms: int = 10000,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 30,
        window_size_samples: int = 512,
        device: str = "cpu",
        debug: bool = False
    ):
        """
        初始化 VAD 检测器
        
        Args:
            sample_rate: 音频采样率，默认 16000 Hz
            threshold: 语音检测阈值 (0.0-1.0)，默认 0.5
            min_speech_duration_ms: 最小语音片段时长（毫秒），默认 250ms
            max_speech_duration_ms: 最大语音片段时长（毫秒），默认 10000ms
            min_silence_duration_ms: 切分静音时长（毫秒），默认 300ms
            speech_pad_ms: 语音片段前后填充时长（毫秒），默认 30ms
            window_size_samples: 处理窗口大小（样本数），默认 512
            device: 推理设备，'cpu' 或 'cuda'
            debug: 调试模式，记录详细日志
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.window_size_samples = window_size_samples
        self.device = device
        self.debug = debug
        
        # 转换时间参数为样本数
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_duration_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_duration_ms / 1000)
        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        
        # 初始化 Silero VAD 模型
        self.model = SileroVAD(sample_rate=sample_rate, device=device)
        
        # 初始化音频缓冲区
        self.buffer = AudioBuffer(
            window_size=window_size_samples,
            max_buffer_size=sample_rate * 60  # 最大 60 秒缓冲
        )
        
        # 状态机
        self.state = VADState.IDLE
        
        # 当前语音片段数据
        self.current_speech: List[np.ndarray] = []
        self.current_speech_confidences: List[float] = []
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.silence_sample_count = 0
        
        # 语音片段前后填充缓冲
        self.pre_speech_buffer: List[np.ndarray] = []
        self.max_pre_speech_samples = self.speech_pad_samples
        
        # 回调函数
        self.callback: Optional[Callable] = None
        
        # 统计信息
        self.total_frames_processed = 0
        self.speech_segments_detected = 0
        self.total_speech_duration = 0.0
        self.frames_dropped = 0
        self.processing_times: List[float] = []
        
        logger.info(f"VADDetector 初始化: sr={sample_rate}, threshold={threshold}, "
                   f"min_speech={min_speech_duration_ms}ms, max_speech={max_speech_duration_ms}ms, "
                   f"min_silence={min_silence_duration_ms}ms, device={device}")
    
    def set_callback(self, callback: Callable[[np.ndarray, Dict], None]):
        """
        设置语音片段输出回调函数
        
        Args:
            callback: 回调函数，签名为 callback(speech_segment, metadata)
        """
        self.callback = callback
        logger.info("语音片段输出回调已设置")
    
    def process_audio(self, audio_data: np.ndarray, timestamp: float):
        """
        处理单帧音频数据
        
        Args:
            audio_data: 音频数据，numpy array，float32，形状为 (n_samples,)
            timestamp: 音频帧时间戳
        """
        start_time = time.perf_counter()
        
        try:
            # 验证输入
            if not isinstance(audio_data, np.ndarray):
                logger.error("audio_data 必须是 numpy.ndarray 类型")
                self.frames_dropped += 1
                return
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 添加到缓冲区
            self.buffer.append(audio_data)
            
            # 处理缓冲区中的窗口
            while True:
                window = self.buffer.get_window()
                if window is None:
                    break
                
                # VAD 推理
                speech_prob = self.model.predict(window)
                
                if self.debug:
                    logger.debug(f"VAD 推理: prob={speech_prob:.3f}, state={self.state.value}")
                
                # 状态机处理
                self._process_state_machine(window, speech_prob, timestamp)
                
                # 消费处理过的样本（使用窗口的一半，实现滑动窗口）
                self.buffer.consume(self.window_size_samples // 2)
                
                self.total_frames_processed += 1
            
            # 记录处理时间
            processing_time = (time.perf_counter() - start_time) * 1000  # 转换为毫秒
            self.processing_times.append(processing_time)
            
            # 保留最近 1000 条记录
            if len(self.processing_times) > 1000:
                self.processing_times.pop(0)
            
        except Exception as e:
            logger.error(f"处理音频数据失败: {e}", exc_info=True)
            self.frames_dropped += 1
    
    def _process_state_machine(self, audio_window: np.ndarray, speech_prob: float, timestamp: float):
        """
        处理状态机逻辑
        
        Args:
            audio_window: 当前音频窗口
            speech_prob: 语音概率
            timestamp: 时间戳
        """
        is_speech = speech_prob > self.threshold
        
        if self.state == VADState.IDLE:
            if is_speech:
                # 转换到 SPEECH_START 状态
                self.state = VADState.SPEECH_START
                self.speech_start_time = timestamp
                self.current_speech = []
                self.current_speech_confidences = []
                self.silence_sample_count = 0
                
                # 添加前置填充（如果有）
                if self.pre_speech_buffer:
                    for pre_audio in self.pre_speech_buffer:
                        self.current_speech.append(pre_audio)
                    self.pre_speech_buffer.clear()
                
                # 添加当前窗口
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                
                if self.debug:
                    logger.debug(f"状态转换: IDLE -> SPEECH_START, 置信度: {speech_prob:.3f}")
            else:
                # 保持静音，更新前置缓冲
                self.pre_speech_buffer.append(audio_window.copy())
                # 限制前置缓冲大小
                total_samples = sum(len(a) for a in self.pre_speech_buffer)
                while total_samples > self.max_pre_speech_samples:
                    removed = self.pre_speech_buffer.pop(0)
                    total_samples -= len(removed)
        
        elif self.state == VADState.SPEECH_START:
            if is_speech:
                # 转换到 SPEECH_ONGOING 状态
                self.state = VADState.SPEECH_ONGOING
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                
                if self.debug:
                    logger.debug(f"状态转换: SPEECH_START -> SPEECH_ONGOING")
            else:
                # 可能是短暂静音，先累积
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                self.silence_sample_count += len(audio_window)
        
        elif self.state == VADState.SPEECH_ONGOING:
            if is_speech:
                # 继续累积语音
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                self.silence_sample_count = 0  # 重置静音计数
                
                # 检查是否超过最大时长
                current_samples = sum(len(a) for a in self.current_speech)
                if current_samples >= self.max_speech_samples:
                    if self.debug:
                        logger.debug(f"语音片段达到最大时长，强制切分")
                    self._output_speech_segment(timestamp)
            else:
                # 检测到静音，转换到 SPEECH_END 状态
                self.state = VADState.SPEECH_END
                self.silence_start_time = timestamp
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                self.silence_sample_count = len(audio_window)
                
                if self.debug:
                    logger.debug(f"状态转换: SPEECH_ONGOING -> SPEECH_END")
        
        elif self.state == VADState.SPEECH_END:
            if is_speech:
                # 再次检测到语音，返回 SPEECH_ONGOING 状态
                self.state = VADState.SPEECH_ONGOING
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                self.silence_sample_count = 0
                
                if self.debug:
                    logger.debug(f"状态转换: SPEECH_END -> SPEECH_ONGOING (撤销结束)")
            else:
                # 继续静音
                self.current_speech.append(audio_window.copy())
                self.current_speech_confidences.append(speech_prob)
                self.silence_sample_count += len(audio_window)
                
                # 检查静音是否足够长，可以切分
                if self.silence_sample_count >= self.min_silence_samples:
                    if self.debug:
                        logger.debug(f"静音持续 {self.silence_sample_count} 样本，输出语音片段")
                    self._output_speech_segment(timestamp)
    
    def _output_speech_segment(self, timestamp: float):
        """
        输出语音片段
        
        Args:
            timestamp: 当前时间戳
        """
        if not self.current_speech:
            self.state = VADState.IDLE
            return
        
        # 拼接语音片段
        speech_segment = np.concatenate(self.current_speech)
        
        # 检查是否满足最小时长要求
        if len(speech_segment) < self.min_speech_samples:
            if self.debug:
                logger.debug(f"语音片段过短 ({len(speech_segment)} < {self.min_speech_samples})，丢弃")
            self.state = VADState.IDLE
            self.current_speech.clear()
            self.current_speech_confidences.clear()
            self.pre_speech_buffer.clear()
            return
        
        # 计算平均置信度
        avg_confidence = np.mean(self.current_speech_confidences) if self.current_speech_confidences else 0.0
        
        # 计算时长
        duration = len(speech_segment) / self.sample_rate
        
        # 构建元数据
        segment_id = str(uuid.uuid4())
        metadata = {
            'segment_id': segment_id,
            'start_time': self.speech_start_time,
            'end_time': timestamp,
            'duration': duration,
            'sample_rate': self.sample_rate,
            'avg_confidence': float(avg_confidence),
            'num_samples': len(speech_segment)
        }
        
        # 更新统计信息
        self.speech_segments_detected += 1
        self.total_speech_duration += duration
        
        logger.info(f"检测到语音片段: id={segment_id}, duration={duration:.2f}s, confidence={avg_confidence:.3f}")
        
        # 调用回调函数
        if self.callback:
            try:
                self.callback(speech_segment, metadata)
            except Exception as e:
                logger.error(f"语音片段回调函数执行失败: {e}", exc_info=True)
        
        # 重置状态
        self.state = VADState.IDLE
        self.current_speech.clear()
        self.current_speech_confidences.clear()
        self.pre_speech_buffer.clear()
        self.silence_sample_count = 0
    
    def reset(self):
        """重置检测器状态"""
        self.state = VADState.IDLE
        self.current_speech.clear()
        self.current_speech_confidences.clear()
        self.pre_speech_buffer.clear()
        self.speech_start_time = None
        self.silence_start_time = None
        self.silence_sample_count = 0
        self.buffer.clear()
        self.model.reset_states()
        
        logger.info("VADDetector 状态已重置")
    
    def update_threshold(self, threshold: float):
        """
        动态更新检测阈值
        
        Args:
            threshold: 新的阈值 (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold 必须在 0.0 到 1.0 之间")
        
        old_threshold = self.threshold
        self.threshold = threshold
        logger.info(f"检测阈值已更新: {old_threshold:.2f} -> {threshold:.2f}")
    
    def get_statistics(self) -> Dict:
        """
        获取检测统计信息
        
        Returns:
            统计信息字典
        """
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        avg_speech_duration = (self.total_speech_duration / self.speech_segments_detected 
                              if self.speech_segments_detected > 0 else 0.0)
        
        buffer_stats = self.buffer.get_statistics()
        
        return {
            'total_frames_processed': self.total_frames_processed,
            'speech_segments_detected': self.speech_segments_detected,
            'total_speech_duration': self.total_speech_duration,
            'avg_speech_duration': avg_speech_duration,
            'frames_dropped': self.frames_dropped,
            'avg_processing_time_ms': avg_processing_time,
            'current_state': self.state.value,
            'threshold': self.threshold,
            'buffer_utilization': buffer_stats['utilization'],
            'buffer_size': buffer_stats['current_size']
        }
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.reset()

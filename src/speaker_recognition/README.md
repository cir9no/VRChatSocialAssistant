# 说话人识别模块

基于声纹注册的说话人验证系统，用于从音频流中识别目标好友的声音。

## 功能特性

- ✅ 目标好友声纹注册与存储
- ✅ 实时声纹特征提取
- ✅ 声纹相似度匹配与验证
- ✅ 自适应阈值调整
- ✅ 持续学习更新声纹模型
- ✅ 多好友同时识别支持（1-10人）
- ✅ 本地化存储，保护隐私

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖：
- `torch>=2.0.0` - PyTorch 框架
- `torchaudio>=2.0.0` - 音频处理
- `pyannote.audio>=3.0.0` - 声纹提取模型（可选）
- `numpy>=1.24.0` - 数值计算
- `pyyaml>=6.0` - 配置文件解析

### 2. 基础使用

```python
from speaker_recognition import SpeakerRecognizer
import numpy as np

# 创建识别器
recognizer = SpeakerRecognizer()

# 注册好友声纹
friend_id = "friend_001"
audio_segments = [
    audio_data_1,  # numpy array, 2-5秒音频
    audio_data_2,
    audio_data_3,
]

success = recognizer.register_speaker(
    friend_id=friend_id,
    name="张三",
    audio_segments=audio_segments,
    sample_rate=16000
)

# 识别音频片段
audio_segment = ...  # 待识别的音频数据
result = recognizer.recognize(
    audio_segment=audio_segment,
    timestamp=1.0,
    sample_rate=16000
)

if result.matched:
    print(f"识别到: {result.speaker_id}")
    print(f"置信度: {result.confidence:.3f}")
else:
    print("未匹配到已注册说话人")
```

### 3. 运行演示

```bash
# 运行演示脚本
python tests/demo_speaker_recognition.py

# 运行单元测试
python tests/test_speaker_recognition.py
```

## 模块结构

```
src/speaker_recognition/
├── __init__.py                 # 模块导出
├── speaker_recognizer.py       # 说话人识别协调器
├── embedding_engine.py         # 声纹提取引擎
├── matching_engine.py          # 声纹匹配引擎
├── profile_database.py         # 声纹数据库
├── models.py                   # 数据模型定义
└── README.md                   # 本文档
```

## 核心组件

### SpeakerRecognizer（说话人识别协调器）

统一的模块入口，协调各子组件的交互。

**主要方法：**

| 方法 | 说明 |
|------|------|
| `register_speaker()` | 注册新的目标好友声纹 |
| `recognize()` | 识别音频片段的说话人 |
| `update_speaker_profile()` | 更新已注册声纹（持续学习） |
| `remove_speaker()` | 删除已注册声纹 |
| `get_registered_speakers()` | 获取所有已注册好友ID |
| `get_statistics()` | 获取识别统计信息 |

### EmbeddingEngine（声纹提取引擎）

从音频中提取192维声纹嵌入向量。

**技术方案：**
- 模型: SpeechBrain ECAPA-TDNN
- 特征维度: 192
- 采样率: 16kHz
- 推理设备: CPU/CUDA（自动检测）
- 模型源: HuggingFace (speechbrain/spkrec-ecapa-voxceleb)

**注意：** 需要安装 `speechbrain` 库：`pip install speechbrain`

### MatchingEngine（声纹匹配引擎）

计算声纹相似度并执行匹配决策。

**相似度方法：**
- 余弦相似度（推荐）
- 欧氏距离

**匹配策略：**
1. 计算测试嵌入与所有注册声纹的相似度
2. 检查最高相似度是否超过基础阈值
3. 检查最高与次高的差值是否足够大（避免歧义）
4. 返回匹配结果

### ProfileDatabase（声纹数据库）

存储和管理已注册的好友声纹数据。

**存储格式：**
- 元数据: JSON 文件 (`{friend_id}.json`)
- 嵌入向量: NPY 文件 (`{friend_id}.npy`)
- 存储位置: `data/speaker_profiles/`

## 配置

配置文件位于 `config/speaker_recognition_config.yaml`

```yaml
# 声纹提取模型配置
embedding:
  model_name: "speechbrain-ecapa-tdnn"
  model_path: "models/speaker_recognition/speechbrain/"
  device: "auto"  # auto, cpu, cuda
  auto_download: true
  download_source: "huggingface"
  
# 匹配配置
matching:
  base_threshold: 0.75        # 基础阈值
  difference_threshold: 0.10   # 差值阈值
  enable_adaptive_threshold: true  # 自适应调整
  
# 注册配置
registration:
  min_samples: 3              # 最少音频样本数
  min_sample_duration: 2.0    # 单段最短时长（秒）
```

### 阈值调优指南

| 阈值 | 误拒率 | 误识率 | 适用场景 |
|------|-------|-------|---------|
| 0.65 | 低 | 高 | 环境嘈杂，宁可识别错误 |
| 0.75 | 中 | 中 | 通用场景（推荐） |
| 0.85 | 高 | 低 | 安静环境，要求高准确性 |

## 性能指标

| 指标 | 目标值 | 说明 |
|------|-------|------|
| 识别延迟 | <50ms | 单次识别耗时 |
| 识别准确率 | >85% | 已注册好友识别准确率 |
| CPU占用率 | <10% | 持续识别时的CPU使用率 |
| 内存占用 | <500MB | 加载3个好友声纹后的内存 |

## 数据存储

```
data/
└── speaker_profiles/
    ├── {friend_id_1}.json       # 声纹元数据
    ├── {friend_id_1}.npy        # 声纹嵌入向量
    ├── {friend_id_2}.json
    ├── {friend_id_2}.npy
    └── ...
```

## 与其他模块集成

### 与VAD模块集成

```python
from vad import VADDetector
from speaker_recognition import SpeakerRecognizer

# 创建VAD和识别器
vad = VADDetector(sample_rate=16000)
recognizer = SpeakerRecognizer()

def on_speech_detected(segment, metadata):
    """VAD检测到语音时的回调"""
    # 识别说话人
    result = recognizer.recognize(
        audio_segment=segment,
        timestamp=metadata['start_time'],
        sample_rate=16000
    )
    
    if result.matched:
        print(f"检测到: {result.speaker_id}, 置信度: {result.confidence:.3f}")
        # 将识别结果传递给STT模块...

vad.set_callback(on_speech_detected)
```

## 注意事项

1. **首次运行：** 如果启用了 `auto_download`，模块会从 HuggingFace 下载 SpeechBrain ECAPA-TDNN 模型（约100MB）
2. **GPU加速：** 如果有CUDA可用，会自动使用GPU加速嵌入提取
3. **音频质量要求：**
   - 采样率: 16kHz
   - 时长: 0.5-10秒
   - 音量: 避免过小或过大
4. **注册要求：**
   - 最少3段音频样本
   - 每段2-5秒
   - 总时长10-20秒
5. **隐私保护：** 所有声纹数据存储在本地，不上传云端

## 故障排除

### 问题: 识别准确率低

**可能原因：**
- 注册样本质量不佳
- 环境噪声过大
- 阈值设置不合理

**解决方法：**
```python
# 调整阈值
recognizer.matching_engine.update_threshold(0.70)  # 降低阈值

# 重新注册（使用更多样本）
recognizer.register_speaker(
    friend_id=friend_id,
    name=name,
    audio_segments=[...],  # 提供5段音频
    sample_rate=16000
)
```

### 问题: 模型加载失败

**可能原因：**
- 未安装 pyannote.audio
- 模型文件损坏

**解决方法：**
```bash
# 安装 SpeechBrain
pip install speechbrain
```

### 问题: 未检测到任何匹配

**可能原因：**
- 没有已注册声纹
- 音频质量不符合要求

**解决方法：**
```python
# 检查已注册说话人
speakers = recognizer.get_registered_speakers()
print(f"已注册: {speakers}")

# 检查音频质量
valid, msg = recognizer.embedding_engine.validate_audio(audio, 16000)
print(f"音频有效性: {valid}, {msg}")
```

## 开发计划

- [x] Phase 1: 基础功能（完成）
  - [x] 声纹提取引擎
  - [x] 声纹匹配引擎
  - [x] 声纹数据库
  - [x] 说话人识别协调器
  
- [ ] Phase 2: 优化增强
  - [ ] 自适应阈值调整
  - [ ] 持续学习机制
  - [ ] 性能优化（批处理、缓存）
  
- [ ] Phase 3: 高级功能
  - [ ] 多人混说场景处理
  - [ ] 声纹加密存储
  - [ ] 实时性能监控

## 参考资料

- [pyannote.audio 官方文档](https://github.com/pyannote/pyannote-audio)
- [ECAPA-TDNN 论文](https://arxiv.org/abs/2005.07143)
- [Speaker Verification 综述](https://arxiv.org/abs/2010.12731)

## 许可证

本项目采用 MIT 许可证。

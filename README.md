# VRChat 社交辅助工具

一个基于AI的VRChat社交辅助系统，通过实时语音识别、声纹识别和大语言模型，为用户提供智能社交提示。

## 项目概述

本项目旨在帮助用户在VRChat社交场景中更好地交流，通过实时分析系统音频和麦克风输入，识别说话人并提供智能对话建议。

### 核心功能

- 🎤 **实时音频采集**：支持WASAPI Loopback采集系统音频和麦克风输入
- 🗣️ **语音活动检测（VAD）**：基于Silero模型的实时语音片段检测
- 👤 **说话人识别**：通过声纹识别区分不同说话人
- 📝 **语音转文本（STT）**：实时语音识别，支持中英文混合
- 🧠 **智能对话辅助**：基于LLM的上下文理解和建议生成
- 💾 **记忆管理**：RAG向量检索，记住好友信息和对话历史
- 🥽 **VR显示**：基于OpenXR的头显HUD提示展示

## 技术架构

### 技术栈

| 组件 | 技术选型 | 说明 |
|------|---------|------|
| 音频处理 | PyAudio + pyaudiowpatch | WASAPI Loopback支持 |
| 语音检测 | Silero VAD | 轻量级、高精度VAD模型 |
| 语音识别 | faster-whisper | 本地部署的高效STT |
| 声纹识别 | pyannote.audio | ECAPA-TDNN声纹模型 |
| 向量数据库 | Chroma | 记忆存储和检索 |
| 大语言模型 | OpenAI API / 本地模型 | 对话理解和建议生成 |
| VR渲染 | OpenXR + pyopenvr | 跨平台VR显示 |

### 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    VR 头显 HUD 显示                      │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│              提示生成 & 智能建议模块                      │
│  - LLM推理引擎  - 记忆检索  - 提示格式化                 │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│                   语音处理模块                           │
│  - VAD检测  - 说话人识别  - 流式STT                      │
└─────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────┐
│                   音频采集模块                           │
│  - WASAPI Loopback  - 麦克风输入  - 音频预处理           │
└─────────────────────────────────────────────────────────┘
```

## 项目进度

### ✅ 已完成

#### 1. 音频采集模块 (完成度: 100%)
- ✅ `DeviceManager`：音频设备管理和枚举
- ✅ `AudioCapturer`：双通道音频采集（Loopback + 麦克风）
- ✅ 自动重采样功能（48kHz → 16kHz）
- ✅ 回调机制和队列管理
- ✅ 完整的单元测试和集成测试

#### 2. VAD（语音活动检测）模块 (完成度: 100%)
- ✅ `SileroVAD`：基于Silero模型的VAD检测
- ✅ `AudioBuffer`：音频缓冲区管理
- ✅ `VADDetector`：语音片段检测和切分
- ✅ 状态机管理（IDLE → SPEECH → SILENCE）
- ✅ 语音片段输出回调
- ✅ 性能优化（处理延迟 <2ms，零丢帧）
- ✅ 完整的单元测试（18个测试用例全部通过）
- ✅ 与音频采集模块的集成测试

#### 3. 配置管理
- ✅ YAML配置文件支持
- ✅ 音频参数配置
- ✅ VAD参数配置

#### 4. 项目基础设施
- ✅ 项目结构设计
- ✅ 依赖管理（requirements.txt）
- ✅ Git版本控制
- ✅ 开发环境配置

### 🚧 进行中

目前所有已开发模块均已完成并通过测试。

### 📋 待开发

#### 1. 说话人识别模块 (优先级: 高)
- ⏳ 声纹特征提取
- ⏳ 目标说话人注册
- ⏳ 实时声纹匹配
- ⏳ 多说话人管理

#### 2. 语音转文本（STT）模块 (优先级: 高)
- ⏳ faster-whisper集成
- ⏳ 流式识别支持
- ⏳ 中英文混合识别
- ⏳ 实时字幕输出

#### 3. 智能对话辅助模块 (优先级: 中)
- ⏳ LLM集成（OpenAI API / 本地模型）
- ⏳ 上下文管理
- ⏳ 对话理解和分析
- ⏳ 建议生成策略

#### 4. 记忆管理模块 (优先级: 中)
- ⏳ 向量数据库集成（Chroma）
- ⏳ 好友档案管理
- ⏳ 对话历史记录
- ⏳ RAG检索增强

#### 5. VR显示模块 (优先级: 低)
- ⏳ OpenXR集成
- ⏳ HUD Overlay渲染
- ⏳ 手柄交互
- ⏳ 视觉优化

#### 6. 系统集成 (优先级: 低)
- ⏳ 主控制流程
- ⏳ 模块间通信
- ⏳ 错误处理和恢复
- ⏳ 性能监控

## 快速开始

### 环境要求

- Python 3.8+
- Windows 10/11（WASAPI支持）
- 支持CUDA的GPU（可选，用于加速）

### 安装依赖

```bash
pip install -r requirements.txt
```

### 模型文件

项目使用的模型文件会在首次运行时自动下载到 `models/` 目录下，无需手动下载。

**VAD模型**: 首次运行时会从 PyTorch Hub 自动下载 Silero VAD 模型（约1-2MB）到 `models/vad/` 目录。

### 运行测试

#### 测试音频采集
```bash
python tests/test_audio_capture.py
```

#### 测试VAD模块
```bash
python tests/test_vad.py
```

#### 测试集成功能
```bash
python tests/test_vad_integration.py
```

#### VAD功能演示
```bash
python tests/demo_vad.py
```

## 项目结构

```
VRChatSocialAssistant/
├── config/                 # 配置文件
│   ├── audio_config.yaml  # 音频和VAD配置
│   └── memory_config.yaml # 记忆模块配置
├── doc/                   # 文档
│   └── 架构设计.md        # 架构设计文档
├── models/                # 模型文件目录
│   └── vad/              # VAD模型存储目录
├── src/                   # 源代码
│   ├── audio_capture/     # 音频采集模块
│   │   ├── device_manager.py
│   │   └── audio_capturer.py
│   ├── memory/           # 记忆管理模块
│   └── vad/              # VAD模块
│       ├── audio_buffer.py
│       ├── silero_vad.py
│       └── vad_detector.py
├── tests/                # 测试代码
│   ├── test_audio_capture.py
│   ├── test_vad.py
│   ├── test_vad_integration.py
│   └── demo_vad.py
└── requirements.txt      # 项目依赖
```

## 测试结果

### VAD模块测试
- ✅ 单元测试：18/18 通过
- ✅ 集成测试：成功
- ✅ 性能指标：
  - 处理延迟：1.48ms（目标 <30ms）
  - 丢帧率：0%
  - 准确率：成功检测语音片段，置信度 0.798

### 音频采集测试
- ✅ WASAPI Loopback采集正常
- ✅ 自动重采样功能正常（48kHz → 16kHz）
- ✅ 零溢出、零丢帧

## 开发计划

### 近期目标（1-2周）
1. 实现说话人识别模块
2. 集成faster-whisper STT
3. 完成音频处理全链路测试

### 中期目标（1-2月）
1. LLM集成和对话理解
2. 记忆管理系统
3. 基础UI原型

### 长期目标（3-6月）
1. VR头显集成
2. 完整系统集成测试
3. 性能优化和稳定性改进

## 贡献指南

欢迎提交Issue和Pull Request！

## 许可证

本项目采用MIT许可证，详见 [LICENSE](LICENSE) 文件。

## 致谢

- [Silero VAD](https://github.com/snakers4/silero-vad) - 高性能VAD模型
- [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch) - WASAPI Loopback支持
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - 高效语音识别

---

**注意**：本项目仍在积极开发中，API可能会发生变化。

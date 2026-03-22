# Brick by Brick · AI Focus Detection Training System

> AI 专注度检测模型训练系统 | 独立于主项目维护  
> 最终与 [brick-by-brick](https://github.com/jojojjjjj/brick-by-brick) 主项目融合

## 项目概述

一砖一瓦 (Brick by Brick) 专注度检测模型训练系统。通过前置摄像头实时检测用户专注度，输出 0-100 专注度分数，驱动虚拟建筑状态变化。

## 核心能力

- **输入**：前置摄像头视频帧（5fps）+ 键盘鼠标活动日志 + 场景 ID
- **输出**：每 5 秒一个 0-100 专注度分数
- **场景支持**：写代码 / 阅读 / 看视频（统一模型 + 场景 embedding）
- **跨平台部署**：华为 NPU / 苹果 ANE / 英特尔 NPU / 高通 Hexagon

## 硬件要求

- **训练**：RTX 5080（16GB GDDR7，Blackwell）— 严重过剩
- **推荐训练规模**：1-10M 参数模型，30 分钟内完成训练
- **推理**：PC (< 20ms) / 手机 (< 80ms)

## 项目结构

```
brick-by-brick-ai-training/
├── docs/
│   ├── AI_TRAINING_SYSTEM_GUIDE.md   # 完整训练系统指导（本文档详细版）
│   └── FOCUS_DETECTION_FEASIBILITY_REPORT.md  # 可行性论证报告
├── src/
│   ├── data/                        # 数据采集与处理
│   ├── models/                      # 模型定义
│   ├── training/                    # 训练脚本
│   ├── annotation/                  # 自动标注流水线
│   └── deployment/                  # 跨 NPU 部署
├── configs/                         # 配置文件
├── scripts/                        # 工具脚本
├── data/                           # 训练数据（gitignore）
├── checkpoints/                    # 模型权重（gitignore）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境安装

```bash
conda create -n focus-ai python=3.10 -y
conda activate focus-ai

# PyTorch + CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install mediapipe mmaction2 onnxruntime-gpu opencv-python
pip install onnx pyyaml pandas numpy pillow
pip install label-studio-sdk yt-dlp
pip install tqdm scipy hmmlearn filterpy

# 开发依赖
pip install jupyter pytest black flake8
```

### 2. 数据采集

```bash
# 自我录制（视频 + 屏幕 + 键鼠同步）
python scripts/record_self.py --output data/session_001 --duration 3600

# YouTube 学习视频下载
python scripts/download_youtube.py --url "https://youtube.com/watch?v=..." --output data/youtube/
```

### 3. 自动标注

```bash
python -m src.annotation.auto_labeler \
    --video data/session_001/video.mp4 \
    --output data/session_001/ \
    --device cuda
```

### 4. 训练

```bash
python -m src.training.trainer \
    --config configs/training.yaml \
    --data-root data/ \
    --output checkpoints/
```

### 5. ONNX 导出

```bash
python -m src.models.ONNX_export \
    --checkpoint checkpoints/focus_model_best.pth \
    --output models/focus_model.onnx
```

### 6. 跨 NPU 推理

```python
from src.deployment.onnx_inference import ONNXFocusDetector

detector = ONNXFocusDetector("models/focus_model.onnx", providers=["CUDAExecutionProvider"])
scores = detector.detect(frames, gaze_pose, scene_id=0)  # 0=写代码
print(f"专注度: {scores}")
```

## 模型架构

**MobileNetV3-Small + 2层 GRU + 场景 Embedding**

- 参数量：~1.5M
- 输入：视频帧 (T×3×224×224) + gaze/pose (T×7) + 场景 ID
- 输出：T×0-100 专注度分数
- 训练策略：预训练（CASIA-WebFace）+ Fine-tuning

详见 [AI_TRAINING_SYSTEM_GUIDE.md](docs/AI_TRAINING_SYSTEM_GUIDE.md)

## 三阶段路线图

| 阶段 | 时间 | 目标 | 准确率 |
|------|------|------|--------|
| MVP | 1-2 周 | MediaPipe + 规则评分 | 60-70% |
| V1 | 2-4 周 | MobileNetV3 + GRU + 自动标注 | 85-90% |
| V2 | 持续 | EfficientNet + Transformer | 90-95% |

## 与主项目融合

| 主项目阶段 | AI 训练项目状态 | 接入方式 |
|-----------|---------------|---------|
| MVP P0 | 独立验证 | MockFocusDetector |
| MVP P1 | 独立 GitHub | NCNN/MediaPipe 接入 |
| 正式版 | 集成 | ArkGraphics 3D + 真实专注度 |

## 关键文档

- [可行性论证报告](docs/FOCUS_DETECTION_FEASIBILITY_REPORT.md) — 详细技术论证
- [训练系统指导](docs/AI_TRAINING_SYSTEM_GUIDE.md) — Codex 可执行的完整代码指南

## 团队

- 3 人小团队
- 核心约束：无法人工标注 1000+ 样本，必须 AI 辅助 + 主动学习

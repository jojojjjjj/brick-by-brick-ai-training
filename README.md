# Brick by Brick · AI Focus Detection Training System

> AI 专注度检测模型训练系统 | **移动端纯视觉方案**  
> 独立于主项目维护 | 最终与 [brick-by-brick](https://github.com/jojojjjjj/brick-by-brick) 主项目融合

## ⚠️ 重要说明

**这是手机端纯视觉方案。**

- ❌ 无键盘鼠标日志
- ❌ 无 PC 端推理
- ✅ 只有前置摄像头 + MediaPipe Face Mesh
- ✅ 推理在手机本地完成（ncnn + HarmonyOS NPU）
- ✅ RTX 5080 用于**训练**（开发机），模型部署在**手机端**

---

## 核心约束

| 约束 | 说明 |
|------|------|
| **平台** | HarmonyOS NEXT 手机应用 |
| **输入** | 前置摄像头视频帧（5fps） |
| **输出** | 每 5 秒一个 0-100 专注度分数 |
| **推理** | 手机本地（ncnn INT8，不上传云端） |
| **训练** | RTX 5080（开发机），FP16 混合精度 |
| **团队** | 3 人，无法人工标注 1000+ 样本 |

---

## 产品背景

一砖一瓦 (Brick by Brick) 是 HarmonyOS 番茄钟应用。通过前置摄像头实时检测用户专注度，在虚拟城市中建造建筑——专注质量越高，建筑越精美；分心则建筑破损。

- **CH3 AI 专注检测系统**（Handbook）：https://github.com/jojojjjjj/brickbybrick-handbook
- **产品手册**：https://jojojjjjj.github.io/brickbybrick-handbook

---

## 项目结构

```
brick-by-brick-ai-training/
├── docs/
│   ├── AI_TRAINING_SYSTEM_GUIDE.md   # Codex 训练系统指导
│   └── FOCUS_DETECTION_FEASIBILITY_REPORT.md  # 本报告
├── src/
│   ├── data/
│   │   ├── video_processor.py      # FFmpeg 抽帧（手机录制视频）
│   │   ├── face_extractor.py        # MediaPipe Face Mesh（手机端核心）
│   │   └── label_fusion.py          # 多信号融合 + HMM 时序平滑
│   ├── models/
│   │   ├── backbone.py              # MobileNetV3 backbone
│   │   ├── focus_head.py             # 专注度输出头（YOLOv5n/MobileFaceNet）
│   │   └── ONNX_export.py           # ONNX 导出（PyTorch → ONNX）
│   ├── training/
│   │   ├── dataset.py               # FocusDataset（手机自拍数据）
│   │   ├── trainer.py               # FP16 混合精度训练
│   │   ├── active_learning.py        # 主动学习（MediaPipe 自动标注）
│   │   └── quantize.py              # ncnn INT8 量化
│   ├── annotation/
│   │   └── auto_labeler.py         # 自动标注流水线
│   └── deployment/
│       ├── base.py                  # 统一推理接口
│       └── ncnn_export.py           # ONNX → ncnn 转换
├── configs/
│   └── training.yaml
├── scripts/
│   └── adb_record.py                # adb 手机视频录制
├── data/                            # 训练数据（gitignore）
├── checkpoints/                     # 模型权重（gitignore）
├── requirements.txt
└── README.md
```

---

## 推理链路

```
RTX 5080 训练（FP16）
    ↓ PyTorch .pth
ONNX 导出
    ↓
ONNX 模型（跨平台）
    ↓ ncnn 量化工具
ncnn INT8 模型（手机端）
    ↓ NAPI C++ 封装
HarmonyOS NEXT 手机
```

---

## 快速开始

### 1. 环境安装

```bash
conda create -n focus-ai python=3.10 -y
conda activate focus-ai

# PyTorch + CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 核心依赖
pip install mediapipe onnxruntime-gpu opencv-python
pip install pyyaml pandas numpy pillow
pip install tqdm scipy hmmlearn filterpy

# ncnn 工具（用于量化）
# https://github.com/Tencent/ncnn/releases
```

### 2. 手机视频采集

```bash
# 使用 adb 录制手机前置摄像头
adb shell screenrecord --time-limit 3600 /sdcard/focus_data.mp4
adb pull /sdcard/focus_data.mp4 ./

# 或使用 Python 脚本（需手机开启 USB 调试）
python scripts/adb_record.py --output data/session_001 --duration 3600
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

### 5. ONNX 导出 → ncnn 量化 → HarmonyOS 部署

```bash
# Step 1: PyTorch → ONNX
python -m src.models.ONNX_export \
    --checkpoint checkpoints/focus_model.pth \
    --output models/focus_model.onnx

# Step 2: ONNX → ncnn（使用 ncnn 官方工具）
# 下载 ncnn: https://github.com/Tencent/ncnn/releases
# ./ncnnoptimize focus_model.onnx focus_model.param focus_model.bin

# Step 3: ncnn INT8 量化
# ./ncnnquantize focus_model.param focus_model.bin \
#              focus_model_int8.param focus_model_int8.bin
```

---

## 准确率目标（统一技术架构）

**核心原则：同一模型架构，不同数据量。**

| 阶段 | 准确率 | 数据量 | 方法 |
|------|--------|--------|------|
| P0 MVP | 60-70% | 0 张 | MediaPipe Face Mesh（预训练权重，直接用） |
| P1 | 75-85% | 2000 张自拍 | **同一模型** Fine-tune（RTX 5080）|
| P2 | 85-93% | 5000+ 张自拍 | **同一模型** 继续训练 + 个人校准 |

---

## 三阶段路线图（统一架构）

| 阶段 | 时间 | 训练数据 | 技术 |
|------|------|---------|------|
| P0 MVP | 1-2 周 | 0 张（不开训练） | MediaPipe Face Mesh 预训练权重 |
| P1 训练版 | 2-4 周 | 2000 张自拍 | **同一模型** Fine-tune（RTX 5080）|
| P2 精调版 | 持续 | 5000+ 张自拍 | **同一模型** 继续训练 + 用户个性化 |

---

## ⚠️ 重要：数据法律风险

**所有主流公开眼动数据集均禁止商业使用：**

| 数据集 | 许可证 | 能否商用 |
|--------|--------|---------|
| GazeCapture (Apple) | Research License | ❌ 明确禁止任何商业应用 |
| MRL Eye Dataset | CC BY-NC-SA 4.0 | ❌ NC 条款禁止 |
| MPIIGaze | 研究协议 | ❌ 禁止商业使用 |
| Columbia Gaze | Research Use Only | ❌ 禁止商业使用 |
| UnityEyes（合成） | MIT License | ✅ 可商用 |

**唯一安全路径：完全自我录制。** 华为有内置眼球追踪接口（Camera Kit），无需训练 gaze 估计模型，只需用自拍数据训练专注度判断模型。

---

## 关键文档

- [可行性论证报告](docs/FOCUS_DETECTION_FEASIBILITY_REPORT.md) — 详细技术论证（移动端版）
- [训练系统指导](docs/AI_TRAINING_SYSTEM_GUIDE.md) — Codex 可执行的完整代码指南

---

## 与主项目融合

| 主项目阶段 | AI 训练项目 | 接入方式 |
|-----------|------------|---------|
| MVP P0 | 不需要 | MediaPipe ArkTS bindings 直接用 |
| MVP P1 | 独立训练 | ncnn 模型通过 NAPI 接入 |
| 正式版 | 集成 | ArkGraphics 3D + 真实专注度检测 |

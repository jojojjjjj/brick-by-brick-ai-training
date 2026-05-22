# Brick by Brick · AI Focus Detection Training System
# 逐块砌砖 · AI 专注度检测训练系统

> AI 专注度检测模型训练系统 | **移动端纯视觉方案**  
> 独立于主项目维护 | 最终与 [brick-by-brick](https://github.com/jojojjjjj/brick-by-brick) 主项目融合

---

## ⚠️ 重要更新 2026-05-20

**基于全新Transformer架构调研的方案更新：**

- ❌ ~~MobileNetV3 + ncnn~~ → ✅ **ViT-S/14 + DINOv2 + ONNX Runtime**
- ❌ ~~单模型VLM标注~~ → ✅ **多教师蒸馏（DINOv2 + MediaPipe + LLaVA）**
- ✅ 保留：前置摄像头纯视觉方案
- ✅ 保留：RTX 5080 训练，手机端推理

---

## 核心约束

| 约束 | 说明 |
|------|------|
| **平台** | HarmonyOS NEXT / Android / iOS |
| **输入** | 前置摄像头视频帧（5-15fps） |
| **输出** | 每 5 秒一个 0-100 专注度分数 |
| **推理** | 手机本地（ONNX Runtime NPU加速） |
| **模型** | FocusNet-Lite (~35M参数，<100MB) |
| **精度目标** | <5°角度误差 |

---

## 产品背景

一砖一瓦 (Brick by Brick) 是 HarmonyOS 番茄钟应用。通过前置摄像头实时检测用户专注度，在虚拟城市中建造建筑——专注质量越高，建筑越精美；分心则建筑破损。

---

## 技术架构 (基于最新调研)

### FocusNet-Lite 学生模型

```
FocusNet-Lite (~35M参数)
├── 视觉编码器: ViT-S/14 (22M参数)
│   └── DINOv2预训练特征
├── 头部姿态编码器: MLP (2M参数)
├── 交叉注意力融合: 4层 (8M参数)
│   └── 视线+头部姿态双向交互
└── 视线头: MLP (2M参数)
    └── 输出: pitch, yaw角度
```

### 多教师蒸馏策略

```
教师集成 → 学生模型 FocusNet-Lite
├── DINOv2-L/14 (304M) → 特征蒸馏
├── MediaPipe Face Mesh (2.1M) → Landmark蒸馏
├── Swin-Face → 注意力蒸馏
└── LLaVA-7B (7B) → 语义蒸馏
```

### 部署链路

```
RTX 5080 训练 (FP16 AMP)
    ↓
PyTorch模型 (.pth)
    ↓
ONNX导出 (opset 17)
    ↓
FP16/INT8量化
    ↓
ONNX Runtime
    ├── iOS: CoreML EP (ANE)
    ├── Android: QNN EP (Qualcomm Hexagon)
    └── HarmonyOS: CANN EP (Huawei Ascend)
```

---

## 项目结构

```
brick-by-brick-ai-training/
├── docs/
│   ├── 01_transformer_gaze_models/     # Transformer视线估计调研
│   ├── 02_head_pose_estimation/        # 头部姿态估计调研
│   ├── 03_knowledge_distillation/      # 知识蒸馏技术调研
│   ├── 04_mobile_deployment/           # 移动端NPU部署调研
│   ├── 05_datasets/                    # 数据集调研
│   ├── 06_attention_fusion/            # 注意力融合策略调研
│   ├── 07_teacher_models/              # 教师模型调研
│   └── AI_TRAINING_SYSTEM_GUIDE.md    # 训练系统指导
├── src/
│   ├── data/                          # 数据处理
│   ├── models/                        # 模型定义
│   ├── training/                      # 训练代码
│   └── deployment/                    # 部署代码
├── FOCUS_MODEL_RESEARCH_REPORT.md     # 英文研究报告
├── FOCUS_MODEL_RESEARCH_REPORT_CN.md  # 中文研究报告
├── FOCUS_MODEL_RESEARCH_REPORT_BILINGUAL.md  # 中英对照报告
└── README.md
```

---

## 准确率目标

| 阶段 | 准确率 | 数据量 | 方法 |
|------|--------|--------|------|
| P0 MVP | 60-70% | 0张 | MediaPipe预训练 |
| P1 | 75-85% | 2000张 | FocusNet-Lite微调 |
| P2 | 85-93% | 5000+张 | 多教师蒸馏+个人校准 |

---

## 路线图

| 阶段 | 周期 | 任务 |
|------|------|------|
| Phase 1 研究基础 | 1-4周 | 文献综述（✓完成） |
| Phase 2 教师选择 | 5-8周 | DINOv2、MediaPipe、LLaVA评估 |
| Phase 3 学生训练 | 9-16周 | 3阶段蒸馏训练 |
| Phase 4 量化优化 | 17-20周 | FP16/INT8量化、NPU优化 |
| Phase 5 移动端部署 | 21-24周 | iOS/Android/HarmonyOS集成 |

---

## 关键文档

- [英文研究报告](FOCUS_MODEL_RESEARCH_REPORT.md)
- [中文研究报告](FOCUS_MODEL_RESEARCH_REPORT_CN.md)
- [中英对照报告](FOCUS_MODEL_RESEARCH_REPORT_BILINGUAL.md)
- [训练系统指导](docs/AI_TRAINING_SYSTEM_GUIDE.md)

---

## 与主项目融合

| 主项目阶段 | AI训练项目 | 接入方式 |
|-----------|-----------|---------|
| MVP P0 | 不需要 | MediaPipe直用 |
| MVP P1 | 独立训练 | ONNX Runtime接入 |
| 正式版 | 集成 | FocusNet-Lite NPU加速 |

---

## 参考文献

完整参考文献请查看 [FOCUS_MODEL_RESEARCH_REPORT.md](FOCUS_MODEL_RESEARCH_REPORT.md)

主要引用:
1. Song et al. (2024). ViTGaze. *Visual Intelligence*
2. Zhang et al. (2020). ETH-XGaze. *ECCV 2020*
3. Hinton et al. (2015). Distilling the Knowledge. *NIPS Workshop*
4. Oquab et al. (2023). DINOv2. *arXiv:2304.07193*
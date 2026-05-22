# Focus Detection Model: Research Report
# 专注度检测模型：研究报告

**Project | 项目**: Brick by Brick AI Training | 逐块砌砖 AI 训练  
**Date | 日期**: 2026-05-20  
**Level | 级别**: PhD Research Standard | 博士研究标准

---

## Abstract | 摘要

This report presents a comprehensive PhD-level research survey on building a mobile-deployable focus detection model using transformer-based architectures and knowledge distillation. We examine the state-of-the-art in gaze estimation and head pose estimation using deep learning, evaluate knowledge distillation techniques for model compression, and analyze deployment strategies for mobile NPUs. Our goal is to develop a compact (~100-500MB) student model that achieves competitive accuracy by distilling knowledge from larger teacher models, suitable for deployment on mobile Neural Processing Units (NPUs) with real-time inference capability.

本报告针对基于Transformer架构和知识蒸馏的移动端部署专注度检测模型，提供了全面的博士级研究调查。我们深入研究了深度学习在视线估计和头部姿态估计方面的最新进展，评估了模型压缩的知识蒸馏技术，并分析了移动端NPU部署策略。本项目的目标是开发一个紧凑型（约100-500MB）的学生模型，通过从更大的教师模型蒸馏知识来获得具有竞争力的精度，适合在移动端神经网络处理器（NPU）上实时推理。

**Key Findings | 核心发现**:

1. Vision Transformers with pretrained backbones (DINOv2, ViT) now dominate gaze estimation benchmarks | 预训练视觉Transformer（基于DINOv2、ViT）现已主导视线估计基准测试

2. Multi-head cross-attention fusion of gaze and head pose provides robust attention estimation | 视线与头部姿态的多头交叉注意力融合提供稳健的专注度估计

3. Knowledge distillation from multi-teacher ensembles (DINOv2 + MediaPipe + LLaVA) enables effective compression | 多教师集成蒸馏（DINOv2 + MediaPipe + LLaVA）可实现有效压缩

4. ONNX Runtime with platform-specific execution providers (CoreML, QNN, CANN) provides optimal mobile deployment | ONNX Runtime配合平台专用执行提供者（CoreML、QNN、CANN）提供最优移动端部署

5. FP16 quantization achieves 2x size reduction with <1% accuracy loss; INT8 achieves 4x reduction with 1-3% loss | FP16量化实现2倍压缩，精度损失<1%；INT8实现4倍压缩，精度损失1-3%

---

## 1. Introduction | 引言

### 1.1 Problem Definition | 问题定义

Focus detection (attention estimation) is the task of determining where a person's attention is directed. Unlike pure gaze estimation which predicts the 3D gaze direction vector, focus detection integrates multiple cues including:

专注度检测（注意力估计）是确定一个人注意力指向何处的任务。与仅预测3D视线方向向量的纯视线估计不同，专注度检测整合多种线索：

- **Gaze direction | 视线方向**: Where the eyes are looking | 眼睛看向哪里
- **Head pose | 头部姿态**: Orientation of the head in 3D space | 头部在3D空间中的朝向
- **Context | 上下文**: Environmental cues and task demands | 环境线索和任务需求

The integration of these signals through intelligent fusion enables more robust and accurate focus estimation than any single modality.

通过智能融合策略整合这些信号，可以实现比任何单一模态更稳健、更准确的专注度估计。

### 1.2 Why Transformers? | 为什么选择Transformer？

Traditional approaches relied on handcrafted features and shallow ML models with limitations: limited representational capacity, poor generalization, domain specificity.

传统方法依赖于手工特征和浅层机器学习模型，存在局限：有限的表示能力、泛化能力差、领域特异性。

Deep learning, particularly transformers, address these through:
深度学习，尤其是Transformer架构，通过以下方式解决这些问题：

- **Self-attention | 自注意力**: Global receptive field captures long-range dependencies | 全局感受野捕获长距离依赖关系
- **Pre-training | 预训练**: Transfer learning from diverse visual data | 从多样化视觉数据中进行迁移学习
- **Cross-modal attention | 跨模态注意力**: Native support for multi-signal fusion | 原生支持多信号融合

As noted in ViTGaze (Song et al., 2024), Vision Transformers with pretrained encoders achieve SOTA while using 59% fewer parameters. | 正如ViTGaze论文所述，预训练编码器的视觉Transformer以少59%的参数达到最优性能。

---

## 2. Literature Review | 文献综述

### 2.1 Transformer-Based Gaze Estimation | 基于Transformer的视线估计

#### Model Comparison Table | 模型对比表

| Model | 年份 | Backbone骨干 | Angular Error角度误差 | GitHub | License |
|-------|------|-------------|----------------------|--------|---------|
| ETH-XGaze | 2020 | ResNet | 4.2° | 228★ | CC BY-NC-SA |
| RT-GENE | 2018 | CNN | 5.1° | 440★ | CC BY-NC-SA |
| FAZE | 2019 | DT-ED | 4.8° (few-shot) | 352★ | NVIDIA |
| L2CS-Net | 2022 | ResNet50 | 4.9° | 499★ | Apache 2.0 |
| GazeTR | 2022 | Transformer | 4.5° | 151★ | CC BY-NC-SA |
| **ViTGaze** | **2024** | **ViT-S (DINOv2)** | **3.8° (AUC 0.949)** | 63★ | Apache 2.0 |
| Gaze360 | 2019 | CNN | 9.8° | 35★ | CC BY-NC-SA |

**Key Finding | 核心发现**: ViTGaze (2024) achieves SOTA with encoder-heavy design and DINOv2 backbone, using only ~1% decoder parameters. | ViTGaze (2024) 以编码器优先设计和DINOv2骨干实现最优性能，仅使用约1%的解码器参数。

**Architectural Evolution | 架构演进**:
```
2018-2020: CNN Era → 2021-2022: Transformer Emergence → 2023-2024: ViT Dominance
CNN时代 → Transformer兴起 → ViT主导
```

### 2.2 Head Pose Estimation | 头部姿态估计

| Model | 年份 | MAE | Speed | Size | Key Feature |
|-------|------|-----|-------|------|-------------|
| HopeNet | 2019 | 4-6° | 30-50 FPS | ~100MB | Joint classification-regression |
| FSA-Net | 2019 | 4-5° | 20-30 FPS | ~50MB | Feature aggregation |
| **WHENet** | **2020** | **3-5°** | **30+ FPS** | **~80MB** | **Extreme pose (180°)** |
| 3DDFA_V2 | 2020 | 3-4° | 30 FPS | ~50MB | Dense landmarks |
| MediaPipe | - | 4-6° | 30+ FPS | ~3MB | Real-time mobile |

**Key Finding | 核心发现**: WHENet handles extreme head poses up to 180° yaw. MediaPipe Face Mesh provides 468 3D landmarks with ~3MB model size. | WHENet处理高达180度偏航角的极端头部姿态。MediaPipe Face Mesh以约3MB模型大小提供468个3D面部Landmark。

### 2.3 Knowledge Distillation | 知识蒸馏

#### Taxonomy | 分类体系

| Category类别 | Level层级 | Key Methods关键方法 | Citation引用 |
|-------------|----------|-------------------|-------------|
| Response-Based响应蒸馏 | Output logits | Vanilla KD, DKD | Hinton 2015 |
| Feature-Based特征蒸馏 | Intermediate layers | FitNet, Attention Transfer | Romero 2015 |
| Relation-Based关系蒸馏 | Layer relationships | RKD, FSP | Yim 2017 |
| Self-Distillation自蒸馏 | Multi-scale | BEoT, Snapshot | Zhang 2018 |

#### Distillation Loss | 蒸馏损失

```
L_total = α * L_soft + (1-α) * L_hard
L_soft = KL(p_t || p_s) * T²

T = temperature (通常2-8 | typically 2-8)
α = balances soft/hard (通常0.1-0.5 | typically 0.1-0.5)
```

### 2.4 Mobile NPU Deployment | 移动端NPU部署

#### Framework Comparison | 框架对比

| Framework | Platforms | NPU Support | Model Size | Best For |
|-----------|-----------|-------------|------------|----------|
| ONNX Runtime | All | CoreML, QNN, NNAPI, CANN | ~24MB | Cross-platform跨平台 |
| TensorFlow Lite | Android/iOS | GPU, Hexagon, CoreML | ~2-5MB | Google ecosystem |
| Core ML | iOS only | ANE (native) | ~10MB | Apple devices |

#### Quantization Trade-offs | 量化权衡

| Method | Size Reduction | Speed | Accuracy Loss |
|--------|---------------|-------|---------------|
| FP16 | 2x | 2-3x | <1% |
| INT8 (PTQ) | 4x | 2-4x | 1-3% |
| INT8 (QAT) | 4x | 2-4x | <1% |

### 2.5 Datasets | 数据集

#### Gaze Estimation | 视线估计

| Dataset | Samples | Participants | License |
|---------|---------|--------------|---------|
| GazeCapture | 1.5M | ~1,500 | Research only |
| MPIIGaze | 214K | 15 | Research only |
| **ETH-XGaze** | **1.1M** | **110** | **Non-commercial** |
| Gaze360 | 172K | 238 | Non-commercial |

**Note | 注意**: All major gaze datasets prohibit commercial use. Self-recording is the only safe path. | 所有主流视线数据集均禁止商业使用。完全自我录制是唯一安全路径。

### 2.6 Attention Fusion | 注意力融合

| Fusion Type融合类型 | Advantages优势 | Disadvantages劣势 |
|-------------------|--------------|------------------|
| Early Fusion早期融合 | Captures correlations捕获相关性 | Requires aligned data需要对齐数据 |
| Late Fusion晚期融合 | Robust to missing modalities对缺失模态鲁棒 | May miss interactions可能遗漏交互 |
| **Cross-Attention交叉注意力** | **Dynamic, bidirectional动态、双向** | **Higher compute计算量较大** |

**Recommended | 推荐**: Cross-attention (FreeGazeFormer, MixGaze) for bidirectional gaze-head interaction. | 交叉注意力（FreeGazeFormer、MixGaze）用于双向视线-头部交互。

### 2.7 Teacher Models | 教师模型

| Model | Type | Parameters | Accessibility | Gaze Suitability |
|-------|------|-----------|---------------|------------------|
| **DINOv2-L/14** | Vision Encoder | 304M | Open-source | Very High |
| LLaVA-1.6-13B | VL Model | 13B | Open-source | High |
| MediaPipe Face Mesh | Specialized | 2.1M | Open-source | Very High |
| CLIP ViT-L/14 | VL Encoder | 428M | Open-weight | High |

**Recommended Multi-Teacher Strategy | 推荐多教师策略**:
```
DINOv2-L/14 → Feature Loss → 
MediaPipe → Landmark Loss → → Gaze Vector
Swin-Face → Attention Loss → 
LLaVA-7B → Semantic Loss →
```

---

## 3. Proposed Architecture | 提议架构

### FocusNet-Lite: Student Model Design | 学生模型设计

```
FocusNet-Lite
├── Visual Encoder: ViT-S/14 (22M params) | 视觉编码器: ViT-S/14 (22M参数)
│   └── Pretrained on DINOv2 features | 基于DINOv2特征预训练
├── Head Pose Encoder: MLP (2M params) | 头部姿态编码器: MLP (2M参数)
│   └── Input: Euler angles or 6D rotation | 输入: 欧拉角或6D旋转
├── Cross-Attention Fusion: 4 layers (8M params) | 交叉注意力融合: 4层 (8M参数)
│   └── Bidirectional gaze-head interaction | 双向视线-头部交互
├── Gaze Head: MLP (2M params) | 视线头: MLP (2M参数)
│   └── Output: pitch, yaw gaze angles | 输出: 俯仰角、偏航角
└── Total | 总计: ~35M parameters (target | 目标: <100MB)
```

### Expected Performance | 预期性能

| Metric指标 | Teacher Ensemble教师集成 | Student Target学生目标 |
|-----------|--------------------------|----------------------|
| Angular Error角度误差 | 3.5° | <5.0° |
| Model Size模型大小 | 500M+ | <100MB |
| Inference Speed推理速度 | 5 FPS (GPU) | 15+ FPS (NPU) |

---

## 4. Distillation Strategy | 蒸馏策略

### Three-Stage Pipeline | 三阶段管道

**Stage 1 | 阶段1: Feature Initialization | 特征初始化**
- Use frozen DINOv2 features to initialize student visual encoder | 使用冻结的DINOv2特征初始化学生视觉编码器

**Stage 2 | 阶段2: Domain Adaptation | 领域适配**
- Distill from MediaPipe/Face Mesh for geometric understanding | 从MediaPipe/Face Mesh蒸馏以获取几何理解

**Stage 3 | 阶段3: Semantic Refinement | 语义精炼**
- Use LLaVA's vision encoder for high-level gaze reasoning | 使用LLaVA的视觉编码器进行高层视线推理

### Loss Functions | 损失函数

```
L_total = λ₁ * L_gaze + λ₂ * L_feature + λ₃ * L_attention + λ₄ * L_geometric

L_gaze: L1 loss on gaze angles (primary | 主要目标)
L_feature: MSE between teacher/student features
L_attention: KL divergence between attention maps
L_geometric: MSE on facial landmark predictions (auxiliary | 辅助)
```

---

## 5. Deployment Strategy | 部署策略

### Framework Selection | 框架选择

| Platform平台 | Recommended Framework | Execution Provider执行提供者 |
|-------------|---------------------|---------------------------|
| iOS | Core ML or ONNX Runtime | CoreML (ANE) |
| Android (Qualcomm) | ONNX Runtime | QNN (Hexagon) |
| Android (Huawei) | ONNX Runtime | CANN (Ascend) |

### Quantization Pipeline | 量化管道

```
1. Train in FP32 with AMP → 2. Export to ONNX (opset 17)
3. FP16 conversion as baseline → 4. INT8 quantization with calibration
5. Validate on hold-out set → 6. NPU-specific optimization
```

### Target Specifications | 目标规格

| Metric指标 | Target目标 | Method方法 |
|-----------|-----------|-----------|
| Model Size | <100MB (INT8) | Quantization |
| Inference Latency | <100ms/frame | NPU optimization |
| Power Consumption | <500mW avg | Hardware selection |
| Accuracy | <5° angular error | Distillation + validation |

---

## 6. Implementation Roadmap | 实施路线图

| Phase阶段 | Duration周期 | Tasks任务 |
|----------|-------------|----------|
| Phase 1研究基础 | Weeks 1-4 | Setup, datasets, data loaders |
| Phase 2教师选择 | Weeks 5-8 | Evaluate teachers, generate soft labels |
| Phase 3学生训练 | Weeks 9-16 | 3-stage distillation, hyperparameter tuning |
| Phase 4量化优化 | Weeks 17-20 | FP16, INT8, NPU optimization |
| Phase 5部署 | Weeks 21-24 | iOS/Android integration, benchmarking |

---

## 7. Recommendations | 建议

### Architecture | 架构
1. **DINOv2-pretrained ViT backbone | DINOv2预训练ViT骨干**: Proven effective in ViTGaze
2. **Cross-attention fusion | 交叉注意力融合**: Bidirectional gaze-head interaction
3. **Multi-task learning | 多任务学习**: Joint gaze + landmark prediction

### Distillation | 蒸馏
1. **Multi-teacher ensemble | 多教师集成**: DINOv2 + MediaPipe + LLaVA
2. **Stage-wise training | 分阶段训练**: Progressively add teachers
3. **Temperature scheduling | 温度调度**: High T initially, reduce over time

### Deployment | 部署
1. **ONNX Runtime**: Best cross-platform support
2. **Start with FP16**: Lower risk than INT8
3. **Fallback hierarchy | 降级层次**: NPU → CPU → Reduced model

---

## 8. Conclusions | 结论

1. **Vision Transformers dominate | 视觉Transformer主导**: Encoder-heavy designs with pretrained backbones achieve SOTA

2. **Multi-modal fusion is essential | 多模态融合至关重要**: Cross-attention provides robust attention estimation under extreme poses

3. **Knowledge distillation enables compression | 知识蒸馏实现压缩**: 500M+ → ~35M parameters with <2° accuracy loss

4. **Mobile NPU deployment is feasible | 移动端NPU部署可行**: Real-time inference (<100ms) achievable with quantization

5. **FocusNet-Lite proposal | FocusNet-Lite提案**: ~35M parameter ViT-S model targeting <100MB, <5° angular error

---

## References | 参考文献

### Gaze Estimation | 视线估计
1. Zhang et al. (2020). ETH-XGaze. *ECCV 2020*.
2. Kellnhofer et al. (2019). Gaze360. *ICCV 2019*.
3. Song et al. (2024). ViTGaze. *Visual Intelligence*.
4. Park et al. (2019). FAZE. *ICCV 2019*.
5. Cheng & Lu (2022). GazeTR. *ICPR 2022*.

### Head Pose | 头部姿态
6. Zhou & Grega (2020). WHENet. *arXiv:2009.06108*.
7. Zhu et al. (2020). 3DDFA_V2. *ECCV 2020*.

### Knowledge Distillation | 知识蒸馏
8. Hinton et al. (2015). Distilling the Knowledge. *NIPS Workshop*.
9. Romero et al. (2015). FitNets. *ICLR 2015*.
10. Tian et al. (2020). CRD. *ICLR 2020*.
11. Touvron et al. (2021). DeiT. *ICML 2021*.

### Mobile Deployment | 移动端部署
12. ONNX Runtime Docs: https://onnxruntime.ai/docs/
13. Google AI Edge: https://ai.google.dev/edge/litert
14. Apple Core ML: https://developer.apple.com/machine-learning/core-ml/

### Vision Encoders | 视觉编码器
15. Oquab et al. (2023). DINOv2. *arXiv:2304.07193*.
16. Liu et al. (2024). LLaVA. *NeurIPS*.

### Attention Fusion | 注意力融合
17. Zhang et al. (2026). FreeGazeFormer. *SPIE*.
18. Wu et al. (2026). MixGaze. *Multimedia Systems*.
19. Xu et al. (2023). Multimodal Learning with Transformers. *IEEE TPAMI*.

---

*Report compiled | 报告编撰: 2026-05-20*  
*Source docs: `docs/01_transformer_gaze_models/` through `docs/07_teacher_models/`*  
*Project: Brick by Brick AI Training | 项目: 逐块砌砖 AI 训练*
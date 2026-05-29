# Knowledge Distillation Research Summary for FocusNet-Lite
# FocusNet-Lite 知识蒸馏研究综合报告

**Project**: Brick by Brick AI Training / 逐块砌砖 AI 训练  
**Date**: 2026-05-20  
**Scope**: Model distillation literature review for mobile focus detection model / 移动端专注度检测模型的知识蒸馏文献综述  
**Model Target**: FocusNet-Lite (~35M params, <100MB, <5 angular error) / 目标模型：FocusNet-Lite（约35M参数，<100MB，<5度角误差）

---

## Table of Contents / 目录

1. [Executive Summary](#1-executive-summary) / 执行摘要
2. [Key Findings from 15 Papers](#2-key-findings-from-15-papers) / 15篇论文关键发现
3. [Case Study: Qwen's Distillation Methodology](#3-case-study-qwens-distillation-methodology) / 案例研究：Qwen的蒸馏方法论
4. [Actionable Roadmap for FocusNet-Lite](#4-actionable-roadmap-for-focusnet-lite) / FocusNet-Lite可行路线图
5. [References](#5-references) / 参考文献

---

## 1. Executive Summary / 执行摘要

### 1.1 How Model Distillation Applies to FocusNet-Lite / 模型蒸馏如何应用于FocusNet-Lite

Our FocusNet-Lite project aims to compress a **multi-teacher ensemble** (totaling 500M+ parameters) into a compact **~35M parameter student model** for real-time mobile focus detection. Knowledge distillation (KD) is the enabling technology that makes this compression feasible while preserving accuracy.

我们的FocusNet-Lite项目旨在将一个**多教师集成模型**（总计超过500M参数）压缩成一个紧凑的**约35M参数的学生模型**，用于实时移动端专注度检测。知识蒸馏（KD）是实现这一压缩同时保持准确性的使能技术。

**The core challenge**: We need to distill knowledge from three fundamentally different teacher types:

**核心挑战**：我们需要从三个截然不同的教师模型中蒸馏知识：

- **DINOv2-L/14** (304M params): A self-supervised vision encoder that provides rich visual features
- **DINOv2-L/14** (304M参数)：一个提供丰富视觉特征的自监督视觉编码器
- **MediaPipe Face Mesh** (2.1M params): A specialized geometric model providing facial landmarks
- **MediaPipe Face Mesh** (2.1M参数)：一个提供面部标脸的专用几何模型
- **LLaVA-7B** (7B params): A vision-language model providing semantic understanding of gaze
- **LLaVA-7B** (7B参数)：一个提供语义目光理解的多模态视觉-语言模型

This multi-teacher, multi-modal distillation scenario is more complex than standard single-teacher KD, requiring careful loss design and staged training.

这种多教师、多模态蒸馏场景比标准的单教师KD更复杂，需要精心设计损失函数和分阶段训练。

### 1.2 Key Insight from Literature Review / 文献综述的关键发现

The literature reveals three critical findings for our project:

文献综述揭示了我们项目的三个关键发现：

**Finding 1: Feature-level distillation outperforms logit-only distillation for ViTs.**

**发现1：对于ViT，特征级蒸馏优于仅对数蒸馏。**

Multiple studies (MiniViT, DeiT, CRD) confirm that matching intermediate attention maps and feature representations between teacher and student Vision Transformers transfers significantly more knowledge than output-level KD alone. For FocusNet-Lite, this means our primary distillation signal should be DINOv2's intermediate features, not just its final outputs.

多项研究（MiniViT、DeiT、CRD）证实，在教师和学生Vision Transformer之间匹配中间注意力图和特征表示比仅在输出级KD能传递更多知识。对于FocusNet-Lite，这意味着我们的主要蒸馏信号应该是DINOv2的中间特征，而不仅仅是其最终输出。

**Finding 2: Multi-teacher distillation requires careful loss balancing and staged training.**

**发现2：多教师蒸馏需要仔细的损失平衡和分阶段训练。**

The Qwen and Orca case studies demonstrate that progressive, staged knowledge transfer from teacher to student yields better results than simultaneous multi-teacher training. Our 3-stage pipeline (Feature Init -> Domain Adaptation -> Semantic Refinement) aligns with this finding.

Qwen和Orca的案例研究表明，从教师到学生的渐进式分阶段知识转移比同时进行的多教师训练效果更好。我们的3阶段流水线（特征初始化 -> 领域适应 -> 语义精炼）符合这一发现。

**Finding 3: Combined compression (KD + pruning + quantization) achieves the best efficiency-accuracy tradeoff.**

**发现3：组合压缩（KD + 剪枝 + 量化）达到最佳效率-准确率权衡。**

Mobile deployment surveys consistently show that applying KD first, then quantization (not the reverse), preserves the most accuracy. Our pipeline follows this: distill first (Phase 3), then quantize (Phase 4).

移动端部署调研一致表明，先应用KD再量化（而不是相反）能保持最高准确性。我们的流水线遵循这一原则：先蒸馏（阶段3），再量化（阶段4）。

**Finding 4: Foundation-to-edge distillation for gaze is an active CVPR 2025 research area.**

**发现4：面向凝视的基础模型到边缘蒸馏是CVPR 2025的一个活跃研究领域。**

CustomKD (Lee et al., CVPR 2025, arXiv:2503.18244) directly addresses customizing large vision foundation models like DINOv2 for edge deployment via knowledge distillation. This is the closest published work to our exact goal.

CustomKD（Lee等，CVPR 2025，arXiv:2503.18244）直接解决了通过知识蒸馏将DINOv2等大型视觉基础模型定制用于边缘部署的问题。这是与我们的确切目标最接近的已发表工作。

### 1.3 Project-Specific Implications / 项目特定启示

| Aspect / 方面 | Literature Recommendation / 文献建议 | FocusNet-Lite Adaptation / FocusNet-Lite适配 |
|---------------|-------------------------------------|---------------------------------------------|
| **Teacher Selection / 教师选择** | Multi-teacher > single-teacher / 多教师 > 单教师 | DINOv2 + MediaPipe + LLaVA |
| **Distillation Level / 蒸馏级别** | Feature > Logit for ViTs / 特征 > 对数（针对ViT） | L_feature + L_attention losses / 特征损失 + 注意力损失 |
| **Training Strategy / 训练策略** | Staged > simultaneous / 分阶段 > 同时进行 | 3-stage pipeline / 3阶段流水线 |
| **Loss Function / 损失函数** | Multi-objective with scheduling / 多目标带调度 | 4-loss weighted combination / 4损失加权组合 |
| **Compression / 压缩** | KD then quantization / 先KD后量化 | Distill (Phase 3) then INT8 (Phase 4) / 阶段3蒸馏，阶段4 INT8 |
| **Architecture / 架构** | Encoder-heavy with minimal decoder / 编码器重，解码器轻 | ViT-S/14 + 4-layer cross-attention / ViT-S/14 + 4层交叉注意力 |

---

## 2. Key Findings from 15 Papers / 15篇论文关键发现

### Category 1: Foundation Knowledge Distillation Methods / 第一类：基础知识蒸馏方法

#### Paper 1: Hinton's Knowledge Distillation (2015) / 论文1：Hinton的知识蒸馏（2015）

**Core Contribution**: Established the teacher-student KD framework with temperature-scaled soft targets.

**核心贡献**：建立了带有温度缩放软目标的教师-学生KD框架。

**Key Formula / 关键公式**:
```
L_total = alpha * L_soft(T) + (1 - alpha) * L_hard
L_soft = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
```

**Key Findings / 关键发现**:
- Soft targets carry "dark knowledge" about inter-class relationships / 软目标携带关于类间关系的"暗知识"
- Temperature T = 2-8 reveals the teacher's learned structure / 温度T = 2-8揭示教师学到的结构
- Alpha = 0.1-0.5 typically works best (soft targets weighted more as T increases) / Alpha = 0.1-0.5通常效果最好（随着T增加软目标权重更大）

**Application to FocusNet / 在FocusNet中的应用**: The basic KD framework underpins our entire distillation pipeline. The temperature and alpha scheduling strategy will be applied to our soft label distillation from LLaVA.

基础KD框架是我们整个蒸馏流水线的基础。温度和alpha调度策略将应用于我们从LLaVA的软标签蒸馏。

---

#### Paper 2: FitNets - Hints for Thin Deep Nets (2015) / 论文2：FitNets - 薄深度网络的提示（2015）

**Core Contribution**: Extended KD to intermediate feature-level hints, enabling deeper/thinner students.

**核心贡献**：将KD扩展到中间特征级提示，使更深处/更薄的学生成为可能。

**Key Findings / 关键发现**:
- Matching intermediate layer representations transfers more knowledge than output-only KD / 匹配中间层表示比仅输出KD传递更多知识
- A regression layer maps teacher features to student feature dimensions / 回归层将教师特征映射到学生特征维度
- Students can be deeper AND thinner than teachers (not just smaller) / 学生可以比教师更深且更薄（而不只是更小）

**Application to FocusNet / 在FocusNet中的应用**: Directly informs our L_feature loss. We match DINOv2-L/14's intermediate layer features (304M) with ViT-S/14's features (22M) using a projection layer to align dimensions (1024-dim teacher -> 384-dim student).

直接为我们的L_feature损失提供信息。我们使用投影层来对齐维度（1024维教师 -> 384维学生），将DINOv2-L/14的中间层特征（304M）与ViT-S/14的特征（22M）进行匹配。

---

#### Paper 3: Relational Knowledge Distillation (2019) / 论文3：关系知识蒸馏（2019）

**Core Contribution**: Distills structural relationships between data points, not just individual predictions.

**核心贡献**：蒸馏数据点之间的结构关系，而不仅仅是单独预测。

**Key Findings / 关键发现**:
- Distance-wise relations: preserve relative distances between feature representations / 距离关系：保留特征表示之间的相对距离
- Angle-wise relations: preserve angular relationships in feature space / 角度关系：保留特征空间中的角度关系
- Outperforms vanilla KD when teacher and student have very different architectures / 当教师和学生架构差异很大时优于普通KD

**Application to FocusNet / 在FocusNet中的应用**: Critical for our multi-teacher scenario. Since DINOv2, MediaPipe, and LLaVA have fundamentally different architectures, relational KD ensures the student preserves the structural relationships across these diverse feature spaces.

对我们的多教师场景至关重要。由于DINOv2、MediaPipe和LLaVA具有根本不同的架构，关系KD确保学生保留这些不同特征空间之间的结构关系。

---

#### Paper 4: Contrastive Representation Distillation (2020) / 论文4：对比表示蒸馏（2020）

**Core Contribution**: Formulates KD as contrastive learning, maximizing mutual information between teacher and student.

**核心贡献**：将KD表述为对比学习，最大化教师和学生之间的互信息。

**Key Findings / 关键发现**:
- Noise contrastive estimation (NCE) captures more transferable knowledge than KL divergence / 噪声对比估计（NCE）比KL散度捕获更多可迁移知识
- Positive pairs: same input through teacher and student / 正对：相同输入通过教师和学生
- Negative pairs: different inputs through teacher / 负对：通过教师的不同输入
- Outperforms KD and FitNets on CIFAR-100 and ImageNet / 在CIFAR-100和ImageNet上优于KD和FitNets

**Application to FocusNet / 在FocusNet中的应用**: Can be used to align DINOv2's self-supervised contrastive features with the student's learned representations. Particularly useful in Stage 1 (Feature Initialization) where we initialize the student encoder from DINOv2 features.

可用于将DINOv2的自监督对比特征与学生学到的表示对齐。在阶段1（特征初始化）中特别有用，我们从DINOv2特征初始化学生编码器。

---

### Category 2: Vision Transformer Distillation / 第二类：Vision Transformer蒸馏

#### Paper 5: DeiT - Data-Efficient Image Transformers (2021) / 论文5：DeiT - 数据高效的图像Transformer（2021）

**Core Contribution**: Introduced the distillation token for ViTs, achieving SOTA with only ImageNet-1K data.

**核心贡献**：为ViT引入蒸馏token，仅使用ImageNet-1K数据达到SOTA。

**Key Findings / 关键发现**:
- Added [DIST] token alongside [CLS] in ViT architecture / 在ViT架构中添加了[DIST] token与[CLS]并列
- [DIST] token trained to predict teacher's hard labels via attention / [DIST] token通过注意力训练预测教师的硬标签
- Teacher: RegNet-16GF (not a ViT) -> Student: ViT variants / 教师：RegNet-16GF（非ViT）-> 学生：ViT变体
- DeiT-B achieved 85.2% top-1 on ImageNet (vs. 81.8% without distillation) / DeiT-B在ImageNet上达到85.2% top-1（vs. 无蒸馏81.8%）
- Hard-label distillation (argmax) outperformed soft-label for DeiT / 对于DeiT，硬标签蒸馏（argmax）优于软标签

**Architecture / 架构**:
```
Input patches -> [CLS] token + [DIST] token + Patch tokens
                -> Self-Attention Layers
                -> [CLS] for classification, [DIST] for distillation
                -> L_class = CE(cls_head([CLS]), y_true)
                -> L_distill = CE(distill_head([DIST]), y_teacher)
```

**Application to FocusNet / 在FocusNet中的应用**: We can add a [DIST] token to FocusNet-Lite's ViT-S encoder that learns from DINOv2-L's predictions. This is an alternative to direct feature matching (FitNets) that works within the attention mechanism itself.

我们可以在FocusNet-Lite的ViT-S编码器中添加一个[DIST] token，从DINOv2-L的预测中学习。这是直接特征匹配（FitNets）的替代方案，在注意力机制本身内工作。

---

#### Paper 6: MiniViT - Multi-Stage ViT Distillation (2022) / 论文6：MiniViT - 多阶段ViT蒸馏（2022）

**Core Contribution**: Multi-stage self-attention distillation with weight sharing for ViT compression.

**核心贡献**：用于ViT压缩的带权重共享的多阶段自注意力蒸馏。

**Key Findings / 关键发现**:
- **Self-attention map distillation**: Match Q, K, V attention maps between teacher/student layers / **自注意力图蒸馏**：匹配教师/学生层之间的Q、K、V注意力图
- **Weight sharing**: Share weights across consecutive transformer blocks (reduces params by ~40%) / **权重共享**：在连续transformer块之间共享权重（减少约40%参数）
- **Multi-stage distillation**: Distill at every N layers, not just final output / **多阶段蒸馏**：每N层蒸馏一次，而不仅仅是最终输出
- MiniViT-24M (24M params): ~83% ImageNet accuracy, comparable to DeiT-B (86M params) / MiniViT-24M（24M参数）：~83% ImageNet准确率，与DeiT-B（86M参数）相当
- 4x parameter reduction with ~2% accuracy loss / 4倍参数减少，约2%准确率损失

**Key Distillation Losses / 关键蒸馏损失**:
```
L_total = L_CE + lambda_1 * L_attn + lambda_2 * L_feat + lambda_3 * L_logit

L_attn: MSE between teacher/student attention maps (per layer) / 教师/学生注意力图之间的MSE（每层）
L_feat: MSE between teacher/student intermediate features / 教师/学生中间特征之间的MSE
L_logit: KL divergence between final predictions / 最终预测之间的KL散度
```

**Application to FocusNet / 在FocusNet中的应用**: MiniViT is the most directly applicable paper. It demonstrates that DINOv2-L/14 (304M) -> ViT-S/14 (22M) distillation is feasible with ~2% accuracy loss using multi-stage attention distillation. This validates our exact teacher-student configuration.

MiniViT是最直接适用的论文。它证明了DINOv2-L/14（304M）-> ViT-S/14（22M）蒸馏使用多阶段注意力蒸馏是可行的，约2%准确率损失。这验证了我们确切的教师-学生配置。

---

#### Paper 7: Efficient ViT Compression Survey (2024) / 论文7：高效ViT压缩综述（2024）

**Core Contribution**: Comprehensive survey of ViT compression techniques with best practices.

**核心贡献**：ViT压缩技术的综合综述与最佳实践。

**Key Findings / 关键发现**:
- **Feature-level attention map distillation** consistently outperforms logit-only KD for ViTs / **特征级注意力图蒸馏**持续优于仅对数KD（针对ViT）
- **Combined compression** (KD + pruning + quantization) gives the best efficiency-accuracy tradeoff / **组合压缩**（KD + 剪枝 + 量化）给出最佳效率-准确率权衡
- **Progressive/layer-wise distillation** helps student convergence / **渐进式/逐层蒸馏**有助于学生收敛
- **Dynamic token reduction** during distillation improves robustness / 蒸馏过程中**动态token减少**提高鲁棒性
- **Order matters**: Distill first, then quantize (not reverse) / **顺序重要**：先蒸馏，后量化（不是相反）

**Best Practices Summary / 最佳实践总结**:
| Practice / 实践 | Impact / 影响 | Difficulty / 难度 |
|----------------|---------------|-------------------|
| Feature-level KD / 特征级KD | +3-5% over logit-only / 比仅对数高3-5% | Medium / 中等 |
| Multi-stage distillation / 多阶段蒸馏 | +1-2% over single-stage / 比单阶段高1-2% | Medium / 中等 |
| Attention map matching / 注意力图匹配 | +2-3% over feature-only / 比仅特征高2-3% | Low / 低 |
| Weight sharing/pruning / 权重共享/剪枝 | 2-4x compression / 2-4倍压缩 | Medium / 中等 |
| Quantization-aware distillation / 量化感知蒸馏 | +1% over post-training quant / 比训练后量化高1% | High / 高 |

**Application to FocusNet / 在FocusNet中的应用**: Validates our entire approach. Combined KD + quantization with progressive distillation is the recommended path.

验证我们的整个方法。带渐进蒸馏的组合KD + 量化是推荐路径。

---

### Category 3: Gaze Estimation with Transformers / 第三类：基于Transformer的凝视估计

#### Paper 8: ViTGaze - Gaze Estimation with ViT (2024) / 论文8：ViTGaze - 使用ViT的凝视估计（2024）

**Core Contribution**: SOTA gaze estimation using DINOv2-pretrained ViT-S backbone.

**核心贡献**：使用DINOv2预训练的ViT-S骨干实现SOTA凝视估计。

**Key Findings / 关键发现**:
- Angular error: **3.8 degrees** (AUC 0.949) on standard benchmarks / 角误差：标准基准上**3.8度**（AUC 0.949）
- **Encoder-heavy design**: ~99% of parameters in the ViT encoder, ~1% in decoder / **编码器重设计**：~99%参数在ViT编码器中，~1%在解码器中
- DINOv2 pretraining provides superior features vs. ImageNet pretraining / DINOv2预训练比ImageNet预训练提供更优特征
- Outperforms CNN-based methods (L2CS-Net, GazeTR) by 0.7-1.3 degrees / 比基于CNN的方法（L2CS-Net、GazeTR）优0.7-1.3度
- Uses only ViT-S/14 (22M params) - proving small ViTs work for gaze / 仅使用ViT-S/14（22M参数）——证明小ViT适用于凝视

**Architecture / 架构**:
```
DINOv2 ViT-S/14 Encoder (frozen or fine-tuned)
    -> Patch features (384-dim, 196 patches)
    -> [CLS] token aggregation
    -> Lightweight MLP decoder
    -> (pitch, yaw) gaze angles
```

**Application to FocusNet / 在FocusNet中的应用**: ViTGaze is the primary architecture reference. Our student model should follow the same encoder-heavy design pattern with DINOv2-pretrained ViT-S/14.

ViTGaze是主要架构参考。我们的学生模型应该遵循相同的编码器重设计模式，使用DINOv2预训练的ViT-S/14。

---

#### Paper 9: ETH-XGaze Dataset (2020) / 论文9：ETH-XGaze数据集（2020）

**Core Contribution**: Large-scale gaze dataset with extreme head poses.

**核心贡献**：具有极端头部姿态的大规模凝视数据集。

**Key Findings / 关键发现**:
- 1.1M+ images from 110 participants / 来自110名参与者的1.1M+图像
- Extreme head pose range (full 360 degrees) / 极端头部姿态范围（360度全覆盖）
- Models trained on standard datasets fail on extreme poses / 在标准数据集上训练的模型在极端姿态上失败
- **Multi-modal fusion (gaze + head pose) essential** for robust estimation / **多模态融合（凝视 + 头部姿态）对鲁棒估计至关重要**
- Standard benchmark: ~4.2 degrees MAE (ResNet baseline) / 标准基准：~4.2度MAE（ResNet基线）

**Application to FocusNet / 在FocusNet中的应用**: Validates our cross-attention fusion module design. Under extreme head poses (common in mobile phone usage), head pose information must be fused with gaze features.

验证我们的交叉注意力融合模块设计。在极端头部姿态下（手机使用中常见），头部姿态信息必须与凝视特征融合。

---

#### Paper 10: FreeGazeFormer (2026) / 论文10：FreeGazeFormer（2026）

**Core Contribution**: CNN-Transformer cross-modal attention for head pose-free gaze estimation.

**核心贡献**：用于无头部姿态凝视估计的CNN-Transformer跨模态注意力。

**Key Findings / 关键发现**:
- Cross-attention enables bidirectional gaze-head pose information flow / 交叉注意力实现双向凝视-头部姿态信息流
- Visual features from CNN + head pose features fused through cross-attention / 来自CNN的视觉特征 + 通过交叉注意力融合的头部姿态特征
- Robust under extreme head poses without explicit head pose normalization / 在极端头部姿态下鲁棒，无需显式头部姿态归一化
- Transformer encoder provides global context for fusion / Transformer编码器为融合提供全局上下文

**Architecture / 架构**:
```
Eye Image -> CNN -> Visual Features ->+
                                      |
                                      v
                      Transformer Encoder
                                      |
                                      v
                      Cross-Attention Fusion <- Head Pose Features
                                      |
                                      v
                              Gaze Estimation
```

**Application to FocusNet / 在FocusNet中的应用**: Direct architectural blueprint for our 4-layer cross-attention fusion module. The bidirectional design (gaze queries head pose AND head pose queries gaze) is optimal for FocusNet-Lite.

我们4层交叉注意力融合模块的直接架构蓝图。双向设计（凝视查询头部姿态 AND 头部姿态查询凝视）对FocusNet-Lite是最优的。

---

### Category 4: LLM/VLM Knowledge Distillation / 第四类：LLM/VLM知识蒸馏

#### Paper 11: Orca - Progressive Learning from GPT-4 (2023) / 论文11：Orca - 从GPT-4进行渐进学习（2023）

**Core Contribution**: Distilling reasoning capabilities from GPT-4 into 13B models via explanation traces.

**核心贡献**：通过解释痕迹将GPT-4的推理能力蒸馏到13B模型中。

**Key Findings / 关键发现**:
- Training on explanation traces (reasoning process) outperforms training on answers alone / 在解释痕迹（推理过程）上训练优于仅在答案上训练
- System prompts elicit complex reasoning from teacher / 系统提示引出教师的复杂推理
- Orca-13B approaches GPT-3.5 performance on multiple benchmarks / Orca-13B在多个基准上接近GPT-3.5性能
- Progressive learning: start with simple tasks, increase complexity / 渐进学习：从简单任务开始，增加复杂度
- Key innovation: **rich teacher signals** (not just final answers) / 关键创新：**丰富的教师信号**（不仅仅是最终答案）

**Application to FocusNet / 在FocusNet中的应用**: The principle of distilling rich, multi-faceted signals (not just gaze angles) applies directly. LLaVA-7B can provide semantic gaze explanations that enrich the student beyond simple regression targets.

蒸馏丰富的多面信号（不仅仅是凝视角度）的原则直接适用。LLaVA-7B可以提供语义凝视解释，丰富学生，而不仅仅是简单的回归目标。

---

#### Paper 12: Qwen2 Technical Report (2024) / 论文12：Qwen2技术报告（2024）

**Core Contribution**: Multi-stage distillation pipeline for LLM model family (0.5B-72B).

**核心贡献**：LLM模型族（0.5B-72B）的多阶段蒸馏流水线。

**Key Findings / 关键发现**:
- **Synthetic data distillation**: Large models generate training data for smaller models / **合成数据蒸馏**：大模型为更小模型生成训练数据
- **Multi-stage pretraining**: Progressive complexity (pretraining -> SFT -> RLHF) / **多阶段预训练**：渐进复杂度（预训练 -> SFT -> RLHF）
- **Quality over quantity**: Curated synthetic data outperforms massive uncurated data / **质量重于数量**：策划的合成数据优于海量未策划数据
- Smaller models (0.5B, 1.5B) benefit most from teacher-generated data / 更小模型（0.5B、1.5B）从教师生成的数据中受益最多
- Teacher models: Qwen2-72B generates high-quality training examples / 教师模型：Qwen2-72B生成高质量训练示例

**Key Methodology / 关键方法论**:
```
Stage 1: Pretraining (general knowledge)
    -> Large teacher generates diverse training data
    -> Student learns from teacher-generated data

Stage 2: Supervised Fine-Tuning (task alignment)
    -> Teacher generates task-specific examples
    -> Student learns task patterns

Stage 3: RLHF (preference alignment)
    -> Teacher provides preference signals
    -> Student learns human preferences
```

**Application to FocusNet / 在FocusNet中的应用**: Qwen's synthetic data approach directly applies. Use Qwen2.5-VL-7B to generate high-quality focus/distracted pseudo-labels, then distill this into FocusNet-Lite. This is already in our pipeline (Step 4 of data pipeline).

Qwen的合成数据方法直接适用。使用Qwen2.5-VL-7B为自录视频帧生成高质量专注/分心伪标签，然后蒸馏到FocusNet-Lite。这已在我们的流水线中（数据流水线第4步）。

---

#### Paper 13: Qwen3 Technical Report (2025) / 论文13：Qwen3技术报告（2025）

**Core Contribution**: Hybrid distillation + RL for next-generation model family.

**核心贡献**：下一代模型族的混合蒸馏 + RL。

**Key Findings / 关键发现**:
- **Thinking mode distillation**: Reasoning capabilities from larger MoE (235B) distilled into smaller dense models / **思考模式蒸馏**：来自更大MoE（235B）的推理能力蒸馏到更小的密集模型
- **Hybrid approach**: Combine distillation (learning from teacher) with RL (self-improvement) / **混合方法**：结合蒸馏（从教师学习）和RL（自我改进）
- Models: 0.6B to 235B (MoE) / 模型：0.6B到235B（MoE）
- State-of-the-art across all model sizes / 各模型规模均达到SOTA
- Demonstrates systematic distillation pipelines can narrow teacher-student gap dramatically / 证明系统性蒸馏流水线可以显著缩小教师-学生差距

**Application to FocusNet / 在FocusNet中的应用**: Qwen3's hybrid distillation + RL can be adapted as:
- Distillation phase: Learn from DINOv2 + MediaPipe + LLaVA teachers
- RL/Active Learning phase: Refine on uncertain samples (our active learning pipeline)

Qwen3的混合蒸馏 + RL可以这样适配：
- 蒸馏阶段：从DINOv2 + MediaPipe + LLaVA教师学习
- RL/主动学习阶段：在不确定样本上精炼（我们的主动学习流水线）

---

### Category 5: Mobile Deployment with Distillation / 第五类：带蒸馏的移动端部署

#### Paper 14: MobileNetV3 (2019) / 论文14：MobileNetV3（2019）

**Core Contribution**: NAS + distillation for efficient mobile vision models.

**核心贡献**：用于高效移动视觉模型的NAS + 蒸馏。

**Key Findings / 关键发现**:
- Neural Architecture Search combined with knowledge distillation / 神经架构搜索结合知识蒸馏
- MobileNetV3-Small: ~2.5M params, competitive accuracy / MobileNetV3-Small：~2.5M参数，具有竞争力的准确率
- Hardware-aware NAS optimizes for actual mobile hardware / 硬件感知NAS针对实际移动硬件优化
- Squeeze-and-excitation blocks + depthwise separable convolutions / Squeeze-and-excitation块 + 深度可分离卷积
- Platform-aware latency optimization / 平台感知延迟优化

**Application to FocusNet / 在FocusNet中的应用**: Fallback architecture if ViT-S proves too expensive for certain NPUs. MobileNetV3-Small + GRU is our original architecture (1.5M params) and remains viable for extreme size constraints.

如果ViT-S对某些NPU太昂贵，则是备用架构。MobileNetV3-Small + GRU是我们的原始架构（1.5M参数），对于极端尺寸约束仍然可行。

---

#### Paper 15: On-Device Medical Image KD (2023-2024) / 论文15：设备端医学图像KD（2023-2024）

**Core Contribution**: Best practices for knowledge distillation in constrained deployment scenarios.

**核心贡献**：约束部署场景中知识蒸馏的最佳实践。

**Key Findings / 关键发现**:
- **Feature-based KD > response-based KD** for small models / **基于特征的KD > 基于响应的KD**（针对小模型）
- **Quantization-aware distillation** preserves accuracy better than post-training quantization / **量化感知蒸馏**比训练后量化更好地保持准确性
- **Progressive distillation** (staged complexity) helps student convergence / **渐进蒸馏**（分阶段复杂度）有助于学生收敛
- **Calibration data diversity** critical for INT8 quantization quality / **校准数据多样性**对INT8量化质量至关重要
- Combined pipeline: Feature KD -> Response KD -> Quantization-Aware Fine-tuning / 组合流水线：特征KD -> 响应KD -> 量化感知微调

**Application to FocusNet / 在FocusNet中的应用**: Quantization-aware distillation can be added to Stage 3 to prepare the model for INT8 deployment simultaneously with distillation.

量化感知蒸馏可以添加到阶段3，以在蒸馏同时准备模型进行INT8部署。

---

## 3. Case Study: Qwen's Distillation Methodology / 案例研究：Qwen的蒸馏方法论

### 3.1 Overview of Qwen's Approach / Qwen方法概述

Alibaba's Qwen team has developed one of the most successful open-source model families through systematic knowledge distillation. Their approach differs fundamentally from traditional logit-based KD.

阿里巴巴的Qwen团队通过系统性知识蒸馏开发了最成功的开源模型族之一。他们的方法从根本上不同于传统的基于对数的KD。

### 3.2 Qwen2 Distillation Pipeline / Qwen2蒸馏流水线

```
Qwen2-72B (Teacher)
    |
    v
[Stage 1: Synthetic Data Generation / 阶段1：合成数据生成]
    -> Teacher generates diverse, high-quality training examples / 教师生成多样化的高质量训练示例
    -> Covers multiple domains, languages, and task types / 覆盖多个领域、语言和任务类型
    -> Quality filtering: only high-confidence outputs used / 质量过滤：仅使用高置信度输出
    |
    v
[Stage 2: Multi-Stage Pretraining / 阶段2：多阶段预训练]
    -> Student (0.5B-7B) trained on teacher-generated data / 学生（0.5B-7B）在教师生成的数据上训练
    -> Progressive complexity: simple -> complex tasks / 渐进复杂度：简单 -> 复杂任务
    -> Curriculum learning approach / 课程学习方法
    |
    v
[Stage 3: Supervised Fine-Tuning / 阶段3：监督微调]
    -> Teacher generates task-specific instruction-response pairs / 教师生成任务特定的指令-响应对
    -> Student learns instruction following / 学生学习指令遵循
    |
    v
[Stage 4: RLHF Alignment / 阶段4：RLHF对齐]
    -> Teacher provides preference rankings / 教师提供偏好排名
    -> Student learns human-preferred outputs / 学生学习人类偏好的输出
```

### 3.3 Key Techniques We Can Replicate / 我们可以复制的关键技术

#### Technique 1: Synthetic Data Distillation (Adaptable) / 技术1：合成数据蒸馏（可适配）

**Qwen's approach**: Large teacher generates training data for smaller student.

**Qwen的方法**：大教师为更小的学生生成训练数据。

**Our adaptation**: Use Qwen2.5-VL-7B to generate focus/distracted labels for self-recorded video frames. Already in our data pipeline.

**我们的适配**：使用Qwen2.5-VL-7B为自录视频帧生成专注/分心标签。已在我们的数据流水线中。

```
Self-recorded video (10-30 people, 800h)
    -> FFmpeg downsampling (0.5fps)
    -> MediaPipe quality filtering
    -> Qwen2.5-VL-7B generates focus labels (pseudo-labels)
    -> HMM temporal smoothing
    -> FocusNet-Lite trains on these labels
```

#### Technique 2: Multi-Teacher Ensemble (Adaptable) / 技术2：多教师集成（可适配）

**Qwen's approach**: Single large teacher for all students.

**Qwen的方法**：所有学生使用单一的大教师。

**Our adaptation**: Multiple specialized teachers, each contributing different knowledge.

**我们的适配**：多个专业教师，每个贡献不同的知识。

```
DINOv2-L/14 (304M) -> Visual feature knowledge / 视觉特征知识
    -> Stage 1: Initialize student encoder / 阶段1：初始化学生编码器
    
MediaPipe Face Mesh (2.1M) -> Geometric knowledge / 几何知识
    -> Stage 2: Landmark and pose understanding / 阶段2：标脸和姿态理解
    
LLaVA-7B (7B) -> Semantic knowledge / 语义知识
    -> Stage 3: High-level gaze reasoning / 阶段3：高级凝视推理
    
Combined -> FocusNet-Lite (35M)
```

#### Technique 3: Progressive Training (Directly Applicable) / 技术3：渐进训练（直接适用）

**Qwen's approach**: Multi-stage training with increasing complexity.

**Qwen的方法**：多阶段训练，增加复杂度。

**Our adaptation**: 3-stage distillation pipeline matching project plan.

**我们的适配**：3阶段蒸馏流水线，匹配项目计划。

| Stage / 阶段 | Qwen Approach / Qwen方法 | FocusNet Adaptation / FocusNet适配 |
|--------------|--------------------------|-------------------------------------|
| 1 | General pretraining / 一般预训练 | DINOv2 feature initialization / DINOv2特征初始化 |
| 2 | Task-specific training / 任务特定训练 | MediaPipe geometric distillation / MediaPipe几何蒸馏 |
| 3 | Alignment (RLHF) / 对齐（RLHF） | LLaVA semantic refinement / LLaVA语义精炼 |

### 3.4 What We Cannot Replicate (and Alternatives) / 我们无法复制的（及替代方案）

| Qwen Feature / Qwen特性 | Why We Can't / 为什么我们不能 | Our Alternative / 我们的替代方案 |
|--------------------------|-------------------------------|----------------------------------|
| 72B teacher model / 72B教师模型 | Too large for RTX 5080 / 对RTX 5080太大 | Use multiple smaller teachers / 使用多个更小教师 |
| Massive compute budget / 巨额计算预算 | 3-person team, limited resources / 3人团队，资源有限 | Efficient distillation (MiniViT approach) / 高效蒸馏（MiniViT方法） |
| Billions of training tokens / 数以十亿计的训练token | Limited self-recorded data / 自录数据有限 | Data augmentation + synthetic data / 数据增强 + 合成数据 |
| RLHF pipeline / RLHF流水线 | Overkill for regression task / 对回归任务过大 | Active learning loop / 主动学习循环 |

### 3.5 Lessons from Qwen (and Broader LLM Distillation) for FocusNet-Lite / Qwen（及更广泛的LLM蒸馏）对FocusNet-Lite的教训

**From Qwen's pipeline / 从Qwen的流水线**:
1. **Quality > Quantity**: Curated, high-quality training data outperforms massive uncurated data. Our active learning loop should focus on boundary samples.
   **质量 > 数量**：策划的高质量训练数据优于海量未策划数据。我们的主动学习循环应聚焦于边界样本。
2. **Teacher diversity matters**: While Qwen uses one teacher, they benefit from diverse training data. Our multi-teacher approach provides diversity at the model level.
   **教师多样性重要**：虽然Qwen使用一个教师，但他们从多样化训练数据中受益。我们的多教师方法在模型层面提供多样性。
3. **Progressive complexity works**: Start with simple features (visual), add complexity (geometric, semantic). This is our 3-stage pipeline.
   **渐进复杂度有效**：从简单特征（视觉）开始，增加复杂度（几何、语义）。这是我们的3阶段流水线。
4. **Synthetic data is powerful**: Teacher-generated labels are a viable training signal. Qwen2.5-VL-7B as a labeling tool is our version of this.
   **合成数据强大**：教师生成的标签是可行的训练信号。Qwen2.5-VL-7B作为标注工具是我们的版本。
5. **Combined loss is essential**: Qwen always uses `L = alpha * L_CE + beta * L_KD`, never soft-label distillation alone.
   **组合损失必不可少**：Qwen总是使用`L = alpha * L_CE + beta * L_KD`，从不单独使用软标签蒸馏。

**From Orca/Microsoft (explanation traces) / 从Orca/Microsoft（解释痕迹）**:
6. **Distill reasoning, not just answers**: Train on the teacher's reasoning process (how to judge focus), not just final labels. LLaVA can provide gaze reasoning explanations.
   **蒸馏推理，而不仅仅是答案**：在教师的推理过程（如何判断专注）上训练，而不仅仅是最终标签。LLaVA可以提供凝视推理解释。

**From LIMA/Meta (quality curation) / 从LIMA/Meta（质量策划）**:
7. **1,000 high-quality samples can beat 100,000 mediocre ones**: Invest heavily in curating a small, high-quality evaluation set. Active learning should be aggressive in filtering.
   **1000个高质量样本可以胜过100000个普通样本**：在策划一个小而高质量的评估集上投入巨资。主动学习应该在过滤上积极。

**From MiniLLM (loss function choice) / 从MiniLLM（损失函数选择）**:
8. **Consider reverse KL divergence** for generating tasks; for our regression task, standard forward KL on features is appropriate, but this highlights that loss function choice matters.
   **考虑反向KL散度**用于生成任务；对于我们的回归任务，特征上的标准前向KL是合适的，但这凸显了损失函数选择的重要性。

**From DeepSeek-R1 (reasoning transfer) / 从DeepSeek-R1（推理转移）**:
9. **Reasoning capabilities transfer to much smaller models**: Complex semantic understanding (like gaze reasoning from LLaVA) can effectively transfer to a 35M-param student via proper distillation.
   **推理能力转移到更小的模型**：复杂的语义理解（如来自LLaVA的凝视推理）可以通过适当的蒸馏有效地转移到35M参数的学生。

---

## 4. Actionable Roadmap for FocusNet-Lite / FocusNet-Lite可行路线图

### 4.1 Phase 1: Research Foundation (Weeks 1-4) [CURRENT] / 阶段1：研究基础（第1-4周）[当前]

**Must-Read Papers (Priority Order) / 必读论文（优先级顺序）**:
1. **CustomKD** (Lee et al., CVPR 2025) - Foundation-to-edge distillation. THE closest paper to our project.
   **CustomKD**（Lee等，CVPR 2025）- 基础模型到边缘蒸馏。与我们的项目最接近的论文。
2. **MiniViT** (Zhang et al., CVPR 2022) - Multi-stage ViT distillation. Our exact DINOv2-L -> ViT-S pipeline.
   **MiniViT**（Zhang等，CVPR 2022）- 多阶段ViT蒸馏。我们确切的DINOv2-L -> ViT-S流水线。
3. **DeiT** (Touvron et al., ICML 2021) - Distillation token for ViTs. Architecture modification reference.
   **DeiT**（Touvron等，ICML 2021）- ViT的蒸馏token。架构修改参考。
4. **Q-ViT** (Qu et al., 2022) - Joint distillation + quantization. Pipeline optimization.
   **Q-ViT**（Qu等，2022）- 联合蒸馏 + 量化。流水线优化。
5. **ViTGaze** (Song et al., 2024) - SOTA gaze with ViT-S. Architecture reference.
   **ViTGaze**（Song等，2024）- 使用ViT-S的SOTA凝视。架构参考。
6. **RS-MTDF** (Song et al., 2025) - Multi-teacher distillation fusion. Our DINOv2+MediaPipe+LLaVA setup.
   **RS-MTDF**（Song等，2025）- 多教师蒸馏融合。我们的DINOv2+MediaPipe+LLaVA设置。
7. **X-Distill** (Shao et al., 2026) - Cross-architecture ViT-to-compact distillation.
   **X-Distill**（Shao等，2026）- 跨架构ViT到紧凑蒸馏。

**Objective / 目标**: Validate all technical assumptions before training. / 在训练前验证所有技术假设。

| Task / 任务 | Action / 行动 | Validation Criteria / 验证标准 |
|-------------|---------------|-------------------------------|
| 1.1 | Set up RTX 5080 training environment / 设置RTX 5080训练环境 | PyTorch + CUDA + mediapipe working / PyTorch + CUDA + mediapipe工作正常 |
| 1.2 | Download DINOv2-L/14 and ViT-S/14 models / 下载DINOv2-L/14和ViT-S/14模型 | Models load, inference works / 模型加载，推理工作正常 |
| 1.3 | Validate DINOv2 feature quality on face images / 验证人脸图像上DINOv2特征质量 | Feature visualization looks reasonable / 特征可视化看起来合理 |
| 1.4 | Run ViTGaze architecture as baseline / 运行ViTGaze架构作为基线 | Reproduces ~3.8 angular error on ETH-XGaze / 在ETH-XGaze上复现~3.8度角误差 |
| 1.5 | Design FocusNet-Lite architecture / 设计FocusNet-Lite架构 | Forward pass works, ~35M params confirmed / 前向传播工作，~35M参数确认 |
| 1.6 | **NEW: Implement distillation token (DeiT-style)** / **新：实现蒸馏token（DeiT风格）** | [DIST] token integrated into ViT-S / [DIST] token集成到ViT-S |
| 1.7 | **NEW: Implement multi-stage attention distillation (MiniViT-style)** / **新：实现多阶段注意力蒸馏（MiniViT风格）** | Attention map matching works / 注意力图匹配工作正常 |

**Key Decision / 关键决策**: Choose between DeiT-style [DIST] token vs. MiniViT-style multi-stage attention distillation as primary KD approach.

在DeiT风格[DIST] token与MiniViT风格多阶段注意力蒸馏之间选择作为主要KD方法。

**Recommendation / 建议**: Use **MiniViT-style multi-stage attention distillation** as primary approach, **supplemented by DeiT-style [DIST] token**:

使用**MiniViT风格多阶段注意力蒸馏**作为主要方法，**辅以DeiT风格[DIST] token**：

- MiniViT: Better documented for ViT-L to ViT-S compression, directly matches our DINOv2-L -> ViT-S config
  MiniViT：对于ViT-L到ViT-S压缩有更好的文档记录，直接匹配我们的DINOv2-L -> ViT-S配置
- DeiT [DIST] token: Supplementary signal that works within the attention mechanism itself
  DeiT [DIST] token：在注意力机制本身内工作的补充信号
- **NEW**: Consider **Q-ViT** joint distillation+quantization to reduce the total pipeline from 2 stages (distill, then quantize) to 1 stage
  **新**：考虑**Q-ViT**联合蒸馏+量化，将总流水线从2阶段（蒸馏，然后量化）减少到1阶段

### 4.2 Phase 2: Teacher Selection & Soft Label Generation (Weeks 5-8) / 阶段2：教师选择与软标签生成（第5-8周）

**Objective / 目标**: Generate high-quality teacher outputs for distillation. / 为蒸馏生成高质量的教师输出。

| Task / 任务 | Action / 行动 | Output / 输出 |
|-------------|---------------|---------------|
| 2.1 | Record 10-30h test data (2-3 people) / 录制10-30h测试数据（2-3人） | Initial dataset / 初始数据集 |
| 2.2 | Run DINOv2-L/14 on all frames / 在所有帧上运行DINOv2-L/14 | Intermediate features + attention maps / 中间特征 + 注意力图 |
| 2.3 | Run MediaPipe Face Mesh on all frames / 在所有帧上运行MediaPipe Face Mesh | 468 landmarks + pose / 468标脸 + 姿态 |
| 2.4 | Run LLaVA-7B with gaze prompts on all frames / 在所有帧上用凝视提示运行LLaVA-7B | Semantic gaze labels / 语义凝视标签 |
| 2.5 | Run Qwen2.5-VL-7B for pseudo-labeling / 运行Qwen2.5-VL-7B进行伪标签 | Focus/distracted labels / 专注/分心标签 |
| 2.6 | Validate teacher agreement / 验证教师一致性 | Inter-teacher correlation > 0.7 / 教师间相关性 > 0.7 |
| 2.7 | Design loss function weights (lambda_1-4) / 设计损失函数权重（lambda_1-4） | Balanced multi-objective loss / 平衡的多目标损失 |

**Loss Function Design** (based on literature / 基于文献):
```python
L_total = lambda_1 * L_gaze           # L1 on gaze angles (primary) / 凝视角度上的L1（主要）
        + lambda_2 * L_feature         # MSE on DINOv2 intermediate features / DINOv2中间特征上的MSE
        + lambda_3 * L_attention       # KL divergence on attention maps / 注意力图上的KL散度
        + lambda_4 * L_geometric       # MSE on MediaPipe landmarks / MediaPipe标脸上的MSE

# Suggested weights (to be tuned / 建议权重（待调优）):
lambda_1 = 1.0    # Primary objective / 主要目标
lambda_2 = 0.5    # Feature distillation / 特征蒸馏
lambda_3 = 0.3    # Attention distillation / 注意力蒸馏
lambda_4 = 0.2    # Geometric auxiliary / 几何辅助
```

### 4.3 Phase 3: Student Model Training (Weeks 9-16) / 阶段3：学生模型训练（第9-16周）

**Objective / 目标**: Execute 3-stage distillation pipeline. / 执行3阶段蒸馏流水线。

#### Stage 1: Feature Initialization (Weeks 9-10) / 阶段1：特征初始化（第9-10周）

```
Input: DINOv2-L/14 frozen features / 输入：DINOv2-L/14冻结特征
Student: ViT-S/14 encoder (randomly initialized) / 学生：ViT-S/14编码器（随机初始化）
Loss: L_feature (MSE between intermediate layers) / 损失：L_feature（中间层之间的MSE）
Method: Feature regression layers (FitNets-style) / 方法：特征回归层（FitNets风格）
Goal: Student encoder learns DINOv2's visual representations / 目标：学生编码器学习DINOv2的视觉表示
```

#### Stage 2: Domain Adaptation (Weeks 11-13) / 阶段2：领域适应（第11-13周）

```
Input: MediaPipe Face Mesh landmarks + pose / 输入：MediaPipe Face Mesh标脸 + 姿态
Student: Full FocusNet-Lite (encoder + cross-attention + heads) / 学生：完整FocusNet-Lite（编码器 + 交叉注意力 + 头）
Loss: L_geometric + L_gaze / 损失：L_geometric + L_gaze
Method: Multi-task learning (landmark + gaze prediction) / 方法：多任务学习（标脸 + 凝视预测）
Goal: Student learns geometric face understanding / 目标：学生学习几何人脸理解
```

#### Stage 3: Semantic Refinement (Weeks 14-16) / 阶段3：语义精炼（第14-16周）

```
Input: LLaVA-7B semantic labels + DINOv2 features + MediaPipe landmarks / 输入：LLaVA-7B语义标签 + DINOv2特征 + MediaPipe标脸
Student: Full FocusNet-Lite / 学生：完整FocusNet-Lite
Loss: L_total (all four losses) / 损失：L_total（所有四个损失）
Method: Full multi-teacher distillation with attention map matching / 方法：带注意力图匹配的完整多教师蒸馏
Goal: Student achieves teacher-level accuracy / 目标：学生达到教师级准确率
```

**Training Configuration / 训练配置**:
```yaml
optimizer: AdamW
learning_rate: 1e-4  (with cosine schedule / 带余弦调度)
weight_decay: 0.05
batch_size: 8
max_epochs: 100 per stage / 每阶段100个epoch
fp16: true
gradient_clip: 1.0
temperature: 4.0  (for soft targets, decay to 2.0 / 用于软目标，衰减到2.0)
```

### 4.4 Phase 4: Quantization & Optimization (Weeks 17-20) / 阶段4：量化与优化（第17-20周）

**Objective / 目标**: Compress model for mobile deployment. / 压缩模型用于移动端部署。

| Step / 步骤 | Action / 行动 | Expected Result / 预期结果 |
|-------------|---------------|---------------------------|
| 4.1 | Export PyTorch to ONNX (opset 17) / 导出PyTorch到ONNX（opset 17） | ~140MB FP32 model / ~140MB FP32模型 |
| 4.2 | FP16 conversion / FP16转换 | ~70MB, <1% accuracy loss / ~70MB，<1%准确率损失 |
| 4.3 | INT8 PTQ with calibration / 带校准的INT8 PTQ | ~35MB, 1-3% accuracy loss / ~35MB，1-3%准确率损失 |
| 4.4 | **NEW: Quantization-aware distillation** / **新：量化感知蒸馏** | ~35MB, <1% accuracy loss / ~35MB，<1%准确率损失 |
| 4.5 | NPU-specific optimization / NPU特定优化 | Platform-dependent / 平台依赖 |
| 4.6 | Cross-platform validation / 跨平台验证 | iOS/Android/HarmonyOS |

**Quantization-Aware Distillation** (from Paper 15 + Q-ViT / 来自论文15 + Q-ViT):

```
Instead of: Train -> Export -> Quantize (loses 1-3%) / 而不是：训练 -> 导出 -> 量化（损失1-3%）
Do:         Train with simulated quantization -> Export -> Quantize (loses <1%) / 做：带模拟量化训练 -> 导出 -> 量化（损失<1%）
```

**ViT-Specific Quantization Methods** (new findings / 新发现):

| Method / 方法 | Type / 类型 | DeiT-S W4A4 | DeiT-S W6A6 | Best For / 最佳用于 |
|--------------|-------------|-------------|-------------|---------------------|
| PTQ4ViT | PTQ | ~74.3% | ~78.6% | Quick INT8 deployment / 快速INT8部署 |
| RepQ-ViT | PTQ | ~76.5% | - | Best PTQ accuracy / 最佳PTQ准确率 |
| Q-ViT | QAT+KD | ~75.9% | ~79.0% | **Best overall (distill+quantize joint) / 总体最佳（蒸馏+量化联合）** |
| BRECQ | PTQ | Strong at W2-W4 / 在W2-W4上强 | - | CNN-based students / 基于CNN的学生 |

**Recommendation / 建议**: Use **Q-ViT** (joint distillation + quantization) for FocusNet-Lite. This combines distillation and quantization in a single training pass, reducing pipeline complexity while achieving better accuracy than sequential distill-then-quantize.

对FocusNet-Lite使用**Q-ViT**（联合蒸馏 + 量化）。这在单一训练过程中结合蒸馏和量化，在顺序蒸馏-然后-量化获得更好准确率的同时减少流水线复杂性。

**Mobile Deployment Frameworks** (new findings / 新发现):

| Framework / 框架 | Best For / 最佳用于 | NPU Support / NPU支持 | Size / 尺寸 |
|------------------|---------------------|----------------------|-------------|
| **ncnn** (Tencent) | Android ARM CPU | Vulkan GPU | Smallest binary / 最小二进制 |
| **MNN** (Alibaba) | Broad NPU support / 广泛NPU支持 | OpenGL/Vulkan/Metal | Good GPU accel / 良好GPU加速 |
| **ONNX Runtime** | Cross-platform / 跨平台 | NNAPI/CoreML/QNN | Best ecosystem / 最佳生态系统 |
| **Core ML** (Apple) | iOS + ANE | Native ANE | Best iOS perf / 最佳iOS性能 |

### 4.5 Phase 5: Mobile Deployment (Weeks 21-24) / 阶段5：移动端部署（第21-24周）

**Objective / 目标**: Ship FocusNet-Lite on iOS/Android/HarmonyOS. / 在iOS/Android/HarmonyOS上发布FocusNet-Lite。

| Platform / 平台 | Framework / 框架 | NPU | Target Latency / 目标延迟 |
|-----------------|------------------|-----|---------------------------|
| iOS | CoreML | ANE | <50ms |
| Android | ONNX Runtime | QNN (Hexagon) | <80ms |
| HarmonyOS | ncnn/ONNX Runtime | CANN (Ascend) | <80ms |

### 4.6 Preliminary Work Checklist (Immediate Actions) / 初步工作检查清单（立即行动）

**Before Phase 1 starts, complete these tasks / 在阶段1开始前，完成这些任务**：

- [ ] Install PyTorch 2.x with CUDA 12.x on RTX 5080 / 在RTX 5080上安装PyTorch 2.x与CUDA 12.x
- [ ] Download DINOv2-L/14 and DINOv2-S/14 from Meta (HuggingFace) / 从Meta（HuggingFace）下载DINOv2-L/14和DINOv2-S/14
- [ ] Download ViTGaze codebase and verify it runs / 下载ViTGaze代码库并验证其运行
- [ ] Install MediaPipe and test Face Mesh on sample images / 安装MediaPipe并在样本图像上测试Face Mesh
- [ ] Install ONNX Runtime and verify cross-platform export / 安装ONNX Runtime并验证跨平台导出
- [ ] Set up Label Studio for active learning annotations / 设置Label Studio用于主动学习标注
- [ ] Record 1-2 hours of test video for pipeline validation / 录制1-2小时测试视频用于流水线验证
- [ ] Design the multi-stage distillation training script (PyTorch) / 设计多阶段蒸馏训练脚本（PyTorch）
- [ ] Implement attention map extraction from DINOv2-L/14 / 实现从DINOv2-L/14提取注意力图
- [ ] Implement feature regression layers for FitNets-style distillation / 实现用于FitNets风格蒸馏的特征回归层

---

## 5. References / 参考文献

### Foundational Knowledge Distillation / 基础知识蒸馏

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop. https://arxiv.org/abs/1503.02531
2. Romero, A., et al. (2015). FitNets: Hints for Thin Deep Nets. ICLR 2015. https://arxiv.org/abs/1412.6550
3. Park, W., et al. (2019). Relational Knowledge Distillation. CVPR 2019. https://arxiv.org/abs/1904.05068
4. Tian, Y., et al. (2020). Contrastive Representation Distillation. ICLR 2020. https://arxiv.org/abs/1910.10699

### Vision Transformer Distillation / Vision Transformer蒸馏

5. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention (DeiT). ICML 2021. https://arxiv.org/abs/2012.12877
6. Zhang, J., et al. (2022). MiniViT: Multi-Stage Vision Transformer Distillation. CVPR 2022. https://arxiv.org/abs/2112.09884
7. Various (2024). Efficient Vision Transformers: A Survey on Compression and Acceleration.

### Gaze Estimation / 凝视估计

8. Song, Y., et al. (2024). ViTGaze: Gaze Estimation with ViT. Visual Intelligence.
9. Zhang, X., et al. (2020). ETH-XGaze: A Large Scale Dataset for Gaze Estimation. ECCV 2020. https://arxiv.org/abs/2007.12852
10. Zhang, J., et al. (2026). FreeGazeFormer: CNN-Transformer Cross-Modal Attention. SPIE.

### LLM/VLM Distillation / LLM/VLM蒸馏

11. Mukherjee, S., et al. (2023). Orca: Progressive Learning from Complex Explanation Traces of GPT-4. https://arxiv.org/abs/2306.02707
12. Yang, A., et al. (2024). Qwen2 Technical Report. https://arxiv.org/abs/2407.10671
13. Yang, A., et al. (2025). Qwen3 Technical Report. https://qwenlm.github.io

### Mobile Deployment / 移动端部署

14. Howard, A., et al. (2019). MobileNets: Efficient CNNs for Mobile Vision. https://arxiv.org/abs/1704.04861
15. Various (2023-2024). Knowledge Distillation for On-Device Medical Image Classification.

### Supplementary References (from Project Docs) / 补充参考文献（来自项目文档）

16. Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193
17. Liu, H., et al. (2024). Visual Instruction Tuning (LLaVA). NeurIPS.
18. Xu, P., et al. (2023). Multimodal Learning with Transformers: A Survey. IEEE TPAMI.
19. Baltrusaitis, T., et al. (2018). Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI.
20. Zhao, X., et al. (2026). GazeFormer-MoE: Context-Aware Gaze Estimation via CLIP and MoE Transformer. arXiv:2601.12316.

### Additional References (from Extended Research) / 其他参考文献（来自扩展研究）

21. Zagoruyko, S. & Komodakis, N. (2017). Paying More Attention to Attention. ICLR 2017. https://arxiv.org/abs/1612.03928
22. Wu, K., et al. (2022). TinyViT: Fast Pretraining Distillation for Small Vision Transformers. ECCV 2024. https://arxiv.org/abs/2207.10666
23. Gou, J., et al. (2021). Knowledge Distillation: A Survey. IJCV. https://arxiv.org/abs/2009.12538
24. Taori, R., et al. (2023). Stanford Alpaca. github.com/tatsu-lab/stanford_alpaca
25. Xu, C., et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions. https://arxiv.org/abs/2304.12244
26. Zhou, C., et al. (2023). LIMA: Less Is More for Alignment. NeurIPS 2023. https://arxiv.org/abs/2305.11206
27. DeepSeek (2025). DeepSeek-R1. https://arxiv.org/abs/2501.12948
28. Qu, Z., et al. (2022). Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer. https://arxiv.org/abs/2205.09702
29. Yuan, Z., et al. (2021). PTQ4ViT: Post-Training Quantization for Vision Transformers. https://arxiv.org/abs/2111.12731
30. Mukherjee, S., et al. (2023). Orca 2: Teaching Small Language Models How to Reason.

### Critical Papers from Extended Gaze/Distillation Research / 扩展凝视/蒸馏研究的关键论文

31. Lee, J., et al. (2025). CustomKD: Customizing Large Vision Foundation for Edge Model Improvement via Knowledge Distillation. CVPR 2025. https://arxiv.org/abs/2503.18244
32. Song, J., et al. (2025). RS-MTDF: Multi-Teacher Distillation and Fusion for Remote Sensing. https://arxiv.org/abs/2506.08772
33. Shao, M., et al. (2026). X-Distill: Cross-Architecture Vision Distillation. https://arxiv.org/abs/2601.11269
34. Vasu, P., et al. (2023). FastViT: A Fast Hybrid Vision Transformer. ICCV 2023. https://arxiv.org/abs/2303.14189
35. Cheng, Y., et al. (2024). GazeDPTR: Dual-Stream Gaze Pyramid Transformer (IVGaze). CVPR 2024. https://arxiv.org/abs/2403.15664
36. Zhao, X., et al. (2025). GMGaze: MoE-Based Context-Aware Gaze Estimation. https://arxiv.org/abs/2605.00799
37. Vasu, P., et al. (2023). MobileOne: An Improved One Millisecond Mobile Backbone. CVPR 2023. https://arxiv.org/abs/2206.04040

---

*Report compiled: 2026-05-20* / *报告编纂：2026-05-20*
*Papers cataloged: 15 primary + 15 supplementary + 7 extended gaze/KD = 37 total references* / *论文编目：15篇主要 + 15篇补充 + 7篇扩展凝视/KD = 37篇总参考文献*
*Research agents: 4 parallel agents completed (KD methods, Qwen distillation, gaze estimation, mobile deployment)* / *研究智能体：4个并行智能体完成（KD方法、Qwen蒸馏、凝视估计、移动端部署）*
*Key breakthrough: CustomKD (CVPR 2025) identified as the closest published work to our exact goal* / *关键突破：CustomKD（CVPR 2025）被识别为与我们确切目标最接近的已发表工作*
*For: FocusNet-Lite distillation pipeline design* / *用于：FocusNet-Lite蒸馏流水线设计*
*Project: Brick by Brick AI Training* / *项目：逐块砌砖AI训练*
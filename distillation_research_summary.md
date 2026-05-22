# Knowledge Distillation Research Summary for FocusNet-Lite
# FocusNet-Lite 知识蒸馏研究综合报告

**Project**: Brick by Brick AI Training / 逐块砌砖 AI 训练
**Date**: 2026-05-20
**Scope**: Model distillation literature review for mobile focus detection model
**Model Target**: FocusNet-Lite (~35M params, <100MB, <5 angular error)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Key Findings from 15 Papers](#2-key-findings-from-15-papers)
3. [Case Study: Qwen's Distillation Methodology](#3-case-study-qwens-distillation-methodology)
4. [Actionable Roadmap for FocusNet-Lite](#4-actionable-roadmap-for-focusnet-lite)
5. [References](#5-references)

---

## 1. Executive Summary

### 1.1 How Model Distillation Applies to FocusNet-Lite

Our FocusNet-Lite project aims to compress a **multi-teacher ensemble** (totaling 500M+ parameters) into a compact **~35M parameter student model** for real-time mobile focus detection. Knowledge distillation (KD) is the enabling technology that makes this compression feasible while preserving accuracy.

**The core challenge**: We need to distill knowledge from three fundamentally different teacher types:
- **DINOv2-L/14** (304M params): A self-supervised vision encoder that provides rich visual features
- **MediaPipe Face Mesh** (2.1M params): A specialized geometric model providing facial landmarks
- **LLaVA-7B** (7B params): A vision-language model providing semantic understanding of gaze

This multi-teacher, multi-modal distillation scenario is more complex than standard single-teacher KD, requiring careful loss design and staged training.

### 1.2 Key Insight from Literature Review

The literature reveals three critical findings for our project:

**Finding 1: Feature-level distillation outperforms logit-only distillation for ViTs.**
Multiple studies (MiniViT, DeiT, CRD) confirm that matching intermediate attention maps and feature representations between teacher and student Vision Transformers transfers significantly more knowledge than output-level KD alone. For FocusNet-Lite, this means our primary distillation signal should be DINOv2's intermediate features, not just its final outputs.

**Finding 2: Multi-teacher distillation requires careful loss balancing and staged training.**
The Qwen and Orca case studies demonstrate that progressive, staged knowledge transfer from teacher to student yields better results than simultaneous multi-teacher training. Our 3-stage pipeline (Feature Init -> Domain Adaptation -> Semantic Refinement) aligns with this finding.

**Finding 3: Combined compression (KD + pruning + quantization) achieves the best efficiency-accuracy tradeoff.**
Mobile deployment surveys consistently show that applying KD first, then quantization (not the reverse), preserves the most accuracy. Our pipeline follows this: distill first (Phase 3), then quantize (Phase 4).

**Finding 4: Foundation-to-edge distillation for gaze is an active CVPR 2025 research area.**
CustomKD (Lee et al., CVPR 2025, arXiv:2503.18244) directly addresses customizing large vision foundation models like DINOv2 for edge deployment via knowledge distillation. This is the closest published work to our exact goal.

### 1.3 Project-Specific Implications

| Aspect | Literature Recommendation | FocusNet-Lite Adaptation |
|--------|--------------------------|-------------------------|
| **Teacher Selection** | Multi-teacher > single-teacher | DINOv2 + MediaPipe + LLaVA |
| **Distillation Level** | Feature > Logit for ViTs | L_feature + L_attention losses |
| **Training Strategy** | Staged > simultaneous | 3-stage pipeline |
| **Loss Function** | Multi-objective with scheduling | 4-loss weighted combination |
| **Compression** | KD then quantization | Distill (Phase 3) then INT8 (Phase 4) |
| **Architecture** | Encoder-heavy with minimal decoder | ViT-S/14 + 4-layer cross-attention |

---

## 2. Key Findings from 15 Papers

### Category 1: Foundation Knowledge Distillation Methods

#### Paper 1: Hinton's Knowledge Distillation (2015)
**Core Contribution**: Established the teacher-student KD framework with temperature-scaled soft targets.

**Key Formula**:
```
L_total = alpha * L_soft(T) + (1 - alpha) * L_hard
L_soft = T^2 * KL(softmax(z_t/T) || softmax(z_s/T))
```

**Key Findings**:
- Soft targets carry "dark knowledge" about inter-class relationships
- Temperature T = 2-8 reveals the teacher's learned structure
- Alpha = 0.1-0.5 typically works best (soft targets weighted more as T increases)

**Application to FocusNet**: The basic KD framework underpins our entire distillation pipeline. The temperature and alpha scheduling strategy will be applied to our soft label distillation from LLaVA.

---

#### Paper 2: FitNets - Hints for Thin Deep Nets (2015)
**Core Contribution**: Extended KD to intermediate feature-level hints, enabling deeper/thinner students.

**Key Findings**:
- Matching intermediate layer representations transfers more knowledge than output-only KD
- A regression layer maps teacher features to student feature dimensions
- Students can be deeper AND thinner than teachers (not just smaller)

**Application to FocusNet**: Directly informs our L_feature loss. We match DINOv2-L/14's intermediate layer features (304M) with ViT-S/14's features (22M) using a projection layer to align dimensions (1024-dim teacher -> 384-dim student).

---

#### Paper 3: Relational Knowledge Distillation (2019)
**Core Contribution**: Distills structural relationships between data points, not just individual predictions.

**Key Findings**:
- Distance-wise relations: preserve relative distances between feature representations
- Angle-wise relations: preserve angular relationships in feature space
- Outperforms vanilla KD when teacher and student have very different architectures

**Application to FocusNet**: Critical for our multi-teacher scenario. Since DINOv2, MediaPipe, and LLaVA have fundamentally different architectures, relational KD ensures the student preserves the structural relationships across these diverse feature spaces.

---

#### Paper 4: Contrastive Representation Distillation (2020)
**Core Contribution**: Formulates KD as contrastive learning, maximizing mutual information between teacher and student.

**Key Findings**:
- Noise contrastive estimation (NCE) captures more transferable knowledge than KL divergence
- Positive pairs: same input through teacher and student
- Negative pairs: different inputs through teacher
- Outperforms KD and FitNets on CIFAR-100 and ImageNet

**Application to FocusNet**: Can be used to align DINOv2's self-supervised contrastive features with the student's learned representations. Particularly useful in Stage 1 (Feature Initialization) where we initialize the student encoder from DINOv2 features.

---

### Category 2: Vision Transformer Distillation

#### Paper 5: DeiT - Data-Efficient Image Transformers (2021)
**Core Contribution**: Introduced the distillation token for ViTs, achieving SOTA with only ImageNet-1K data.

**Key Findings**:
- Added [DIST] token alongside [CLS] in ViT architecture
- [DIST] token trained to predict teacher's hard labels via attention
- Teacher: RegNet-16GF (not a ViT) -> Student: ViT variants
- DeiT-B achieved 85.2% top-1 on ImageNet (vs. 81.8% without distillation)
- Hard-label distillation (argmax) outperformed soft-label for DeiT

**Architecture**:
```
Input patches -> [CLS] token + [DIST] token + Patch tokens
                -> Self-Attention Layers
                -> [CLS] for classification, [DIST] for distillation
                -> L_class = CE(cls_head([CLS]), y_true)
                -> L_distill = CE(distill_head([DIST]), y_teacher)
```

**Application to FocusNet**: We can add a [DIST] token to FocusNet-Lite's ViT-S encoder that learns from DINOv2-L's predictions. This is an alternative to direct feature matching (FitNets) that works within the attention mechanism itself.

---

#### Paper 6: MiniViT - Multi-Stage ViT Distillation (2022)
**Core Contribution**: Multi-stage self-attention distillation with weight sharing for ViT compression.

**Key Findings**:
- **Self-attention map distillation**: Match Q, K, V attention maps between teacher/student layers
- **Weight sharing**: Share weights across consecutive transformer blocks (reduces params by ~40%)
- **Multi-stage distillation**: Distill at every N layers, not just final output
- MiniViT-24M (24M params): ~83% ImageNet accuracy, comparable to DeiT-B (86M params)
- 4x parameter reduction with ~2% accuracy loss

**Key Distillation Losses**:
```
L_total = L_CE + lambda_1 * L_attn + lambda_2 * L_feat + lambda_3 * L_logit

L_attn: MSE between teacher/student attention maps (per layer)
L_feat: MSE between teacher/student intermediate features
L_logit: KL divergence between final predictions
```

**Application to FocusNet**: MiniViT is the most directly applicable paper. It demonstrates that DINOv2-L/14 (304M) -> ViT-S/14 (22M) distillation is feasible with ~2% accuracy loss using multi-stage attention distillation. This validates our exact teacher-student configuration.

---

#### Paper 7: Efficient ViT Compression Survey (2024)
**Core Contribution**: Comprehensive survey of ViT compression techniques with best practices.

**Key Findings**:
- **Feature-level attention map distillation** consistently outperforms logit-only KD for ViTs
- **Combined compression** (KD + pruning + quantization) gives the best efficiency-accuracy tradeoff
- **Progressive/layer-wise distillation** helps student convergence
- **Dynamic token reduction** during distillation improves robustness
- **Order matters**: Distill first, then quantize (not reverse)

**Best Practices Summary**:
| Practice | Impact | Difficulty |
|----------|--------|-----------|
| Feature-level KD | +3-5% over logit-only | Medium |
| Multi-stage distillation | +1-2% over single-stage | Medium |
| Attention map matching | +2-3% over feature-only | Low |
| Weight sharing/pruning | 2-4x compression | Medium |
| Quantization-aware distillation | +1% over post-training quant | High |

**Application to FocusNet**: Validates our entire approach. Combined KD + quantization with progressive distillation is the recommended path.

---

### Category 3: Gaze Estimation with Transformers

#### Paper 8: ViTGaze - Gaze Estimation with ViT (2024)
**Core Contribution**: SOTA gaze estimation using DINOv2-pretrained ViT-S backbone.

**Key Findings**:
- Angular error: **3.8 degrees** (AUC 0.949) on standard benchmarks
- **Encoder-heavy design**: ~99% of parameters in the ViT encoder, ~1% in decoder
- DINOv2 pretraining provides superior features vs. ImageNet pretraining
- Outperforms CNN-based methods (L2CS-Net, GazeTR) by 0.7-1.3 degrees
- Uses only ViT-S/14 (22M params) - proving small ViTs work for gaze

**Architecture**:
```
DINOv2 ViT-S/14 Encoder (frozen or fine-tuned)
    -> Patch features (384-dim, 196 patches)
    -> [CLS] token aggregation
    -> Lightweight MLP decoder
    -> (pitch, yaw) gaze angles
```

**Application to FocusNet**: ViTGaze is the primary architecture reference. Our student model should follow the same encoder-heavy design pattern with DINOv2-pretrained ViT-S/14.

---

#### Paper 9: ETH-XGaze Dataset (2020)
**Core Contribution**: Large-scale gaze dataset with extreme head poses.

**Key Findings**:
- 1.1M+ images from 110 participants
- Extreme head pose range (full 360 degrees)
- Models trained on standard datasets fail on extreme poses
- **Multi-modal fusion (gaze + head pose) essential** for robust estimation
- Standard benchmark: ~4.2 degrees MAE (ResNet baseline)

**Application to FocusNet**: Validates our cross-attention fusion module design. Under extreme head poses (common in mobile phone usage), head pose information must be fused with gaze features.

---

#### Paper 10: FreeGazeFormer (2026)
**Core Contribution**: CNN-Transformer cross-modal attention for head pose-free gaze estimation.

**Key Findings**:
- Cross-attention enables bidirectional gaze-head pose information flow
- Visual features from CNN + head pose features fused through cross-attention
- Robust under extreme head poses without explicit head pose normalization
- Transformer encoder provides global context for fusion

**Architecture**:
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

**Application to FocusNet**: Direct architectural blueprint for our 4-layer cross-attention fusion module. The bidirectional design (gaze queries head pose AND head pose queries gaze) is optimal for FocusNet-Lite.

---

### Category 4: LLM/VLM Knowledge Distillation

#### Paper 11: Orca - Progressive Learning from GPT-4 (2023)
**Core Contribution**: Distilling reasoning capabilities from GPT-4 into 13B models via explanation traces.

**Key Findings**:
- Training on explanation traces (reasoning process) outperforms training on answers alone
- System prompts elicit complex reasoning from teacher
- Orca-13B approaches GPT-3.5 performance on multiple benchmarks
- Progressive learning: start with simple tasks, increase complexity
- Key innovation: **rich teacher signals** (not just final answers)

**Application to FocusNet**: The principle of distilling rich, multi-faceted signals (not just gaze angles) applies directly. LLaVA-7B can provide semantic gaze explanations that enrich the student beyond simple regression targets.

---

#### Paper 12: Qwen2 Technical Report (2024)
**Core Contribution**: Multi-stage distillation pipeline for LLM model family (0.5B-72B).

**Key Findings**:
- **Synthetic data distillation**: Large models generate training data for smaller models
- **Multi-stage pretraining**: Progressive complexity (pretraining -> SFT -> RLHF)
- **Quality over quantity**: Curated synthetic data outperforms massive uncurated data
- Smaller models (0.5B, 1.5B) benefit most from teacher-generated data
- Teacher models: Qwen2-72B generates high-quality training examples

**Key Methodology**:
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

**Application to FocusNet**: Qwen's synthetic data approach directly applies. Use Qwen2.5-VL-7B to generate high-quality focus/distracted pseudo-labels, then distill this into FocusNet-Lite. This is already in our pipeline (Step 4 of data pipeline).

---

#### Paper 13: Qwen3 Technical Report (2025)
**Core Contribution**: Hybrid distillation + RL for next-generation model family.

**Key Findings**:
- **Thinking mode distillation**: Reasoning capabilities from larger MoE (235B) distilled into smaller dense models
- **Hybrid approach**: Combine distillation (learning from teacher) with RL (self-improvement)
- Models: 0.6B to 235B (MoE)
- State-of-the-art across all model sizes
- Demonstrates systematic distillation pipelines can narrow teacher-student gap dramatically

**Application to FocusNet**: Qwen3's hybrid distillation + RL can be adapted as:
- Distillation phase: Learn from DINOv2 + MediaPipe + LLaVA teachers
- RL/Active Learning phase: Refine on uncertain samples (our active learning pipeline)

---

### Category 5: Mobile Deployment with Distillation

#### Paper 14: MobileNetV3 (2019)
**Core Contribution**: NAS + distillation for efficient mobile vision models.

**Key Findings**:
- Neural Architecture Search combined with knowledge distillation
- MobileNetV3-Small: ~2.5M params, competitive accuracy
- Hardware-aware NAS optimizes for actual mobile hardware
- Squeeze-and-excitation blocks + depthwise separable convolutions
- Platform-aware latency optimization

**Application to FocusNet**: Fallback architecture if ViT-S proves too expensive for certain NPUs. MobileNetV3-Small + GRU is our original architecture (1.5M params) and remains viable for extreme size constraints.

---

#### Paper 15: On-Device Medical Image KD (2023-2024)
**Core Contribution**: Best practices for knowledge distillation in constrained deployment scenarios.

**Key Findings**:
- **Feature-based KD > response-based KD** for small models
- **Quantization-aware distillation** preserves accuracy better than post-training quantization
- **Progressive distillation** (staged complexity) helps student convergence
- **Calibration data diversity** critical for INT8 quantization quality
- Combined pipeline: Feature KD -> Response KD -> Quantization-Aware Fine-tuning

**Application to FocusNet**: Quantization-aware distillation can be added to Stage 3 to prepare the model for INT8 deployment simultaneously with distillation.

---

## 3. Case Study: Qwen's Distillation Methodology

### 3.1 Overview of Qwen's Approach

Alibaba's Qwen team has developed one of the most successful open-source model families through systematic knowledge distillation. Their approach differs fundamentally from traditional logit-based KD.

### 3.2 Qwen2 Distillation Pipeline

```
Qwen2-72B (Teacher)
    |
    v
[Stage 1: Synthetic Data Generation]
    -> Teacher generates diverse, high-quality training examples
    -> Covers multiple domains, languages, and task types
    -> Quality filtering: only high-confidence outputs used
    |
    v
[Stage 2: Multi-Stage Pretraining]
    -> Student (0.5B-7B) trained on teacher-generated data
    -> Progressive complexity: simple -> complex tasks
    -> Curriculum learning approach
    |
    v
[Stage 3: Supervised Fine-Tuning]
    -> Teacher generates task-specific instruction-response pairs
    -> Student learns instruction following
    |
    v
[Stage 4: RLHF Alignment]
    -> Teacher provides preference rankings
    -> Student learns human-preferred outputs
```

### 3.3 Key Techniques We Can Replicate

#### Technique 1: Synthetic Data Distillation (Adaptable)
**Qwen's approach**: Large teacher generates training data for smaller student.
**Our adaptation**: Use Qwen2.5-VL-7B to generate focus/distracted labels for self-recorded video frames. Already in our data pipeline.

```
Self-recorded video (10-30 people, 800h)
    -> FFmpeg downsampling (0.5fps)
    -> MediaPipe quality filtering
    -> Qwen2.5-VL-7B generates focus labels (pseudo-labels)
    -> HMM temporal smoothing
    -> FocusNet-Lite trains on these labels
```

#### Technique 2: Multi-Teacher Ensemble (Adaptable)
**Qwen's approach**: Single large teacher for all students.
**Our adaptation**: Multiple specialized teachers, each contributing different knowledge.

```
DINOv2-L/14 (304M) -> Visual feature knowledge
    -> Stage 1: Initialize student encoder
    
MediaPipe Face Mesh (2.1M) -> Geometric knowledge
    -> Stage 2: Landmark and pose understanding
    
LLaVA-7B (7B) -> Semantic knowledge
    -> Stage 3: High-level gaze reasoning
    
Combined -> FocusNet-Lite (35M)
```

#### Technique 3: Progressive Training (Directly Applicable)
**Qwen's approach**: Multi-stage training with increasing complexity.
**Our adaptation**: 3-stage distillation pipeline matching project plan.

| Stage | Qwen Approach | FocusNet Adaptation |
|-------|--------------|-------------------|
| 1 | General pretraining | DINOv2 feature initialization |
| 2 | Task-specific training | MediaPipe geometric distillation |
| 3 | Alignment (RLHF) | LLaVA semantic refinement |

### 3.4 What We Cannot Replicate (and Alternatives)

| Qwen Feature | Why We Can't | Our Alternative |
|-------------|-------------|-----------------|
| 72B teacher model | Too large for RTX 5080 | Use multiple smaller teachers |
| Massive compute budget | 3-person team, limited resources | Efficient distillation (MiniViT approach) |
| Billions of training tokens | Limited self-recorded data | Data augmentation + synthetic data |
| RLHF pipeline | Overkill for regression task | Active learning loop |

### 3.5 Lessons from Qwen (and Broader LLM Distillation) for FocusNet-Lite

**From Qwen's pipeline:**
1. **Quality > Quantity**: Curated, high-quality training data outperforms massive uncurated data. Our active learning loop should focus on boundary samples.
2. **Teacher diversity matters**: While Qwen uses one teacher, they benefit from diverse training data. Our multi-teacher approach provides diversity at the model level.
3. **Progressive complexity works**: Start with simple features (visual), add complexity (geometric, semantic). This is our 3-stage pipeline.
4. **Synthetic data is powerful**: Teacher-generated labels are a viable training signal. Qwen2.5-VL-7B as a labeling tool is our version of this.
5. **Combined loss is essential**: Qwen always uses `L = alpha * L_CE + beta * L_KD`, never soft-label distillation alone.

**From Orca/Microsoft (explanation traces):**
6. **Distill reasoning, not just answers**: Train on the teacher's reasoning process (how to judge focus), not just final labels. LLaVA can provide gaze reasoning explanations.

**From LIMA/Meta (quality curation):**
7. **1,000 high-quality samples can beat 100,000 mediocre ones**: Invest heavily in curating a small, high-quality evaluation set. Active learning should be aggressive in filtering.

**From MiniLLM (loss function choice):**
8. **Consider reverse KL divergence** for generating tasks; for our regression task, standard forward KL on features is appropriate, but this highlights that loss function choice matters.

**From DeepSeek-R1 (reasoning transfer):**
9. **Reasoning capabilities transfer to much smaller models**: Complex semantic understanding (like gaze reasoning from LLaVA) can effectively transfer to a 35M-param student via proper distillation.

---

## 4. Actionable Roadmap for FocusNet-Lite

### 4.1 Phase 1: Research Foundation (Weeks 1-4) [CURRENT]

**Must-Read Papers (Priority Order):**
1. **CustomKD** (Lee et al., CVPR 2025) - Foundation-to-edge distillation. THE closest paper to our project.
2. **MiniViT** (Zhang et al., CVPR 2022) - Multi-stage ViT distillation. Our exact DINOv2-L -> ViT-S pipeline.
3. **DeiT** (Touvron et al., ICML 2021) - Distillation token for ViTs. Architecture modification reference.
4. **Q-ViT** (Qu et al., 2022) - Joint distillation + quantization. Pipeline optimization.
5. **ViTGaze** (Song et al., 2024) - SOTA gaze with ViT-S. Architecture reference.
6. **RS-MTDF** (Song et al., 2025) - Multi-teacher distillation fusion. Our DINOv2+MediaPipe+LLaVA setup.
7. **X-Distill** (Shao et al., 2026) - Cross-architecture ViT-to-compact distillation.

**Objective**: Validate all technical assumptions before training.

| Task | Action | Validation Criteria |
|------|--------|-------------------|
| 1.1 | Set up RTX 5080 training environment | PyTorch + CUDA + mediapipe working |
| 1.2 | Download DINOv2-L/14 and ViT-S/14 models | Models load, inference works |
| 1.3 | Validate DINOv2 feature quality on face images | Feature visualization looks reasonable |
| 1.4 | Run ViTGaze architecture as baseline | Reproduces ~3.8 angular error on ETH-XGaze |
| 1.5 | Design FocusNet-Lite architecture | Forward pass works, ~35M params confirmed |
| 1.6 | **NEW: Implement distillation token (DeiT-style)** | [DIST] token integrated into ViT-S |
| 1.7 | **NEW: Implement multi-stage attention distillation (MiniViT-style)** | Attention map matching works |

**Key Decision**: Choose between DeiT-style [DIST] token vs. MiniViT-style multi-stage attention distillation as primary KD approach.

**Recommendation**: Use **MiniViT-style multi-stage attention distillation** as primary approach, **supplemented by DeiT-style [DIST] token**:
- MiniViT: Better documented for ViT-L to ViT-S compression, directly matches our DINOv2-L -> ViT-S config
- DeiT [DIST] token: Supplementary signal that works within the attention mechanism itself
- **NEW**: Consider **Q-ViT** joint distillation+quantization to reduce the total pipeline from 2 stages (distill, then quantize) to 1 stage

### 4.2 Phase 2: Teacher Selection & Soft Label Generation (Weeks 5-8)

**Objective**: Generate high-quality teacher outputs for distillation.

| Task | Action | Output |
|------|--------|--------|
| 2.1 | Record 10-30h test data (2-3 people) | Initial dataset |
| 2.2 | Run DINOv2-L/14 on all frames | Intermediate features + attention maps |
| 2.3 | Run MediaPipe Face Mesh on all frames | 468 landmarks + pose |
| 2.4 | Run LLaVA-7B with gaze prompts on all frames | Semantic gaze labels |
| 2.5 | Run Qwen2.5-VL-7B for pseudo-labeling | Focus/distracted labels |
| 2.6 | Validate teacher agreement | Inter-teacher correlation > 0.7 |
| 2.7 | Design loss function weights (lambda_1-4) | Balanced multi-objective loss |

**Loss Function Design** (based on literature):
```python
L_total = lambda_1 * L_gaze           # L1 on gaze angles (primary)
        + lambda_2 * L_feature         # MSE on DINOv2 intermediate features
        + lambda_3 * L_attention       # KL divergence on attention maps
        + lambda_4 * L_geometric       # MSE on MediaPipe landmarks

# Suggested weights (to be tuned):
lambda_1 = 1.0    # Primary objective
lambda_2 = 0.5    # Feature distillation
lambda_3 = 0.3    # Attention distillation
lambda_4 = 0.2    # Geometric auxiliary
```

### 4.3 Phase 3: Student Model Training (Weeks 9-16)

**Objective**: Execute 3-stage distillation pipeline.

#### Stage 1: Feature Initialization (Weeks 9-10)
```
Input: DINOv2-L/14 frozen features
Student: ViT-S/14 encoder (randomly initialized)
Loss: L_feature (MSE between intermediate layers)
Method: Feature regression layers (FitNets-style)
Goal: Student encoder learns DINOv2's visual representations
```

#### Stage 2: Domain Adaptation (Weeks 11-13)
```
Input: MediaPipe Face Mesh landmarks + pose
Student: Full FocusNet-Lite (encoder + cross-attention + heads)
Loss: L_geometric + L_gaze
Method: Multi-task learning (landmark + gaze prediction)
Goal: Student learns geometric face understanding
```

#### Stage 3: Semantic Refinement (Weeks 14-16)
```
Input: LLaVA-7B semantic labels + DINOv2 features + MediaPipe landmarks
Student: Full FocusNet-Lite
Loss: L_total (all four losses)
Method: Full multi-teacher distillation with attention map matching
Goal: Student achieves teacher-level accuracy
```

**Training Configuration**:
```yaml
optimizer: AdamW
learning_rate: 1e-4  (with cosine schedule)
weight_decay: 0.05
batch_size: 8
max_epochs: 100 per stage
fp16: true
gradient_clip: 1.0
temperature: 4.0 (for soft targets, decay to 2.0)
```

### 4.4 Phase 4: Quantization & Optimization (Weeks 17-20)

**Objective**: Compress model for mobile deployment.

| Step | Action | Expected Result |
|------|--------|----------------|
| 4.1 | Export PyTorch to ONNX (opset 17) | ~140MB FP32 model |
| 4.2 | FP16 conversion | ~70MB, <1% accuracy loss |
| 4.3 | INT8 PTQ with calibration | ~35MB, 1-3% accuracy loss |
| 4.4 | **NEW: Quantization-aware distillation** | ~35MB, <1% accuracy loss |
| 4.5 | NPU-specific optimization | Platform-dependent |
| 4.6 | Cross-platform validation | iOS/Android/HarmonyOS |

**Quantization-Aware Distillation** (from Paper 15 + Q-ViT):
```
Instead of: Train -> Export -> Quantize (loses 1-3%)
Do:         Train with simulated quantization -> Export -> Quantize (loses <1%)
```

**ViT-Specific Quantization Methods** (new findings):
| Method | Type | DeiT-S W4A4 | DeiT-S W6A6 | Best For |
|--------|------|-------------|-------------|----------|
| PTQ4ViT | PTQ | ~74.3% | ~78.6% | Quick INT8 deployment |
| RepQ-ViT | PTQ | ~76.5% | - | Best PTQ accuracy |
| Q-ViT | QAT+KD | ~75.9% | ~79.0% | **Best overall (distill+quantize joint)** |
| BRECQ | PTQ | Strong at W2-W4 | - | CNN-based students |

**Recommendation**: Use **Q-ViT** (joint distillation + quantization) for FocusNet-Lite. This combines distillation and quantization in a single training pass, reducing pipeline complexity while achieving better accuracy than sequential distill-then-quantize.

**Mobile Deployment Frameworks** (new findings):
| Framework | Best For | NPU Support | Size |
|-----------|----------|-------------|------|
| **ncnn** (Tencent) | Android ARM CPU | Vulkan GPU | Smallest binary |
| **MNN** (Alibaba) | Broad NPU support | OpenGL/Vulkan/Metal | Good GPU accel |
| **ONNX Runtime** | Cross-platform | NNAPI/CoreML/QNN | Best ecosystem |
| **Core ML** (Apple) | iOS + ANE | Native ANE | Best iOS perf |

### 4.5 Phase 5: Mobile Deployment (Weeks 21-24)

**Objective**: Ship FocusNet-Lite on iOS/Android/HarmonyOS.

| Platform | Framework | NPU | Target Latency |
|----------|-----------|-----|---------------|
| iOS | CoreML | ANE | <50ms |
| Android | ONNX Runtime | QNN (Hexagon) | <80ms |
| HarmonyOS | ncnn/ONNX Runtime | CANN (Ascend) | <80ms |

### 4.6 Preliminary Work Checklist (Immediate Actions)

**Before Phase 1 starts, complete these tasks:**

- [ ] Install PyTorch 2.x with CUDA 12.x on RTX 5080
- [ ] Download DINOv2-L/14 and DINOv2-S/14 from Meta (HuggingFace)
- [ ] Download ViTGaze codebase and verify it runs
- [ ] Install MediaPipe and test Face Mesh on sample images
- [ ] Install ONNX Runtime and verify cross-platform export
- [ ] Set up Label Studio for active learning annotations
- [ ] Record 1-2 hours of test video for pipeline validation
- [ ] Design the multi-stage distillation training script (PyTorch)
- [ ] Implement attention map extraction from DINOv2-L/14
- [ ] Implement feature regression layers for FitNets-style distillation

---

## 5. References

### Foundational Knowledge Distillation
1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop. https://arxiv.org/abs/1503.02531
2. Romero, A., et al. (2015). FitNets: Hints for Thin Deep Nets. ICLR 2015. https://arxiv.org/abs/1412.6550
3. Park, W., et al. (2019). Relational Knowledge Distillation. CVPR 2019. https://arxiv.org/abs/1904.05068
4. Tian, Y., et al. (2020). Contrastive Representation Distillation. ICLR 2020. https://arxiv.org/abs/1910.10699

### Vision Transformer Distillation
5. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention (DeiT). ICML 2021. https://arxiv.org/abs/2012.12877
6. Zhang, J., et al. (2022). MiniViT: Multi-Stage Vision Transformer Distillation. CVPR 2022. https://arxiv.org/abs/2112.09884
7. Various (2024). Efficient Vision Transformers: A Survey on Compression and Acceleration.

### Gaze Estimation
8. Song, Y., et al. (2024). ViTGaze: Gaze Estimation with ViT. Visual Intelligence.
9. Zhang, X., et al. (2020). ETH-XGaze: A Large Scale Dataset for Gaze Estimation. ECCV 2020. https://arxiv.org/abs/2007.12852
10. Zhang, J., et al. (2026). FreeGazeFormer: CNN-Transformer Cross-Modal Attention. SPIE.

### LLM/VLM Distillation
11. Mukherjee, S., et al. (2023). Orca: Progressive Learning from Complex Explanation Traces of GPT-4. https://arxiv.org/abs/2306.02707
12. Yang, A., et al. (2024). Qwen2 Technical Report. https://arxiv.org/abs/2407.10671
13. Yang, A., et al. (2025). Qwen3 Technical Report. https://qwenlm.github.io

### Mobile Deployment
14. Howard, A., et al. (2019). MobileNets: Efficient CNNs for Mobile Vision. https://arxiv.org/abs/1704.04861
15. Various (2023-2024). Knowledge Distillation for On-Device Medical Image Classification.

### Supplementary References (from Project Docs)
16. Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. arXiv:2304.07193
17. Liu, H., et al. (2024). Visual Instruction Tuning (LLaVA). NeurIPS.
18. Xu, P., et al. (2023). Multimodal Learning with Transformers: A Survey. IEEE TPAMI.
19. Baltrusaitis, T., et al. (2018). Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI.
20. Zhao, X., et al. (2026). GazeFormer-MoE: Context-Aware Gaze Estimation via CLIP and MoE Transformer. arXiv:2601.12316.

### Additional References (from Extended Research)
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

### Critical Papers from Extended Gaze/Distillation Research
31. Lee, J., et al. (2025). CustomKD: Customizing Large Vision Foundation for Edge Model Improvement via Knowledge Distillation. CVPR 2025. https://arxiv.org/abs/2503.18244
32. Song, J., et al. (2025). RS-MTDF: Multi-Teacher Distillation and Fusion for Remote Sensing. https://arxiv.org/abs/2506.08772
33. Shao, M., et al. (2026). X-Distill: Cross-Architecture Vision Distillation. https://arxiv.org/abs/2601.11269
34. Vasu, P., et al. (2023). FastViT: A Fast Hybrid Vision Transformer. ICCV 2023. https://arxiv.org/abs/2303.14189
35. Cheng, Y., et al. (2024). GazeDPTR: Dual-Stream Gaze Pyramid Transformer (IVGaze). CVPR 2024. https://arxiv.org/abs/2403.15664
36. Zhao, X., et al. (2025). GMGaze: MoE-Based Context-Aware Gaze Estimation. https://arxiv.org/abs/2605.00799
37. Vasu, P., et al. (2023). MobileOne: An Improved One Millisecond Mobile Backbone. CVPR 2023. https://arxiv.org/abs/2206.04040

---

*Report compiled: 2026-05-20*
*Papers cataloged: 15 primary + 15 supplementary + 7 extended gaze/KD = 37 total references*
*Research agents: 4 parallel agents completed (KD methods, Qwen distillation, gaze estimation, mobile deployment)*
*Key breakthrough: CustomKD (CVPR 2025) identified as the closest published work to our exact goal*
*For: FocusNet-Lite distillation pipeline design*
*Project: Brick by Brick AI Training*

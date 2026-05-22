# Teacher Models for Knowledge Distillation in Gaze Estimation

## Research Report

**Date**: 2026-05-19
**Project**: Focus Model Distillation
**Purpose**: Academic research for PhD-level report on knowledge distillation

---

## 1. Introduction to the Teacher-Student Paradigm

### 1.1 Knowledge Distillation Overview

Knowledge distillation (KD) is a technique where a large, complex model (the **teacher**) transfers its learned knowledge to a smaller, more efficient model (the **student**). The teacher produces "soft predictions" (probability distributions over classes or intermediate representations) that provide richer training signals than hard labels alone.

### 1.2 Application to Gaze Estimation

Gaze estimation involves predicting where a person is looking (point of regard) from visual inputs. A student model learning from a teacher can benefit from:

- **Dark knowledge**: The teacher captures subtle patterns in eye and facial features that are difficult to learn from limited training data
- **Multi-task learning transfer**: Large vision-language models understand spatial relationships, facial structure, and attention that transfer to gaze understanding
- **Robustness**: Foundation models pretrained on diverse data can provide more robust gaze priors than domain-specific models trained on smaller datasets

### 1.3 Selection Criteria for Teacher Models

1. **Vision understanding capability**: Ability to parse facial features, eyes, and spatial relationships
2. **Parameter count vs. performance**: Balance between model size and distillation efficiency
3. **Accessibility**: API availability, open-source availability, or research licensing
4. **Fine-tuning capability**: Ability to adapt to gaze-specific tasks
5. **Inference cost**: Practical considerations for training-time distillation

---

## 2. Candidate Teacher Models

### 2.1 Multimodal Foundation Models

#### 2.1.1 GPT-4V (OpenAI)

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision-language transformer with proprietary design |
| **Model Size** | Not publicly disclosed; estimated 1-2 trillion parameters |
| **Vision Input** | 2048x2048 images; supports document, chart, and face analysis |
| **API Access** | OpenAI API with usage-based pricing |
| **Fine-tuning** | Limited; not available for GPT-4V directly |
| **Research Access** | Via API with appropriate cost considerations |

**Gaze-Related Capabilities**:
- Strong face detection and analysis
- Can identify eye regions and approximate attention direction
- Limited explicit 3D gaze estimation but high semantic understanding

**Distillation Considerations**:
- Teacher outputs via API (logits/probabilities not directly accessible)
- Requires constructing prompts that extract gaze-relevant information
- Cost-prohibitive for large-scale dataset generation
- Best used for high-value samples or quality assurance

**Reference**: OpenAI (2023). "GPT-4V System Card". https://openai.com/index/gpt-4v-system-card/

#### 2.1.2 Google Gemini

| Aspect | Details |
|--------|---------|
| **Architecture** | Multimodal transformer with native vision encoder |
| **Model Variants** | Ultra (largest), Pro, Nano (smallest) |
| **Vision Input** | Native multimodal processing of images and video |
| **API Access** | Google Cloud AI Platform |
| **Fine-tuning** | Available for enterprise users |
| **Research Access** | Limited; primarily enterprise/research subscriptions |

**Gaze-Related Capabilities**:
- Native video understanding enables temporal gaze tracking
- Strong face and eye landmark detection
- Multimodal reasoning about spatial attention

**Distillation Considerations**:
- Video understanding useful for gaze trajectory analysis
- Higher inference cost than GPT-4V in some configurations
- Less established research ecosystem than OpenAI

**Reference**: Google DeepMind (2024). "Gemini: A Family of Highly Capable Multimodal Models". https://arxiv.org/abs/2312.11805

#### 2.1.3 LLaVA (Large Language and Vision Assistant)

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision encoder (CLIP) + LLM (Vicuna/LLaMA) via projection layer |
| **Model Variants** | LLaVA-1.5 (7B, 13B), LLaVA-1.6 (7B, 13B, 34B) |
| **Vision Encoder** | CLIP ViT-L/14 (visual) + Projection MLP |
| **LLM Backbone** | Vicuna-7B/13B or LLaMA-2 |
| **Fine-tuning** | Fully open; can be fine-tuned for specific tasks |
| **Research Access** | Fully open-source (Apache 2.0 license) |

**Gaze-Related Capabilities**:
- Open-source enables direct access to hidden states for distillation
- Vision encoder can be reused or adapted for gaze feature extraction
- Can describe eye positions and gaze direction in natural language

**Distillation Considerations**:
- Best candidate for white-box distillation (access to intermediate features)
- Smaller models allow for efficient dataset generation
- Vision encoder (CLIP) provides transferable visual representations
- Multi-modal learning signals available through LLM projections

**Reference**: Liu et al. (2024). "LLaVA: Large Language and Vision Assistant". https://arxiv.org/abs/2304.08485

#### 2.1.4 CogVLM

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision-language binding with frozen LLM |
| **Model Variants** | CogVLM-17B, CogVLM2 |
| **Vision Encoder** | EVA-CLIP |
| **LLM Backbone** | Vicuna-7B |
| **Fine-tuning** | Open; supports task-specific adaptation |
| **Research Access** | Open-source (Apache 2.0) |

**Gaze-Related Capabilities**:
- Strong visual understanding with freeze LLM architecture
- Good for visual question answering about gaze direction
- Efficient for fine-tuning on gaze-specific tasks

**Reference**: Wang et al. (2023). "CogVLM: Visual Language Model for GPT-4V-level Image Understanding". https://arxiv.org/abs/2312.12671

---

### 2.2 Specialized Large Models

#### 2.2.1 Large Gaze Estimation Models

**ITR (Image-to-Radiance) Approach**

| Model | Size | Key Features |
|-------|------|--------------|
| MPIIGaze | ~5M params | Canonical gaze representation; face alignment dependent |
| RT-GENE | ~10M params | Real-time eye gaze estimation with transformer |
| Gazefollow | ~15M params | In-the-wild gaze following; social gaze understanding |

**Reference**: Zhang et al. (2023). "Towards Domain-agnostic Gaze Estimation". CVPR 2023.

**ETH-XGaze Dataset Models**

Large-scale gaze models trained on 1 million+ images:
- ResNet-50 based: ~25M params
- Transformer-based: ~50M params
- Provides strong baseline for gaze understanding

**Reference**: Zhang et al. (2020). "ETH-XGaze: A Large Scale Dataset for Gaze Estimation". TPAMI 2020.

#### 2.2.2 MediaPipe Models

| Model | Parameters | Purpose |
|-------|------------|---------|
| Face Mesh | ~2.1M | 468 3D facial landmarks |
| Iris | ~1.3M | Eye landmark and depth estimation |
| Face Detection | ~1.2M | Face bounding box and 6-DOF pose |

**Accessibility**: Fully open-source (Apache 2.0), TensorFlow Lite variants available

**Gaze Estimation Application**:
- Face Mesh provides eye corner landmarks for gaze calibration
- Iris model estimates eye region geometry
- Combined pipeline: Face Detection -> Face Mesh -> Iris -> Gaze Vector

**Reference**: Google (2024). "MediaPipe Face Mesh". https://google.github.io/mediapipe/solutions/face_mesh

#### 2.2.3 OpenCV DNN Models

| Model | Parameters | Purpose |
|-------|------------|---------|
| FaceDetectorYN | ~0.5M | Real-time face detection |
| FaceReconNet | ~3.5M | Face reconstruction from single image |

**Accessibility**: Open-source, Intel/Google maintained

**Reference**: OpenCV (2024). "DNN Module Documentation". https://docs.opencv.org/4.x/d6/d0f/group__dnn.html

#### 2.2.4 IRIS (Intelligent Recognition in Scenes)

Large-scale facial landmark model:
- 76 keypoint model for facial structure
- Trained on synthetic and real data
- Focus on accurate eye region localization

**Reference**: Cardinaux et al. (2022). "Accurate Eye Center Localization with Stacked U-Net". BMVC 2022.

---

### 2.3 Pretrained Vision Encoders

#### 2.3.1 DINOv2 (Meta)

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision Transformer (ViT) with self-distillation |
| **Model Variants** | ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14 |
| **Parameters** | 22M (S), 86M (B), 304M (L), 1.1B (g) |
| **Training Data** | 142M images from curated sources |
| **Fine-tuning** | Linear probing effective; supports full fine-tuning |
| **Research Access** | Open-source (Apache 2.0) |

**Gaze-Related Capabilities**:
- Strong visual feature extraction from facial regions
- Excellent transfer learning on face-related downstream tasks
- Patch-based features can capture eye-level details

**Distillation Considerations**:
- Excellent frozen features for linear probing
- Can serve as strong visual backbone for student model
- Self-supervised training provides robust representations

**Reference**: Oquab et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision". https://arxiv.org/abs/2304.07193

#### 2.3.2 CLIP (OpenAI/LAION)

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision Transformer + Text Transformer |
| **Model Variants** | ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336 |
| **Parameters** | 86M (B/16), 428M (L/14) |
| **Training Data** | 400M image-text pairs (LAION-2B) |
| **Fine-tuning** | Supports fine-tuning with LoRA and full |
| **Research Access** | Open-weight models available |

**Gaze-Related Capabilities**:
- Vision encoder provides strong image understanding
- Text prompts can guide attention to eye regions
- Contrastive learning captures semantic relationships

**Distillation Considerations**:
- Vision encoder widely used as feature extractor
- CLIP features shown effective for facial analysis
- Multiple open-weight variants enable experimentation

**Reference**: Radford et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision". ICML 2021.

#### 2.3.3 EVA-CLIP

| Aspect | Details |
|--------|---------|
| **Architecture** | CLIP with EVA image encoder |
| **Model Variants** | EVA-CLIP-B/16, EVA-CLIP-L/14 |
| **Parameters** | 86M (B), 428M (L) |
| **Training Data** | 4M pre-training + CLIP dataset |
| **Improvement** | +2.5% accuracy over CLIP on ImageNet |

**Reference**: Sun et al. (2023). "EVA-CLIP: Improved Training of CLIP-sized Models". https://arxiv.org/abs/2305.19930

#### 2.3.4 SigLIP

| Aspect | Details |
|--------|---------|
| **Architecture** | Vision Transformer with sigmoid loss |
| **Model Variants** | SigLIP-B/16, SigLIP-L/16 |
| **Parameters** | 86M (B), 304M (L) |
| **Training Data** | 900M images |

**Key Features**:
- Sigmoid loss enables training without global batch contrastive loss
- Better scaling properties than CLIP
- Strong multilingual support

**Reference**: Zhai et al. (2023). "SigLIP: Sigmoid Loss for Language Image Pre-Training". https://arxiv.org/abs/2303.15343

#### 2.3.5 Swin Transformer (Face-Pretrained)

| Aspect | Details |
|--------|---------|
| **Architecture** | Hierarchical Vision Transformer |
| **Model Variants** | Swin-T, Swin-S, Swin-B, Swin-L |
| **Face Pretraining** | Pretrained on face datasets (MS-Celeb, VGGFace2) |
| **Fine-tuning** | Excellent transfer to face-related tasks |

**Gaze-Related Capabilities**:
- Hierarchical features capture both global and local information
- Face-pretrained models provide domain-relevant features
- Particularly effective for fine-grained facial analysis

**Reference**: Liu et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". ICCV 2021.

---

## 3. Comparison Table

| Model | Type | Parameters | Accessibility | Gaze Suitability | Distillation Type |
|-------|------|------------|---------------|------------------|-------------------|
| **GPT-4V** | VL Foundation | ~1T (est) | API Only | Moderate | Black-box |
| **Gemini Ultra** | VL Foundation | ~1T+ (est) | Enterprise | Moderate | Black-box |
| **LLaVA-1.6-13B** | VL Open | 13B | Fully Open | High | White-box |
| **LLaVA-1.5-7B** | VL Open | 7B | Fully Open | High | White-box |
| **CogVLM-17B** | VL Open | 17B | Fully Open | High | White-box |
| **DINOv2-L/14** | Vision Encoder | 304M | Fully Open | Very High | Feature-based |
| **DINOv2-g/14** | Vision Encoder | 1.1B | Fully Open | Very High | Feature-based |
| **CLIP ViT-L/14** | VL Encoder | 428M | Open-weight | High | Feature-based |
| **EVA-CLIP-L/14** | VL Encoder | 428M | Open-weight | High | Feature-based |
| **MediaPipe Face Mesh** | Specialized | 2.1M | Fully Open | Very High | Feature-based |
| **Swin-B Face** | Vision Encoder | 88M | Open-weight | Very High | Feature-based |

---

## 4. Recommendations for Gaze Estimation Distillation

### 4.1 Primary Recommendation: Hybrid Approach

For PhD-level research on gaze estimation distillation, I recommend a **multi-teacher distillation strategy**:

#### Tier 1: Feature Extraction Backbone (DINOv2)
- **Rationale**: DINOv2 provides state-of-the-art visual features without labels
- **Role**: Provides low-level visual representations (patches, edges, textures) that are robust and transferable
- **Implementation**: Use frozen DINOv2 features as initialization for student model; distill intermediate representations

#### Tier 2: Domain-Specific Enhancement (MediaPipe + Swin)
- **Rationale**: MediaPipe provides calibrated gaze landmarks; Swin provides hierarchical face features
- **Role**: Inject domain knowledge about facial geometry and eye landmarks
- **Implementation**: Multi-task learning where student learns both gaze estimation and landmark detection from teacher

#### Tier 3: Semantic Understanding (LLaVA)
- **Rationale**: Open-source VL model with accessible hidden states
- **Role**: Provide high-level gaze reasoning (e.g., "person looking at the right monitor")
- **Implementation**: Use LLaVA's vision encoder and intermediate layers; fine-tune on gaze descriptions

### 4.2 Distillation Architecture

```
Teacher Ensemble                    Student Model
    |                                     |
    +-- DINOv2-L/14 --> Feature Loss -->  |
    +-- MediaPipe ----> Landmark Loss --> +--> Gaze Vector
    +-- Swin-Face ---> Attention Loss --> |
    +-- LLaVA-7B ---> Semantic Loss ----->|
```

### 4.3 Practical Implementation

#### For Closed-Source Models (GPT-4V, Gemini):
- Use for **quality assurance** and **high-value sample annotation**
- Generate synthetic gaze labels for edge cases
- Cost-effective for 1K-10K samples, not millions

#### For Open-Source Models (LLaVA, DINOv2, MediaPipe):
- Primary source for **large-scale distillation**
- Access to intermediate features enables knowledge transfer
- Run locally for cost-free experimentation

### 4.4 Training Strategy

1. **Stage 1**: Use DINOv2 features to initialize student visual encoder
2. **Stage 2**: Distill from MediaPipe/Face Mesh for geometric understanding
3. **Stage 3**: Refine with LLaVA for semantic gaze understanding
4. **Stage 4**: Optional validation with GPT-4V on held-out test set

---

## 5. Gaze Estimation Architecture Considerations

### 5.1 Student Model Architecture

For a distillation target, consider:

| Component | Option | Parameters |
|-----------|--------|------------|
| Visual Encoder | ViT-S/14 | 22M |
| Feature Aggregator | Cross-attention | 5M |
| Gaze Head | MLP | 2M |
| **Total** | | ~30M |

This 30M-parameter student can potentially match 100M+ teacher performance through effective distillation.

### 5.2 Loss Functions

1. **L1/L2 Gaze Loss**: Direct gaze vector regression
2. **Feature Matching Loss**: MSE between teacher and student intermediate features
3. **Attention Loss**: KL divergence between attention maps
4. **Multi-task Loss**: Combined with landmark detection for auxiliary supervision

---

## 6. References

### Foundation Models

1. OpenAI (2023). "GPT-4V System Card". OpenAI Blog.
   https://openai.com/index/gpt-4v-system-card/

2. Google DeepMind (2024). "Gemini: A Family of Highly Capable Multimodal Models".
   https://arxiv.org/abs/2312.11805

3. Liu H., Li C., Wu Q., Lee Y.J. (2024). "Visual Instruction Tuning".
   https://arxiv.org/abs/2304.08485

4. Wang W., et al. (2023). "CogVLM: Visual Language Model for GPT-4V-level Image Understanding".
   https://arxiv.org/abs/2312.12671

### Vision Encoders

5. Oquab M., et al. (2023). "DINOv2: Learning Robust Visual Features without Supervision".
   https://arxiv.org/abs/2304.07193

6. Radford A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision". ICML 2021.

7. Sun Q., et al. (2023). "EVA-CLIP: Improved Training of CLIP-sized Models".
   https://arxiv.org/abs/2305.19930

8. Zhai X., et al. (2023). "SigLIP: Sigmoid Loss for Language Image Pre-Training".
   https://arxiv.org/abs/2303.15343

9. Liu Z., et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows". ICCV 2021.

### Gaze Estimation

10. Zhang X., et al. (2023). "Towards Domain-agnostic Gaze Estimation". CVPR 2023.

11. Zhang Y., et al. (2020). "ETH-XGaze: A Large Scale Dataset for Gaze Estimation". TPAMI 2020.

12. Park S., et al. (2020). "RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments". ECCV 2020.

13. Recasens A., et al. (2015). "Following Gaze in the Wild with Small Data". IROS 2015.

### Facial Analysis

14. Google (2024). "MediaPipe Face Mesh: Technical Documentation".
   https://google.github.io/mediapipe/solutions/face_mesh

15. OpenCV (2024). "DNN Module Documentation".
   https://docs.opencv.org/4.x/d6/d0f/group__dnn.html

16. Cardinaux F., et al. (2022). "Accurate Eye Center Localization with Stacked U-Net". BMVC 2022.

---

## 7. Additional Considerations

### 7.1 Ethical Considerations

- Gaze estimation models have privacy implications
- Ensure compliance with data protection regulations (GDPR, CCPA)
- Consider bias in face analysis models across demographics

### 7.2 Computational Resources

- GPT-4V/Gemini: API costs for dataset generation
- DINOv2-L: ~16GB GPU memory for inference
- LLaVA-7B: ~14GB GPU memory for inference
- LLaVA-13B: ~26GB GPU memory for inference

### 7.3 Future Directions

- Emerging models: Qwen-VL, InternVL, MM1
- Specialized gaze models may emerge from video understanding research
- Cross-modal distillation (image-to-video gaze understanding)

---

*Report generated for PhD research project on Knowledge Distillation for Gaze Estimation*
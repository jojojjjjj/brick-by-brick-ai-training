# Model Distillation Literature for Focus Detection Model (A-Level Venues Only)
# 专注度检测模型知识蒸馏文献集（仅含A级会议/期刊论文）

**Curated / 整理日期**: 2026-05-20
**Project / 项目**: Brick by Brick AI Training / FocusNet-Lite
**Purpose / 用途**: Knowledge distillation research for mobile gaze/focus detection model
专注度检测移动端模型的知识蒸馏研究

---

## Paper Catalog / 论文目录 (16 Papers / 16篇论文)

### Category 1: Foundation Knowledge Distillation Methods / 类别一：基础知识蒸馏方法

---

### Paper 1: Distilling the Knowledge in a Neural Network / 神经网络中的知识蒸馏

- **Authors / 作者**: Hinton, G., Vinyals, O., & Dean, J.
- **Year / 年份**: 2015
- **Venue / 发表刊物**: NeurIPS 2015 Workshop (顶级会议 NeurIPS 工作坊，知识蒸馏奠基之作)
- **Link / 链接**: https://arxiv.org/abs/1503.02531

**Summary / 摘要**:

The foundational paper introducing knowledge distillation. Proposed using soft targets from a large teacher model (with temperature scaling) to train a smaller student model. The "dark knowledge" in soft probability distributions provides richer training signals than hard labels. Key formula: L_total = alpha * L_soft + (1-alpha) * L_hard, where L_soft uses KL divergence with temperature T.

本文是知识蒸馏领域的奠基性工作，首次提出了知识蒸馏的概念。论文建议使用大型教师模型（配合温度缩放 temperature scaling）生成的软目标（soft targets）来训练较小的学生模型。软概率分布中蕴含的"暗知识（dark knowledge）"比硬标签提供了更丰富的训练信号。核心公式为：L_total = alpha * L_soft + (1-alpha) * L_hard，其中 L_soft 使用温度为 T 的 KL 散度（KL divergence）计算。

**Relevance to FocusNet / 与FocusNet的关联**:

Defines the core KD framework we will use. Temperature scheduling and alpha balancing are directly applicable to our multi-teacher distillation pipeline.

本文定义了我们将要使用的核心知识蒸馏框架。温度调度（temperature scheduling）和 alpha 平衡策略可直接应用于我们的多教师蒸馏流程（multi-teacher distillation pipeline）。

---

### Paper 2: FitNets: Hints for Thin Deep Nets / FitNets：面向窄而深网络的提示学习

- **Authors / 作者**: Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., & Bengio, Y.
- **Year / 年份**: 2015
- **Venue / 发表刊物**: ICLR 2015 (顶级会议 ICLR，特征级蒸馏开创性工作)
- **Link / 链接**: https://arxiv.org/abs/1412.6550

**Summary / 摘要**:

Extended knowledge distillation to include intermediate feature-level hints from teacher to student. Demonstrated that matching intermediate representations (not just output logits) allows deeper, thinner student networks to learn effectively. Introduced the concept of feature regression layers to align teacher/student feature dimensions.

本文将知识蒸馏扩展到包含从教师到学生的中间层特征提示（intermediate feature-level hints）。论文证明了匹配中间层表征（而非仅仅匹配输出 logits）可以使更深、更窄的学生网络有效学习。论文引入了特征回归层（feature regression layers）的概念，用于对齐教师和学生之间的特征维度。

**Relevance to FocusNet / 与FocusNet的关联**:

Our distillation strategy uses feature-level distillation from DINOv2-L to ViT-S. FitNets' feature matching approach directly informs our L_feature loss.

我们的蒸馏策略使用了从 DINOv2-L 到 ViT-S 的特征级蒸馏（feature-level distillation）。FitNets 的特征匹配方法直接启发了我们的 L_feature 损失函数设计。

---

### Paper 3: Relational Knowledge Distillation / 关系知识蒸馏

- **Authors / 作者**: Park, W., Kim, D., Lu, Y., & Cho, M.
- **Year / 年份**: 2019
- **Venue / 发表刊物**: CVPR 2019 (顶级会议 CVPR，关系级蒸馏开创性工作)
- **Link / 链接**: https://arxiv.org/abs/1904.05068

**Summary / 摘要**:

Proposed distilling structural relationships between data points rather than individual predictions. Uses distance-wise and angle-wise relational losses to preserve the geometric structure of the teacher's representation space in the student model.

本文提出了蒸馏数据点之间的结构关系（structural relationships）而非单个预测结果的方法。使用基于距离（distance-wise）和基于角度（angle-wise）的关系损失函数，在学生模型中保持教师表征空间的几何结构（geometric structure）。

**Relevance to FocusNet / 与FocusNet的关联**:

When distilling from multiple teachers (DINOv2, MediaPipe, LLaVA), preserving relational structures across different feature spaces ensures coherent multi-modal fusion in the student.

当从多个教师模型（DINOv2、MediaPipe、LLaVA）进行蒸馏时，保持不同特征空间中的关系结构可以确保学生模型中多模态融合（multi-modal fusion）的连贯性。

---

### Paper 4: Contrastive Representation Distillation (CRD) / 对比表征蒸馏

- **Authors / 作者**: Tian, Y., Krishnan, D., & Isola, P.
- **Year / 年份**: 2020
- **Venue / 发表刊物**: ICLR 2020 (顶级会议 ICLR，对比学习与蒸馏的结合)
- **Link / 链接**: https://arxiv.org/abs/1910.10699

**Summary / 摘要**:

Framed knowledge distillation as a contrastive learning problem. Uses noise contrastive estimation to maximize mutual information between teacher and student representations. Outperforms KD and FitNets on multiple benchmarks by capturing more transferable knowledge.

本文将知识蒸馏框架化为一个对比学习（contrastive learning）问题。使用噪声对比估计（noise contrastive estimation）来最大化教师和学生表征之间的互信息（mutual information）。通过捕获更多可迁移的知识，在多个基准测试上超越了传统 KD 和 FitNets。

**Relevance to FocusNet / 与FocusNet的关联**:

CRD's contrastive framework can be applied to align DINOv2's self-supervised features with the student's learned representations, improving feature transfer quality.

CRD 的对比学习框架可用于对齐 DINOv2 的自监督特征（self-supervised features）与学生模型学习到的表征，从而提升特征迁移质量。

---

### Category 2: Vision Transformer Distillation / 类别二：Vision Transformer 蒸馏

---

### Paper 5: Training Data-Efficient Image Transformers & Distillation through Attention (DeiT) / 数据高效的图像Transformer训练与基于注意力的蒸馏

- **Authors / 作者**: Touvron, H., Cordova, M., Sablayrolles, A., Douze, M., & Jegou, H.
- **Year / 年份**: 2021
- **Venue / 发表刊物**: ICML 2021 (顶级会议 ICML，ViT 蒸馏里程碑工作)
- **Link / 链接**: https://arxiv.org/abs/2012.12877

**Summary / 摘要**:

Introduced the distillation token approach for Vision Transformers. A special [DIST] token is added alongside [CLS] in the ViT, trained to predict the teacher's (RegNet) hard labels via attention. Achieved 85.2% top-1 on ImageNet using only ImageNet-1K data, demonstrating ViTs don't need massive pretraining datasets when paired with distillation.

本文引入了针对 Vision Transformer 的蒸馏令牌（distillation token）方法。在 ViT 中与 [CLS] 令牌一起添加了一个特殊的 [DIST] 令牌，通过注意力机制（attention）训练预测教师模型（RegNet）的硬标签。仅使用 ImageNet-1K 数据即在 ImageNet 上达到 85.2% 的 top-1 准确率，证明了 ViT 在配合蒸馏时不需要大规模预训练数据集。

**Relevance to FocusNet / 与FocusNet的关联**:

Directly applicable architecture modification. We can add a distillation token to FocusNet-Lite's ViT-S to learn from DINOv2-L's predictions through attention, similar to DeiT's approach.

直接可用的架构修改方案。我们可以在 FocusNet-Lite 的 ViT-S 中添加蒸馏令牌（distillation token），通过注意力机制从 DINOv2-L 的预测中学习，类似于 DeiT 的方法。

---

### Paper 6: MiniViT: Multi-Stage Vision Transformer Distillation / MiniViT：多阶段 Vision Transformer 蒸馏

- **Authors / 作者**: Zhang, J., Peng, H., Wu, K., Liu, M., Xiao, B., Fu, J., & Yuan, L.
- **Year / 年份**: 2022
- **Venue / 发表刊物**: CVPR 2022 (顶级会议 CVPR，多阶段 ViT 蒸馏)
- **Link / 链接**: https://arxiv.org/abs/2112.09884

**Summary / 摘要**:

Proposes multi-stage self-attention distillation for compressing Vision Transformers. Uses weight sharing across transformer blocks and multi-level feature distillation (attention maps, [CLS] token, patch tokens). MiniViT-24M achieves ~83% accuracy, rivaling DeiT-B with 4x fewer parameters.

本文提出了用于压缩 Vision Transformer 的多阶段自注意力蒸馏（multi-stage self-attention distillation）方法。使用跨 Transformer 块的权重共享（weight sharing）以及多层特征蒸馏（注意力图、[CLS] 令牌、patch 令牌）。MiniViT-24M 达到约 83% 的准确率，与参数量为其四倍的 DeiT-B 相当。

**Relevance to FocusNet / 与FocusNet的关联**:

Directly demonstrates that ViT-S/14 (22M params) can effectively learn from ViT-L/14 (304M params) through multi-stage attention distillation - exactly our teacher-student configuration.

直接证明了 ViT-S/14（2200万参数）可以通过多阶段注意力蒸馏（multi-stage attention distillation）从 ViT-L/14（3.04亿参数）中有效学习——这正是我们的教师-学生配置。

---

### Category 3: Gaze Estimation with Transformers / 类别三：基于Transformer的注视估计

---

### Paper 7: ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose / ETH-XGaze：面向极端头部姿态下注视估计的大规模数据集

- **Authors / 作者**: Zhang, X., Park, S., Beeler, T., Bradley, D., Tang, S., & McNamara, S.
- **Year / 年份**: 2020
- **Venue / 发表刊物**: ECCV 2020 (顶级会议 ECCV，注视估计基准数据集)
- **Link / 链接**: https://arxiv.org/abs/2007.12852

**Summary / 摘要**:

Created a 1.1M+ image dataset for gaze estimation with extreme head poses. Found that models trained on standard datasets (MPIIGaze) fail on extreme poses, motivating multi-modal approaches that fuse gaze and head pose signals.

创建了一个包含超过110万张图像的极端头部姿态注视估计数据集。研究发现，在标准数据集（MPIIGaze）上训练的模型在极端姿态下表现不佳，促使研究者采用融合注视和头部姿态信号的多模态方法（multi-modal approaches）。

**Relevance to FocusNet / 与FocusNet的关联**:

Validates the need for our cross-attention fusion module that combines gaze and head pose. Also provides benchmark performance targets.

验证了我们用于结合注视和头部姿态的交叉注意力融合模块（cross-attention fusion module）的必要性。同时提供了基准性能目标。

---

### Paper 12: GazeDPTR - Comprehensive Vision Solution for In-Vehicle Gaze Estimation / GazeDPTR：车载注视估计的综合视觉方案

- **Authors / 作者**: Cheng, Y., Zhu, Y., Wang, Z., Hao, H., et al.
- **Year / 年份**: 2024
- **Venue / 发表刊物**: CVPR 2024 (顶级会议 CVPR，多流注视估计)
- **Link / 链接**: https://arxiv.org/abs/2403.15664

**Summary / 摘要**:

Proposes a comprehensive vision-based solution for in-vehicle gaze estimation. Introduces the IVGaze dataset (125 subjects), a dual-stream gaze pyramid transformer (GazeDPTR), and a novel gaze zone classification strategy. The gaze pyramid transformer addresses low-resolution face images by integrating transformer-based multilevel features, while the dual-stream design employs perspective transformation to normalize images via virtual camera rotation and fuses them with original images.

提出了一套完整的基于视觉的车内视线估计解决方案。包含三个关键贡献：IVGaze 数据集（125 名受试者）、dual-stream gaze pyramid transformer (GazeDPTR) 以及新颖的视线区域分类策略。Gaze pyramid transformer 通过整合基于 transformer 的多层级特征来解决低分辨率人脸图像问题，而 dual-stream 设计采用 perspective transformation 通过虚拟摄像机旋转归一化图像并与原始图像融合。

**Relevance to FocusNet / 与FocusNet的关联**:

Multi-stream fusion architecture and pyramid transformer's multi-level feature integration provide architectural insights for FocusNet-Lite's student model design. The dual-stream fusion strategy (normalized + original images) can be adapted for our cross-attention fusion module.

多流融合架构和 pyramid transformer 的多层级特征整合为 FocusNet-Lite 的学生模型设计提供了架构层面的启发。其 dual-stream 融合策略（归一化图像 + 原始图像）可适配到我们的交叉注意力融合模块中。

---

### Category 4: Additional KD Methods / 类别四：其他知识蒸馏方法

---

### Paper 8: Paying More Attention to Attention (Attention Transfer) / 更加关注注意力：注意力迁移

- **Authors / 作者**: Zagoruyko, S. & Komodakis, N.
- **Year / 年份**: 2017
- **Venue / 发表刊物**: ICLR 2017 (顶级会议 ICLR，注意力迁移开创性工作)
- **Link / 链接**: https://arxiv.org/abs/1612.03928

**Summary / 摘要**:

Introduces Attention Transfer (AT) - distilling by matching spatial attention maps (sum of squared activations across channels). Student learns WHERE the teacher focuses, not just WHAT it predicts. Simple to implement and combines well with soft-target KD.

本文引入了注意力迁移（Attention Transfer, AT）方法——通过匹配空间注意力图（spatial attention maps，即跨通道的激活值平方和）进行蒸馏。学生模型学习教师模型关注"哪里"（WHERE），而不仅仅是预测"什么"（WHAT）。实现简单，且与软目标 KD（soft-target KD）配合效果良好。

**Relevance to FocusNet / 与FocusNet的关联**:

Directly applicable for distilling DINOv2's attention maps (which naturally track object boundaries) to FocusNet-Lite's attention layers.

直接适用于将 DINOv2 的注意力图（天然跟踪物体边界）蒸馏到 FocusNet-Lite 的注意力层中。

---

### Paper 9: TinyViT - Fast Pretraining Distillation for Small ViTs / TinyViT：面向小型ViT的快速预训练蒸馏

- **Authors / 作者**: Wu, K., Zhang, J., Peng, H., et al. (Microsoft)
- **Year / 年份**: 2022 (arXiv), ECCV 2024 (正式发表)
- **Venue / 发表刊物**: ECCV 2024 (顶级会议 ECCV，小型 ViT 蒸馏)
- **Link / 链接**: https://arxiv.org/abs/2207.10666

**Summary / 摘要**:

One-stage distillation from large pretrained ViTs (including CLIP) into very small ViTs (5M-21M params). Demonstrates small ViTs can be highly competitive when distilled from strong teachers. Directly applicable to mobile/edge deployment.

从大型预训练 ViT（包括 CLIP）到超小型 ViT（500万至2100万参数）的单阶段蒸馏（one-stage distillation）。证明了小型 ViT 在从强教师模型蒸馏后可以具有极高的竞争力。可直接应用于移动/边缘端部署（mobile/edge deployment）。

**Relevance to FocusNet / 与FocusNet的关联**:

Demonstrates that even 5M-param ViTs can be effective with proper distillation - relevant for extreme mobile constraints.

证明了即使仅有500万参数的 ViT 也能通过适当的蒸馏变得高效——这对于极端移动端约束（extreme mobile constraints）非常相关。

---

### Paper 13: CustomKD - Customizing Large Vision Foundation for Edge Model Improvement / CustomKD：定制大型视觉基础模型以提升边缘端模型性能

- **Authors / 作者**: Lee, J., Das, D., Hayat, M., Choi, S., Hwang, K., & Porikli, F.
- **Year / 年份**: 2025
- **Venue / 发表刊物**: CVPR 2025 (顶级会议 CVPR，基础模型到边缘端蒸馏)
- **Link / 链接**: https://arxiv.org/abs/2503.18244

**Summary / 摘要**:

Proposes a method to customize large vision foundation models (specifically DINOv2) for edge device improvement through knowledge distillation. Addresses the challenge of transferring rich self-supervised features from billion-parameter foundation models to lightweight edge models. Introduces custom distillation strategies that preserve the most transferable representations while adapting to edge constraints.

提出了一种通过知识蒸馏定制大型视觉基础模型（特别是 DINOv2）以提升边缘端设备性能的方法。解决了将数十亿参数基础模型中丰富的自监督特征（self-supervised features）迁移到轻量级边缘模型的挑战。引入了定制蒸馏策略，在适应边缘端约束的同时保持最具可迁移性的表征。

**Relevance to FocusNet / 与FocusNet的关联**:

**Most directly relevant paper to our project.** CustomKD specifically addresses DINOv2-to-edge distillation — exactly our pipeline from DINOv2-L teacher to ViT-S student. Their customization strategies can be directly adopted for FocusNet-Lite's distillation pipeline.

**与本项目关联度最高的论文。** CustomKD 专门解决 DINOv2 到边缘端的蒸馏问题——这正是我们从 DINOv2-L 教师到 ViT-S 学生的流程。其定制化策略可直接应用于 FocusNet-Lite 的蒸馏流程。

---

### Category 5: Comprehensive Surveys / 类别五：综合综述

---

### Paper 10: Knowledge Distillation Survey (Gou et al., 2021) / 知识蒸馏综述

- **Authors / 作者**: Gou, J., Yu, B., Maybank, S.J., & Tao, D.
- **Year / 年份**: 2021
- **Venue / 发表刊物**: IJCV 2021 (顶级期刊 International Journal of Computer Vision，知识蒸馏最全面综述)
- **Link / 链接**: https://arxiv.org/abs/2009.12538

**Summary / 摘要**:

Most comprehensive KD survey. Taxonomy by knowledge type (response/feature/relation), distillation scheme (teacher-student, self, online, multi-teacher), and student strategy. Essential reading for choosing the right approach.

最全面的知识蒸馏综述。按照知识类型（响应/特征/关系，即 response/feature/relation）、蒸馏方案（教师-学生、自蒸馏、在线蒸馏、多教师蒸馏，即 teacher-student/self/online/multi-teacher）以及学生策略进行分类。是选择合适蒸馏方法的必读文献。

**Relevance to FocusNet / 与FocusNet的关联**:

Provides the definitive reference framework for designing our multi-teacher distillation strategy.

为我们设计多教师蒸馏策略（multi-teacher distillation strategy）提供了权威的参考框架。

---

### Category 6: LLM Distillation Case Studies / 类别六：大语言模型蒸馏案例研究

---

### Paper 11: LIMA - Less Is More for Alignment / LIMA：对齐中的少即是多

- **Authors / 作者**: Zhou, C., Liu, P., et al. (Meta/CMU)
- **Year / 年份**: 2023
- **Venue / 发表刊物**: NeurIPS 2023 (顶级会议 NeurIPS，高质量数据对齐)
- **Link / 链接**: https://arxiv.org/abs/2305.11206

**Summary / 摘要**:

Only 1,000 carefully curated instruction-response pairs needed for strong alignment. Superficial Alignment Hypothesis: model knowledge comes from pretraining; alignment teaches style/format. GPT-4 preferred over LIMA only ~43% of the time.

仅需要1000个精心筛选的指令-响应对即可实现出色的对齐效果。表面对齐假设（Superficial Alignment Hypothesis）：模型的知识来自预训练（pretraining），对齐（alignment）只教授风格和格式。GPT-4 相比 LIMA 的偏好率仅为约 43%。

**Relevance to FocusNet / 与FocusNet的关联**:

Strongly validates quality over quantity approach. Our active learning should focus on curating fewer, higher-quality samples.

有力验证了"质量优先于数量"的方法。我们的主动学习（active learning）应专注于筛选更少但更高质量的样本。

---

### Category 7: Mobile/Edge Deployment / 类别七：移动端/边缘端部署

---

### Paper 14: Searching for MobileNetV3 / MobileNetV3 搜索

- **Authors / 作者**: Howard, A. G., Sandler, M., Chu, G., et al. (Google)
- **Year / 年份**: 2019
- **Venue / 发表刊物**: CVPR 2019 (顶级会议 CVPR，移动端NAS+蒸馏)
- **Link / 链接**: https://arxiv.org/abs/1905.02244

**Summary / 摘要**:

Uses Neural Architecture Search (NAS) combined with knowledge distillation to design efficient mobile models. Employs depthwise separable convolutions, squeeze-and-excitation blocks, and hardware-aware NAS. MobileNetV3-Small achieves competitive accuracy at approximately 2.5M parameters, making it one of the most efficient mobile architectures.

使用神经架构搜索（Neural Architecture Search, NAS）结合知识蒸馏来设计高效的移动端模型。采用深度可分离卷积（depthwise separable convolutions）、挤压激励模块（squeeze-and-excitation blocks）以及硬件感知 NAS。MobileNetV3-Small 以约250万参数实现了具有竞争力的精度，是最高效的移动端架构之一。

**Relevance to FocusNet / 与FocusNet的关联**:

MobileNetV3-Small is our fallback backbone if ViT-S proves too expensive for certain NPUs. The NAS + distillation methodology validates the combined compression approach we plan for FocusNet-Lite.

MobileNetV3-Small 是当 ViT-S 在某些 NPU 上过于昂贵时的备选骨干网络。其 NAS + 蒸馏方法验证了我们为 FocusNet-Lite 计划的组合压缩方案。

---

### Paper 15: FastViT - A Fast Hybrid Vision Transformer / FastViT：快速混合视觉Transformer

- **Authors / 作者**: Vasu, P.K.A., Gabriel, J., Zhu, J., Tuzel, O., et al. (Apple)
- **Year / 年份**: 2023
- **Venue / 发表刊物**: ICCV 2023 (顶级会议 ICCV，结构重参数化混合架构)
- **Link / 链接**: https://arxiv.org/abs/2303.14189

**Summary / 摘要**:

Introduces a fast hybrid Vision Transformer using structural reparameterization. Combines CNN-like local processing with Transformer-based global attention, achieving state-of-the-art accuracy-speed tradeoff. At inference, the architecture collapses into an efficient form through reparameterization, eliminating the computational overhead of the training-time architecture.

引入了一种使用结构重参数化（structural reparameterization）的快速混合 Vision Transformer。将类 CNN 的局部处理与基于 Transformer 的全局注意力相结合，实现了最先进的精度-速度平衡。在推理时，架构通过重参数化折叠为高效形式，消除了训练时架构的计算开销。

**Relevance to FocusNet / 与FocusNet的关联**:

FastViT's hybrid CNN-Transformer design is a strong candidate for FocusNet-Lite's student backbone. The structural reparameterization technique could enable us to maintain training-time expressiveness while achieving deployment-time efficiency.

FastViT 的混合 CNN-Transformer 设计是 FocusNet-Lite 学生骨干网络的强有力候选。结构重参数化技术可以使我们在保持训练时表达能力的同时实现部署时的高效性。

---

### Paper 16: MobileOne - An Improved One Millisecond Mobile Backbone / MobileOne：改进的亚毫秒移动端骨干网络

- **Authors / 作者**: Vasu, P.K.A., Gabriel, J., Zhu, J., Tuzel, O., et al. (Apple)
- **Year / 年份**: 2023
- **Venue / 发表刊物**: CVPR 2023 (顶级会议 CVPR，亚毫秒推理)
- **Link / 链接**: https://arxiv.org/abs/2206.04040

**Summary / 摘要**:

Proposes an improved mobile backbone achieving sub-millisecond inference on mobile devices through structural reparameterization. Uses over-parameterized training architecture that collapses to an efficient inference architecture. MobileOne-S0 achieves 71.4% top-1 on ImageNet with only 2.1M parameters and sub-ms latency on iPhone 12.

提出了一种改进的移动端骨干网络，通过结构重参数化（structural reparameterization）在移动设备上实现亚毫秒推理。使用过参数化的训练架构（over-parameterized training architecture），在推理时折叠为高效架构。MobileOne-S0 以仅210万参数在 ImageNet 上实现 71.4% 的 top-1 精度，在 iPhone 12 上的延迟低于1毫秒。

**Relevance to FocusNet / 与FocusNet的关联**:

MobileOne represents the extreme end of mobile efficiency. If FocusNet-Lite needs to run on resource-constrained devices (wearables, older phones), MobileOne's architecture provides the blueprint for sub-ms inference with minimal accuracy loss.

MobileOne 代表了移动端效率的极致。如果 FocusNet-Lite 需要在资源受限的设备（可穿戴设备、旧款手机）上运行，MobileOne 的架构提供了以最小精度损失实现亚毫秒推理的蓝图。

---

## Key Papers Summary Table / 核心论文总结表

| # | Paper / 论文 | Year / 年份 | Venue / 发表刊物 | Key Contribution / 核心贡献 | FocusNet Relevance / FocusNet关联性 |
|---|-------------|-------------|-----------------|---------------------------|-----------------------------------|
| 1 | Hinton KD / Hinton知识蒸馏 | 2015 | NeurIPS Workshop | Foundational KD framework / 基础知识蒸馏框架 | Core distillation formula / 核心蒸馏公式 |
| 2 | FitNets / 特征提示蒸馏 | 2015 | ICLR | Feature-level hints / 特征级提示 | Intermediate feature matching / 中间层特征匹配 |
| 3 | RKD / 关系知识蒸馏 | 2019 | CVPR | Relational distillation / 关系蒸馏 | Multi-teacher alignment / 多教师对齐 |
| 4 | CRD / 对比表征蒸馏 | 2020 | ICLR | Contrastive distillation / 对比蒸馏 | Feature space alignment / 特征空间对齐 |
| 5 | DeiT / 数据高效ViT蒸馏 | 2021 | ICML | Distillation token / 蒸馏令牌 | Architecture modification / 架构修改 |
| 6 | MiniViT / 多阶段ViT蒸馏 | 2022 | CVPR | Multi-stage ViT distillation / 多阶段ViT蒸馏 | DINOv2-L到ViT-S的蒸馏流程 |
| 7 | ETH-XGaze / 极端姿态注视数据集 | 2020 | ECCV | Extreme pose benchmark / 极端姿态基准 | Multi-modal fusion validation / 多模态融合验证 |
| 8 | Attention Transfer / 注意力迁移 | 2017 | ICLR | Spatial attention maps / 空间注意力图 | DINOv2注意力蒸馏 / DINOv2 attention distillation |
| 9 | TinyViT / 小型ViT蒸馏 | 2024 | ECCV | Small ViT distillation / 小型ViT蒸馏 | 5M参数移动端部署 / 极端移动端约束 |
| 10 | KD Survey / KD综述 (Gou) | 2021 | IJCV | Comprehensive taxonomy / 全面分类体系 | Strategy selection reference / 策略选择参考 |
| 11 | LIMA / 少即是多对齐 | 2023 | NeurIPS | Quality over quantity / 质量优于数量 | Active learning design / 主动学习设计 |
| 12 | GazeDPTR / 多流注视估计 | 2024 | CVPR | Multi-stream gaze transformer / 多流注视估计 transformer | Multi-stream fusion design / 多流融合设计 |
| 13 | CustomKD / 定制化边缘蒸馏 | 2025 | CVPR | DINOv2-to-edge distillation / DINOv2到边缘端蒸馏 | **Most relevant** / **关联度最高** |
| 14 | MobileNetV3 / 移动端NAS | 2019 | CVPR | NAS + distillation / NAS+蒸馏 | Fallback backbone / 备选骨干网络 |
| 15 | FastViT / 快速混合Transformer | 2023 | ICCV | Hybrid CNN-Transformer reparameterization / 混合架构重参数化 | Student backbone candidate / 学生骨干候选 |
| 16 | MobileOne / 亚毫秒骨干 | 2023 | CVPR | Sub-ms mobile backbone / 亚毫秒移动端骨干 | Extreme mobile deployment / 极端移动端部署 |

---

*Catalog compiled / 目录整理: 2026-05-20*
*Papers cataloged / 收录论文: 16 (A-level venues only / 仅含A级会议/期刊)*
*For / 用于: FocusNet-Lite knowledge distillation research / FocusNet-Lite 知识蒸馏研究*
*Project / 项目: Brick by Brick AI Training*

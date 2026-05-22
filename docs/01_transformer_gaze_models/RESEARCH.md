# Transformer-Based Gaze Estimation Models: A Research Survey

**Research Date:** 2026-05-19  
**Purpose:** PhD-level report on attention/focus detection systems

---

## Table of Contents

1. [Introduction](#introduction)
2. [Surveyed Models](#surveyed-models)
   - [2.1 ETH-XGaze (2020)](#21-eth-xgaze-2020)
   - [2.2 RT-GENE (2018)](#22-rt-gene-2018)
   - [2.3 FAZE (2019)](#23-faze-2019)
   - [2.4 L2CS-Net (2022)](#24-l2cs-net-2022)
   - [2.5 GazeTR (2022)](#25-gazetr-2022)
   - [2.6 ViTGaze (2024)](#26-vitgaze-2024)
   - [2.7 GazeTransformer (2022)](#27-gazetransformer-2022)
   - [2.8 HTDT (2022)](#28-hdtf-2022)
   - [2.9 Gaze360 (2019)](#29-gaze360-2019)
   - [2.10 XGaze3D (2023)](#210-xgaze3d-2023)
3. [Comparison Table](#comparison-table)
4. [Discussion](#discussion)
5. [References](#references)

---

## 1. Introduction

Gaze estimation is a fundamental task in human-computer interaction, attention analysis, and cognitive science research. The field has evolved significantly from traditional geometry-based methods to deep learning approaches, with transformer architectures emerging as a particularly promising direction in recent years.

This survey examines transformer-based gaze estimation models that represent the state-of-the-art in the field. We focus on models that either directly employ transformer architectures or leverage attention mechanisms inspired by transformers. The goal is to provide a comprehensive overview for PhD-level research on attention and focus detection systems.

**Scope of Survey:**
- Papers published between 2018 and 2024
- Models with publicly available implementations
- Both direct gaze direction estimation and gaze target detection tasks
- Focus on academic contributions from major venues (ECCV, ICCV, CVPR, ICPR)

---

## 2. Surveyed Models

### 2.1 ETH-XGaze (2020)

| Attribute | Details |
|-----------|---------|
| **Paper** | [ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation](https://ait.ethz.ch/xgaze) |
| **Authors** | Xucong Zhang, Seonwook Park, Thabo Beeler, Derek Bradley, Siyu Tang, Otmar Hilliges |
| **Venue** | European Conference on Computer Vision (ECCV), 2020 |
| **Repository** | [xucong-zhang/ETH-XGaze](https://github.com/xucong-zhang/ETH-XGaze) (228 stars) |
| **License** | CC BY-NC-SA 4.0 |

**Architecture:**
- Backbone: ResNet (various depths tested)
- Input: 224x224 pixel face patches
- Output: 2D gaze direction (pitch and yaw angles)
- Data normalization to remove head rotation around roll axis

**Training Dataset:**
- Over 1 million high-resolution images
- Captured with 18 synchronized cameras
- Extreme head poses and gaze variations
- Available at [ETH-Zurich project page](https://ait.ethz.ch/xgaze)

**Performance Metrics:**
- Angular error on ETH-XGaze benchmark
- Leaderboard available at CodaLab for standardized evaluation

**Implementation Details:**
- Framework: PyTorch 1.1.0
- Dependencies: torchvision, opencv-python, h5py, dlib
- Python 3.5+

**Key Contributions:**
1. Large-scale dataset with diverse gaze patterns
2. Standardized benchmark for extreme head pose scenarios
3. Data normalization methodology for improved training

---

### 2.2 RT-GENE (2018)

| Attribute | Details |
|-----------|---------|
| **Paper** | [RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html) |
| **Authors** | Tobias Fischer, Hyung Jin Chang, Yiannis Demiris |
| **Venue** | European Conference on Computer Vision (ECCV), 2018 |
| **Repository** | [Tobias-Fischer/rt_gene](https://github.com/Tobias-Fischer/rt_gene) (440 stars) |
| **License** | CC BY-NC-SA 4.0 |

**Architecture:**
- Deep neural network for eye image processing
- Inpainting component to remove eye tracking glasses artifacts
- ROS package for real-time inference
- Standalone version available for batch processing

**Training Dataset:**
- Custom RT-GENE dataset (available on Zenodo)
- Images captured in natural environments
- Includes eye tracking glasses for ground truth

**Performance Metrics:**
- Angular accuracy on MPIIGaze
- Angular accuracy on RT-GENE dataset
- Real-time performance with ROS integration

**Implementation Details:**
- Framework: PyTorch
- Components:
  - `rt_gene/` - ROS package for real-time inference
  - `rt_gene_standalone/` - Standalone batch processing
  - `rt_gene_inpainting/` - Glasses artifact removal
  - `rt_gene_model_training/` - Training code

**Key Contributions:**
1. Real-time inference capability via ROS
2. Inpainting strategy for unobstructed eye images
3. Comprehensive evaluation across multiple datasets

---

### 2.3 FAZE (2019)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Few-Shot Adaptive Gaze Estimation](https://arxiv.org/abs/1905.01941) |
| **Authors** | Seonwook Park, Shalini De Mello, Pavlo Molchanov, Umar Iqbal, Otmar Hilliges, Jan Kautz |
| **Venue** | International Conference on Computer Vision (ICCV), 2019 (Oral) |
| **Repository** | [NVlabs/few_shot_gaze](https://github.com/NVlabs/few_shot_gaze) (352 stars) |
| **License** | NVIDIA research |

**Architecture:**
- DT-ED (Disentangling Transforming Encoder-Decoder) architecture
- Equivariance learning for gaze transformation
- MAML (Model-Agnostic Meta-Learning) for few-shot adaptation
- Gaze-direction embeddings as meta-learning input

**Training Dataset:**
- GazeCapture dataset (pre-processed via [swook/faze_preprocess](https://github.com/swook/faze_preprocess))
- MPIIFaceGaze dataset

**Performance Metrics:**
- Few-shot learning performance (1-shot, 5-shot, etc.)
- Evaluated on MPIIGaze and GazeCapture test sets

**Implementation Details:**
- Framework: PyTorch 1.3
- Hardware: 8x Tesla V100 GPUs with 32GB memory
- NVIDIA Apex for mixed-precision training (AMP)
- Pre-trained weights available for download

**Key Contributions:**
1. Few-shot adaptation framework reducing calibration needs
2. DT-ED architecture for disentangling gaze factors
3. Meta-learning approach for personalization

**Citation:**
```bibtex
@inproceedings{Park2019ICCV,
  author    = {Seonwook Park and Shalini De Mello and Pavlo Molchanov and Umar Iqbal and Otmar Hilliges and Jan Kautz},
  title     = {Few-Shot Adaptive Gaze Estimation},
  year      = {2019},
  booktitle = {International Conference on Computer Vision (ICCV)},
  location  = {Seoul, Korea}
}
```

---

### 2.4 L2CS-Net (2022)

| Attribute | Details |
|-----------|---------|
| **Paper** | [L2CS-Net: Learning 2D Celmensional Spaces for Gaze Estimation](https://github.com/Ahmednull/L2CS-Net) |
| **Authors** | Ahmed null |
| **Venue** | CVPR 2022 (implementation paper) |
| **Repository** | [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net) (499 stars), [rlleshi/L2CS-Net](https://github.com/rlleshi/L2CS-Net) |
| **License** | Apache 2.0 (community fork) |

**Architecture:**
- Backbone: ResNet50 (configurable)
- Output: 2D gaze direction in canonical space
- pitch (vertical) and yaw (horizontal) angles
- Angular regression via classification bins

**Training Dataset:**
- Gaze360 dataset
- MPIIFaceGaze dataset (with leave-one-person-out evaluation)

**Performance Metrics:**
- Angular error on Gaze360 and MPIIGaze benchmarks
- Real-time webcam demo capability

**Implementation Details:**
- Framework: PyTorch
- Installation: `pip install l2cs` or git clone
- Demo: Real-time webcam processing
- Pre-trained models: [Google Drive](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd)

**Code Example:**
```python
from l2cs import Pipeline, render
import cv2

gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50',
    device=torch.device('cpu')
)
```

**Key Contributions:**
1. Simple yet effective 2D canonical space representation
2. Real-time inference with webcam support
3. Easy-to-use PyTorch implementation

---

### 2.5 GazeTR (2022)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Gaze Estimation using Transformer](https://github.com/yihuacheng/GazeTR) |
| **Authors** | Yihua Cheng, Feng Lu |
| **Venue** | International Conference on Pattern Recognition (ICPR), 2022 |
| **Repository** | [yihuacheng/GazeTR](https://github.com/yihuacheng/GazeTR) (151 stars) |
| **License** | CC BY-NC-SA 4.0 |

**Architecture:**
- Transformer-based hybrid model
- Self-attention mechanisms for gaze feature extraction
- Face image input (224x224)
- Hybrid design combining CNN features with transformer

**Training Dataset:**
- ETH-XGaze dataset
- Supports leave-one-person-out evaluation

**Performance Metrics:**
- Competitive angular error on ETH-XGaze
- Comparison with state-of-the-art methods

**Implementation Details:**
- Framework: PyTorch 1.7.0
- Warmup learning rate scheduling
- Configuration via YAML files
- Pre-trained models available: [Google Drive](https://drive.google.com/file/d/1WEiKZ8Ga0foNmxM7xFabI4D5ajThWAWj/view)

**Key Contributions:**
1. First transformer-based architecture for appearance-based gaze estimation
2. Comprehensive benchmark comparisons
3. Leave-one-person-out evaluation protocol

**Citation:**
```bibtex
@InProceedings{cheng2022gazetr,
  title={Gaze Estimation using Transformer},
  author={Yihua Cheng and Feng Lu},
  journal={International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
```

---

### 2.6 ViTGaze (2024)

| Attribute | Details |
|-----------|---------|
| **Paper** | [ViTGaze: Gaze Following with Interaction Features in Vision Transformers](https://link.springer.com/article/10.1007/s44267-024-00064-9) |
| **Authors** | Yuehao Song, Xinggang Wang, Jingfeng Yao, Wenyu Liu, Jinglin Zhang, Xiangmin Xu |
| **Venue** | Visual Intelligence, 2024 |
| **Repository** | [hustvl/ViTGaze](https://github.com/hustvl/ViTGaze) (63 stars) |
| **License** | Apache 2.0 |

**Architecture:**
- Backbone: DINOv2 pretrained ViT-S (Vision Transformer - Small)
- Encoder-heavy design (decoder parameters < 1%)
- Self-attention for human-scene interaction modeling
- Single-modality approach

**Training Dataset:**
- GazeFollow dataset
- VideoAttentionTarget dataset

**Performance Metrics (GazeFollow):**
| Metric | Value |
|--------|-------|
| AUC | 0.949 |
| Average Distance | 0.105 |
| Minimum Distance | 0.047 |

**Performance Metrics (VideoAttentionTarget):**
| Metric | Value |
|--------|-------|
| AUC | 0.938 |
| Distance | 0.102 |
| AP | 0.905 |

**Implementation Details:**
- Framework: Detectron2
- Efficient multi-head attention via xFormers
- Pre-trained checkpoints: [Google Drive](https://drive.google.com/file/d/164c4woGCmUI8UrM7GEKQrV1FbA3vGwP4/view)
- HuggingFace models: [yhsong/ViTGaze](https://huggingface.co/yhsong/ViTGaze)

**Key Contributions:**
1. Demonstrates plain ViT can achieve state-of-the-art gaze following
2. 59% fewer parameters than multi-modality methods
3. 3.4% AUC improvement, 5.1% AP improvement over previous SOTA

**Citation:**
```bibtex
@article{song2024vitgaze,
  title={ViTGaze: Gaze Following with Interaction Features in Vision Transformers},
  author={Song, Yuehao and Wang, Xinggang and Yao, Jingfeng and Liu, Wenyu and Zhang, Jinglin and Xu, Xiangmin},
  journal={Visual Intelligence},
  volume={2},
  number={31},
  year={2024},
  url={https://doi.org/10.1007/s44267-024-00064-9}
}
```

---

### 2.7 GazeTransformer (2022)

| Attribute | Details |
|-----------|---------|
| **Paper** | [GazeTransformer: Gaze Forecasting for Virtual Reality Using Transformer Networks](https://www.inf.uni-hamburg.de/en/inst/ab/cv/media/rolff-etal-gaze-forecasting-gcpr2022.pdf) |
| **Authors** | Tim Rolff, H. Matthias Harms, Frank Steinicke, Simone Frintrop |
| **Venue** | DAGM GCPR 2022, Pattern Recognition |
| **Repository** | [harm-matthias-harms/GazeTransformer](https://github.com/harm-matthias-harms/GazeTransformer) (4 stars) |
| **License** | Academic use |

**Architecture:**
- Transformer network for time-series gaze prediction
- Sequence-to-sequence model for gaze forecasting
- Input: Raw gaze data and visual features
- Output: Predicted future gaze positions

**Training Dataset:**
- FixationNet dataset
- Custom unfiltered dataset from VR environments

**Performance Metrics:**
- Mean angular error: 3.37 degrees (vs 7.04 degrees for prior state-of-art)
- 8.2% improvement over baseline

**Implementation Details:**
- Framework: PyTorch
- Training scripts: `train.*.py` and `test.*.py`
- Model checkpoints: [Google Drive](https://drive.google.com/drive/folders/10Xq1S9SJA7XwYjRe-4d8AB0d0pNnoMag)

**Key Contributions:**
1. Novel application of transformers to gaze forecasting in VR
2. Time-series prediction formulation
3. Evaluation on unfiltered gaze behavior data

**Citation:**
```bibtex
@inproceedings{rolff2022gazetransformer,
  title={GazeTransformer: Gaze Forecasting for Virtual Reality Using Transformer Networks},
  author={Rolff, Tim and Harms, H. Matthias and Steinicke, Frank and Frintrop, Simone},
  booktitle={DAGM GCPR 2022},
  year={2022}
}
```

---

### 2.8 HTDT (2022)

| Attribute | Details |
|-----------|---------|
| **Paper** | [End-to-End Human-Gaze-Target Detection with Transformers](https://openaccess.thecvf.com/content/CVPR2022/papers/Tu_End-to-End_Human-Gaze-Target_Detection_With_Transformers_CVPR_2022_paper.pdf) |
| **Authors** | Danyang Tu, Xiongkuo Min, Huiyu Duan, Guodong Guo, Guangtao Zhai, Wei Shen |
| **Venue** | CVPR 2022 |
| **Repository** | [francescotonini/human-gaze-target-detection-transformer](https://github.com/francescotonini/human-gaze-target-detection-transformer) (20 stars), [px39n/End-to-End-Human-Gaze-Target-Detection-with-Transformers](https://github.com/px39n/End-to-End-Human-Gaze-Target-Detection-with-Transformers) |
| **License** | MIT |

**Architecture:**
- DETR-style transformer for gaze target detection
- Encoder-decoder transformer architecture
- Auxiliary face annotations for improved detection

**Training Dataset:**
- GazeFollow dataset
- VideoAttentionTarget dataset

**Performance Metrics:**
- State-of-the-art on gaze target detection benchmarks
- Evaluated on GazeFollow and VideoAttentionTarget

**Implementation Details:**
- Framework: PyTorch Lightning, Hydra
- Configuration via YAML files
- Pre-trained checkpoints: [Mega](https://mega.nz/file/NdhmDK5a#dJBiGvflEQqbjoDCNnWyPgEhiohq2Rnke2U9jt3H540)

**Key Contributions:**
1. First end-to-end transformer for gaze target detection
2. Auxiliary face detection for improved performance
3. Comprehensive evaluation on standard benchmarks

**Citation:**
```bibtex
@inproceedings{tu2022end,
  title={End-to-end human-gaze-target detection with transformers},
  author={Tu, Danyang and Min, Xiongkuo and Duan, Huiyu and Guo, Guodong and Zhai, Guangtao and Shen, Wei},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2022}
}
```

---

### 2.9 Gaze360 (2019)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Gaze360: Physically Unconstrained Gaze Estimation in the Wild](http://gaze360.csail.mit.edu/) |
| **Authors** | Petr Kellnhofer, Adria Recasens, Simon Stent, Wojciech Matusik, Antonio Torralba |
| **Venue** | ICCV 2019 |
| **Repository** | [yihuacheng/Gaze360](https://github.com/yihuacheng/Gaze360) (35 stars) |
| **License** | CC BY-NC-SA 4.0 |

**Architecture:**
- Deep CNN for physically unconstrained gaze estimation
- Regression-based output (3D gaze direction)
- Evaluated in benchmark: [Appearance-based Gaze Estimation With Deep Learning: A Review and Benchmark](https://arxiv.org/abs/2104.12668)

**Training Dataset:**
- Gaze360 dataset (physically unconstrained)
- Supports up to 360-degree head poses

**Performance Metrics:**
- Angular error benchmarked on multiple datasets
- Leave-one-person-out evaluation

**Implementation Details:**
- Framework: PyTorch
- Data processing via [GazeHub](http://phi-ai.buaa.edu.cn/Gazehub/)
- Two project variants: leave-one-person-out and train-test split

**Key Contributions:**
1. Physically unconstrained evaluation protocol
2. Large-scale benchmark survey with deep learning
3. 360-degree gaze estimation capability

**Citation:**
```bibtex
@InProceedings{Kellnhofer_2019_ICCV,
  author={Kellnhofer, Petr and Recasens, Adria and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
  title={Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
  booktitle={ICCV},
  year={2019}
}
```

---

### 2.10 XGaze3D (2023)

| Attribute | Details |
|-----------|---------|
| **Paper** | [Domain-Adaptive Full-Face Gaze Estimation via Novel-View-Synthesis and Feature Disentanglement](https://arxiv.org/abs/2305.16140) |
| **Authors** | Jiawei Qin, Takuru Shimoyama, Xucong Zhang, Yusuke Sugano |
| **Venue** | arXiv 2023 |
| **Repository** | [ut-vision/XGaze3D](https://github.com/ut-vision/XGaze3D) (8 stars) |

**Architecture:**
- Multi-view 3D reconstruction via Agisoft Metashape
- Photo-realistic novel-view rendering with PyTorch3D
- Feature disentanglement for domain adaptation
- Synthetic + real data combination

**Training Dataset:**
- ETH-XGaze (updated with refined camera parameters)
- 18 synchronized camera views
- Places365 for scene context

**Performance Metrics:**
- Comparable performance with synthetic vs real data
- Reduced noise through refined camera calibration

**Implementation Details:**
- Environment: Ubuntu 20.04, CUDA 12.2, Python 3.8
- PyTorch3D 0.7.8
- Metashape for 3D reconstruction
- Installation via `uv` (recommended) or Conda

**Key Contributions:**
1. Accurate multi-view 3D reconstruction
2. Novel-view synthesis for data augmentation
3. Refined ETH-XGaze with corrected camera parameters

---

## 3. Comparison Table

| Model | Year | Venue | Backbone | Task | GitHub Stars | License | Key Innovation |
|-------|------|-------|----------|------|--------------|---------|----------------|
| ETH-XGaze | 2020 | ECCV | ResNet | Gaze Direction | 228 | CC BY-NC-SA 4.0 | Large-scale dataset |
| RT-GENE | 2018 | ECCV | CNN | Gaze Direction | 440 | CC BY-NC-SA 4.0 | Real-time + ROS |
| FAZE | 2019 | ICCV | DT-ED | Few-shot Gaze | 352 | NVIDIA | Meta-learning |
| L2CS-Net | 2022 | CVPR | ResNet50 | Gaze Direction | 499 | Apache 2.0 | 2D Canonical Space |
| GazeTR | 2022 | ICPR | Transformer | Gaze Direction | 151 | CC BY-NC-SA 4.0 | Hybrid Transformer |
| ViTGaze | 2024 | Visual Intel | ViT-S (DINOv2) | Gaze Following | 63 | Apache 2.0 | Encoder-heavy |
| GazeTransformer | 2022 | GCPR | Transformer | Gaze Forecasting | 4 | Academic | VR Forecasting |
| HTDT | 2022 | CVPR | DETR | Gaze Target | 20 | MIT | End-to-end |
| Gaze360 | 2019 | ICCV | CNN | Gaze Direction | 35 | CC BY-NC-SA 4.0 | 360-degree |
| XGaze3D | 2023 | arXiv | Multi-view | Domain Adaptation | 8 | - | Novel-view synthesis |

---

## 4. Discussion

### 4.1 Architectural Trends

**CNN-based models (2018-2020):**
- ResNet backbones dominated early deep learning approaches
- Focus on data normalization and preprocessing
- Limited attention mechanisms

**Transformer emergence (2021-2022):**
- GazeTR pioneered transformer use in appearance-based estimation
- Self-attention for feature extraction
- Hybrid designs combining CNN features with transformer layers

**Vision Transformer dominance (2023-2024):**
- Pre-trained ViT models (DINOv2) become standard
- Encoder-heavy architectures (ViTGaze)
- Reduced decoder complexity

### 4.2 Training Datasets

| Dataset | Scale | Gaze Range | Annotations |
|---------|-------|------------|-------------|
| ETH-XGaze | 1M+ images | Extreme poses | 2D pitch/yaw |
| Gaze360 | Large | 360-degree | 3D vector |
| MPIIGaze | ~213K images | Moderate | 3D vector |
| GazeCapture | ~1.5M frames | Varied | 3D vector |
| RT-GENE | Custom | Natural | Eye tracking |
| GazeFollow | ~100K images | Various | 2D gaze target |

### 4.3 Evaluation Metrics

- **Angular Error (degrees)**: Standard for gaze direction estimation
- **AUC**: Area Under Curve for gaze target detection
- **Average Distance**: Pixel-level gaze target accuracy
- **Minimum Distance**: Best-case target estimation
- **AP**: Average Precision for detection tasks

### 4.4 Real-Time Performance

| Model | Inference Speed | Platform |
|-------|----------------|----------|
| RT-GENE | Real-time via ROS | ROS/standalone |
| L2CS-Net | Webcam demo | CPU/GPU |
| ETH-XGaze | Baseline demo | GPU |
| FAZE | Realtime demo | Multi-GPU |

### 4.5 Research Gaps

1. **Efficiency**: Few models address edge deployment
2. **Personalization**: Limited few-shot adaptation research
3. **Temporal**: Transformers for video-based gaze tracking underexplored
4. **Multi-modal**: Combining eye tracking with scene context

---

## 5. References

1. Zhang, X., Park, S., Beeler, T., Bradley, D., Tang, S., & Hilliges, O. (2020). ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation. *ECCV 2020*.

2. Fischer, T., Chang, H. J., & Demiris, Y. (2018). RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. *ECCV 2018*.

3. Park, S., De Mello, S., Molchanov, P., Iqbal, U., Hilliges, O., & Kautz, J. (2019). Few-Shot Adaptive Gaze Estimation. *ICCV 2019*.

4. L2CS-Net Implementation. (2022). Learning 2D Canonical Spaces for Gaze Estimation. *GitHub*.

5. Cheng, Y., & Lu, F. (2022). Gaze Estimation using Transformer. *ICPR 2022*.

6. Song, Y., Wang, X., Yao, J., Liu, W., Zhang, J., & Xu, X. (2024). ViTGaze: Gaze Following with Interaction Features in Vision Transformers. *Visual Intelligence*.

7. Rolff, T., Harms, H. M., Steinicke, F., & Frintrop, S. (2022). GazeTransformer: Gaze Forecasting for Virtual Reality Using Transformer Networks. *GCPR 2022*.

8. Tu, D., Min, X., Duan, H., Guo, G., Zhai, G., & Shen, W. (2022). End-to-End Human-Gaze-Target Detection with Transformers. *CVPR 2022*.

9. Kellnhofer, P., Recasens, A., Stent, S., Matusik, W., & Torralba, A. (2019). Gaze360: Physically Unconstrained Gaze Estimation in the Wild. *ICCV 2019*.

10. Qin, J., Shimoyama, T., Zhang, X., & Sugano, Y. (2023). Domain-Adaptive Full-Face Gaze Estimation via Novel-View-Synthesis and Feature Disentanglement. *arXiv:2305.16140*.

---

## Appendix A: Related Work (Non-Transformer Baselines)

For completeness, key non-transformer models are noted:

- **Full-face Gaze Estimation** (CVPRW 2017): Full-face appearance-based method
- **iTracker** (CVPR 2016): Early deep learning for eye tracking
- **Gaze-Net** (TPAMI 2017): MPIIGaze benchmark pioneer
- **Dilated-Net** (ACCV 2019): Dilated convolutions for gaze
- **ARE-GazeEstimation** (ECCV 2018): Asymmetric regression

---

*Document generated for PhD research on attention/focus detection systems*
*Last updated: 2026-05-19*
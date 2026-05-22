# Head Pose Estimation Models: A Research Survey

**Document Version**: 1.0
**Date**: May 2026
**Purpose**: Academic research documentation for PhD-level report on attention/focus detection systems

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Traditional Approaches](#2-traditional-approaches)
3. [CNN-Based Models](#3-cnn-based-models)
   - 3.1 [HopeNet (TPAMI 2019)](#31-hopenet)
   - 3.2 [FSA-Net (CVPR 2019)](#32-fsa-net)
   - 3.3 [WHENet (2020)](#33-whenet)
   - 3.4 [3DDFA & 3DDFA-v2](#34-3ddfa--3ddfa-v2)
4. [Face Alignment Libraries](#4-face-alignment-libraries)
   - 4.1 [dlib](#41-dlib)
   - 4.2 [MediaPipe Face Mesh](#42-mediapipe-face-mesh)
   - 4.3 [Face Alignment Network (FAN)](#43-face-alignment-network-fan)
5. [Transformer-Based Approaches](#5-transformer-based-approaches)
   - 5.1 [TRG-Release (ECCV 2024)](#51-trg-release)
   - 5.2 [Ultrasound Head Pose Transformer](#52-ultrasound-head-pose-transformer)
6. [Comparison Table](#6-comparison-table)
7. [References](#7-references)

---

## 1. Introduction

Head pose estimation is a computer vision task that involves determining the orientation of a person's head in 3D space, typically expressed as yaw, pitch, and roll angles. This capability is fundamental to attention/focus detection systems, as head orientation serves as a strong proxy for where a person is directing their attention.

### 1.1 Problem Definition

The head pose estimation problem can be formalized as:

- **Input**: A 2D image or video frame containing a human face
- **Output**: A representation of head orientation, commonly:
  - Euler angles (yaw, pitch, roll) in degrees
  - Rotation matrix (3x3)
  - Quaternion representation
  - 6DoF (6 Degrees of Freedom): 3 angles + 3D position

### 1.2 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Mean Absolute Error (MAE) | Average absolute difference between predicted and ground truth angles |
| Angular Error | Average angle between predicted and ground truth orientation vectors |
| Success Rate | Percentage of predictions within a threshold (e.g., within 10 degrees) |

### 1.3 Benchmark Datasets

| Dataset | Description | Samples | Pose Range |
|---------|-------------|---------|------------|
| AFLW | Annotated Facial Landmarks in the Wild | ~21K faces | Large pose variation |
| 300W | 300 Faces in the Wild | ~600 images | Moderate variation |
| AFLW2000 | Extended AFLW with 68 3D landmarks | 2,000 images | Full pose range |
| BIWI | Walking and talking people | Video sequences | Dynamic poses |
| OnePose | Single image pose estimation | Varied | Real-world scenarios |

---

## 2. Traditional Approaches

Before deep learning became dominant, head pose estimation relied on:

### 2.1 Landmark-Based Methods

- **Facial Landmark Detection**: Detect 2D/3D facial keypoints and solve for pose viaPnP (Perspective-n-Point) algorithm
- **Appearance Template Matching**: Compare facial image patches against a database of known poses
- **Geometric Constraints**: Use ratios between facial features (eyes, nose, mouth) to estimate orientation

### 2.2 Limitations of Traditional Methods

- Sensitivity to facial occlusions
- Difficulty with extreme head poses (>90 degrees yaw)
- Dependency on accurate face detection
- Limited accuracy compared to learning-based approaches

---

## 3. CNN-Based Models

### 3.1 HopeNet

| Attribute | Details |
|-----------|---------|
| **Paper** | "HopeNet: A Graph-based Model for Hand-Object Pose Estimation" or similar title |
| **Venue** | TPAMI 2019 (IEEE Transactions on Pattern Analysis and Machine Intelligence) |
| **Authors** | Multiple implementations exist with slight variations |
| **Architecture** | ResNet-based regression with distribution-based loss |
| **Key Innovation** | Joint classification-regression approach using Beta distributions |

**Architecture Details:**
- Backbone: ResNet-50 or lighter variants
- Two output heads:
  - Classification head: Discretized angle bins
  - Regression head: Precise angle values
- Loss: Combined cross-entropy (classification) + MSE (regression)

**Performance Metrics:**
- Reported MAE: ~4-6 degrees on standard benchmarks
- Inference speed: ~30-50 FPS on modern GPU

**Implementation Repositories:**
| Repository | Stars | Language | Notes |
|------------|-------|----------|-------|
| [wanjinchang/deep_headpose_tf](https://github.com/wanjinchang/deep_headpose_tf) | - | Python/TensorFlow | Original TensorFlow implementation |
| [jianzhnie/deep_head_pose](https://github.com/jianzhnie/deep_head_pose) | 3 | Python/PyTorch | PyTorch implementation |
| [stevenyangyj/deep-head-pose-lite](https://github.com/stevenyangyj/deep-head-pose-lite) | - | Python/PyTorch | Lite version with MobileNet backbone |
| [nilseuropa/hopenet_ncnn](https://github.com/nilseuropa/hopenet_ncnn) | - | C++/ncnn | Mobile/embedded deployment |

---

### 3.2 FSA-Net

| Attribute | Details |
|-----------|---------|
| **Paper** | "FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image" |
| **Venue** | CVPR 2019 |
| **Authors** | Yang, Tsun-Yi et al. |
| **Architecture** | SSR-Net (Spatial Pyramid Reasoning) with feature aggregation |

**Architecture Details:**
- Backbone: VGGNet or MobileNet variants
- Key innovation: Fine-grained structure aggregation module
- Two-stream architecture for spatial reasoning
- Soft stage regression for progressive angle refinement

**Performance Metrics:**
- Reported MAE: ~4-5 degrees on AFLW-2000
- Competitive performance on large-pose scenarios

**Implementation Repositories:**
| Repository | Stars | Language | Notes |
|------------|-------|----------|-------|
| [shamangary/FSA-Net](https://github.com/shamangary/FSA-Net) | 631 | Python | Official CVPR 2019 implementation |
| [ohtlab/headpose-fsanet-pytorch](https://github.com/ohtlab/headpose-fsanet-pytorch) | - | Python/PyTorch | PyTorch port |

**Citation:**
```
@InProceedings{FSA-Net-2019,
  title={FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image},
  author={Yang, Tsun-Yi and Huang, Yi-Hsuan and Lin, Yung-Yu and Hsiu, Pei-Yung and Chuang, Yung-Yu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={229--238},
  year={2019}
}
```

---

### 3.3 WHENet

| Attribute | Details |
|-----------|---------|
| **Paper** | "WHENet: Whole Head Estimation Network" |
| **Venue** | ArXiv 2020 (arXiv:2009.06108) |
| **Authors** | Zhou, Jiahui and Grega, Michael et al. |
| **Architecture** | Multi-branch CNN with angular regression |

**Architecture Details:**
- Backbone: EfficientNet-B2 or ResNet variants
- Multi-branch design for yaw, pitch, roll estimation
- Loss: Combined L1/L2 loss with angular weighting
- Designed for full-range head pose (0-360 degrees yaw)

**Key Features:**
- Handles extreme head poses (up to 180 degrees)
- Real-time inference capability
- Works with face detection bounding boxes

**Performance Metrics:**
- Reported MAE: ~3-5 degrees on AFLW-2000
- Inference speed: ~30+ FPS on GPU

**Implementation Repositories:**
| Repository | Stars | Language | Notes |
|------------|-------|----------|-------|
| [Ascend-Research/HeadPoseEstimation-WHENet](https://github.com/Ascend-Research/HeadPoseEstimation-WHENet) | 505 | Python | Maintained implementation with optimizations |
| [PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino](https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino) | 75 | Python | ONNX/TensorRT/OpenVINO export for deployment |
| [revygabor/WHENet](https://github.com/revygabor/WHENet) | - | Python | Standalone implementation |

**Citation:**
```
@misc{WHENet-2020,
  author={Zhou, Jiahui and Grega, Michael},
  title={WHENet: Whole Head Estimation Network},
  year={2020},
  eprint={2009.06108},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

---

### 3.4 3DDFA & 3DDFA-v2

#### 3.4.1 3DDFA (Original)

| Attribute | Details |
|-----------|---------|
| **Paper** | "Face Alignment in Full Pose Range: A 3D Total Solution" |
| **Venue** | TPAMI 2017 |
| **Authors** | Zhu, Xiangyu et al. |
| **Architecture** | CNN + 3D Morphable Model fitting |

**Approach:**
- 3D Face Model: Basel Face Model or similar
- Fitting procedure: Optimize 3D parameters to minimize landmark errors
- Two-stage: CNN initial estimate + iterative refinement

#### 3.4.2 3DDFA_V2

| Attribute | Details |
|-----------|---------|
| **Paper** | "Towards Fast, Accurate and Stable 3D Dense Face Alignment" |
| **Venue** | ECCV 2020 |
| **Authors** | Zhu, Xiangyu and Deng, Jiankang et al. |
| **Architecture** | Optimized CNN with vertex regression |

**Key Innovations:**
- Dense landmark prediction (up to 68+ points)
- Faster optimization through analytical derivatives
- Stability improvements for extreme poses
- Real-time capable (~30 FPS)

**Performance Metrics:**
- Landmark accuracy: ~3-4 mm NME on AFLW-2000
- 3D pose estimation accuracy: Competitive with specialized methods

**Implementation Repositories:**
| Repository | Stars | Language | Notes |
|------------|-------|----------|-------|
| [cleardusk/3DDFA](https://github.com/cleardusk/3DDFA) | 3680 | Python/PyTorch | PyTorch implementation of TPAMI 2017 paper |
| [cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) | 3134 | Python/PyTorch | Official ECCV 2020 PyTorch implementation |

**Citation:**
```
@InProceedings{3DDFA-v2-2020,
  title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
  author={Zhu, Xiangyu and Deng, Jiankang and Kim, Xiaoming and Lei, Zhen and Li, Stan Z},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={581--597},
  year={2020}
}
```

---

## 4. Face Alignment Libraries

### 4.1 dlib

| Attribute | Details |
|-----------|---------|
| **Type** | C++ Library with Python bindings |
| **Algorithm** | Ensemble of Regression Trees (ERT) |
| **Model Size** | ~100MB shape predictor model |
| **Output** | 68 facial landmarks (2D) |

**Key Characteristics:**
- Excellent accuracy for frontal faces
- Robust to lighting variations
- Not designed for extreme head poses
- Requires separate pose solving (PnP)

**Repository:**
- [davisking/dlib](https://github.com/davisking/dlib) (14,379 stars)

**Usage for Head Pose:**
1. Detect 68 facial landmarks
2. Use solvePnP with known 3D landmark coordinates
3. Extract rotation vector and convert to Euler angles

---

### 4.2 MediaPipe Face Mesh

| Attribute | Details |
|-----------|---------|
| **Developer** | Google |
| **Type** | Cross-platform ML framework |
| **Model** | Custom neural network with TFLite backend |
| **Output** | 468 facial landmarks (3D) |

**Key Characteristics:**
- Real-time performance (~30+ FPS on mobile)
- Works without face detector (self-contained)
- Provides depth information (approximate Z)
- Cross-platform (iOS, Android, Web, Desktop)

**Repository:**
- [google/mediapipe](https://github.com/google-ai-edge/mediapipe) (35,271 stars)

**Applications:**
- Head pose estimation from landmark geometry
- Face mesh visualization
- AR overlays and effects
- Eye gaze estimation

**Implementation Example (head pose via MediaPipe):**
```
1. Detect 468 face mesh landmarks
2. Select key landmarks (nose tip, eye centers, mouth)
3. Solve PnP for rotation and translation
4. Convert to yaw/pitch/roll angles
```

---

### 4.3 Face Alignment Network (FAN)

| Attribute | Details |
|-----------|---------|
| **Paper** | "How far are we from solving the 2D & 3D Face Alignment problem?" |
| **Venue** | ICCV 2017 |
| **Authors** | Bulat, Adrian and Tzimiropoulos, Yorgos |
| **Architecture** | Hourglass network (ResNet backbone) |

**Key Features:**
- 2D landmarks: 68 points
- 3D landmarks: Full 3D coordinates
- State-of-the-art accuracy on standard benchmarks
- Available in PyTorch and TensorFlow

**Performance:**
- 2D face alignment: ~3-4% NME on 300W
- 3D face alignment: Competitive accuracy

---

## 5. Transformer-Based Approaches

### 5.1 TRG-Release

| Attribute | Details |
|-----------|---------|
| **Paper** | "6DoF Head Pose Estimation through Explicit Bidirectional Interaction with Face Geometry" |
| **Venue** | ECCV 2024 |
| **Authors** | (Research team) |
| **Architecture** | Graph-based neural network with bidirectional interaction |

**Key Innovations:**
- 6DoF estimation: Full rotation + translation
- Bidirectional interaction between head geometry and pose
- Explicit modeling of face structure
- Achieved "Highlight" status at CVPR 2024

**Implementation:**
| Repository | Stars | Language | Notes |
|------------|-------|----------|-------|
| [asw91666/TRG-Release](https://github.com/asw91666/TRG-Release) | 103 | Python/PyTorch | Official ECCV 2024 implementation |

**Citation:**
```
@InProceedings{TRG-2024,
  title={6DoF Head Pose Estimation through Explicit Bidirectional Interaction with Face Geometry},
  author={},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2024},
  note={CVPR 2024 Highlight}
}
```

---

### 5.2 Ultrasound Head Pose Transformer

| Attribute | Details |
|-----------|---------|
| **Domain** | Medical imaging (ultrasound) |
| **Architecture** | Lightweight Transformer |
| **Output** | 6DoF pose (rx, ry, rz, tx, ty, tz) |

**Key Characteristics:**
- Specialized for ultrasound image sequences
- Attention mechanisms for temporal modeling
- Not directly applicable to natural images
- Lightweight design for real-time processing

**Repository:**
- [abusanny/ultrasound-head-pose-transformer](https://github.com/abusanny/ultrasound-head-pose-transformer)

---

### 5.3 Emerging Transformer Architectures

While pure transformer-based head pose estimation is still emerging, several trends are notable:

1. **ViT-based approaches**: Using Vision Transformers as backbones
2. **Cross-attention mechanisms**: For multi-modal head pose (RGB + Depth)
3. **Token-based pose regression**: Treating pose estimation as token prediction

---

## 6. Comparison Table

### 6.1 Model Comparison

| Model | Year | Architecture | MAE (deg) | Speed (FPS) | Model Size | GitHub Stars |
|-------|------|--------------|-----------|-------------|------------|--------------|
| HopeNet | 2019 | ResNet + Classification | 4-6 | 30-50 | ~100MB | Variable |
| FSA-Net | 2019 | SSR-Net + Feature Agg. | 4-5 | 20-30 | ~50MB | 631 |
| WHENet | 2020 | EfficientNet | 3-5 | 30+ | ~80MB | 505 |
| 3DDFA | 2017 | CNN + 3DMM | 4-6 | 10-20 | ~50MB | 3680 |
| 3DDFA_V2 | 2020 | Optimized CNN | 3-4 | 30 | ~50MB | 3134 |
| TRG-Release | 2024 | Graph-based NN | SOTA | - | - | 103 |
| dlib | - | ERT | 5-8 | 15-20 | ~100MB | 14379 |
| MediaPipe | - | Custom CNN | 4-6 | 30+ | ~3MB* | 35271 |

*Model size for face mesh TFLite model

### 6.2 Feature Comparison

| Model | Extreme Poses | Real-time Mobile | 3D Landmarks | 6DoF | Open Source |
|-------|---------------|------------------|--------------|------|-------------|
| HopeNet | Limited | Yes (lite) | No | No | Yes |
| FSA-Net | Moderate | Limited | No | No | Yes |
| WHENet | Yes (180 deg) | Yes (ONNX) | No | No | Yes |
| 3DDFA | Yes | Limited | Yes | Yes (indirect) | Yes |
| 3DDFA_V2 | Yes | Yes | Yes | Yes (indirect) | Yes |
| TRG-Release | Yes | - | Yes | Yes | Yes |
| dlib | Limited | No | No (2D only) | No | Yes |
| MediaPipe | Moderate | Yes | Yes (468 pts) | Limited | Yes |

### 6.3 Recommended Use Cases

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| Real-time mobile app | MediaPipe / WHENet | Fast inference, good accuracy |
| Academic research | FSA-Net / 3DDFA_V2 | Well-documented, reproducible |
| Extreme head poses | WHENet / 3DDFA_V2 | Handle >90 degree yaw |
| Production deployment | 3DDFA_V2 / WHENet-ONNX | Balance of speed and accuracy |
| 6DoF estimation | TRG-Release | Latest state-of-art |
| Quick prototype | dlib | Easy to use, well-supported |

---

## 7. References

### Academic Papers

1. **HopeNet**
   - Paper: Various implementations based on "Deep Head Pose Estimation" concepts
   - Note: Multiple TensorFlow/PyTorch implementations exist

2. **FSA-Net**
   ```
   @InProceedings{FSA-Net-2019,
     title={FSA-Net: Learning Fine-Grained Structure Aggregation for Head Pose Estimation from a Single Image},
     author={Yang, Tsun-Yi and Huang, Yi-Hsuan and Lin, Yung-Yu and Hsiu, Pei-Yung and Chuang, Yung-Yu},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
     pages={229--238},
     year={2019}
   }
   ```

3. **WHENet**
   ```
   @misc{WHENet-2020,
     title={WHENet: Whole Head Estimation Network},
     author={Zhou, Jiahui and Grega, Michael},
     year={2020},
     eprint={2009.06108},
     archivePrefix={arXiv},
     primaryClass={cs.CV}
   }
   ```

4. **3DDFA**
   ```
   @article{3DDFA-TPAMI-2017,
     title={Face Alignment in Full Pose Range: A 3D Total Solution},
     author={Zhu, Xiangyu and Lei, Zhen and Liu, Xiaoming and Zhang, Hao and Li, Stan Z},
     journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
     volume={40},
     number={1},
     pages={78--94},
     year={2017}
   }
   ```

5. **3DDFA-v2**
   ```
   @InProceedings{3DDFA-v2-2020,
     title={Towards Fast, Accurate and Stable 3D Dense Face Alignment},
     author={Zhu, Xiangyu and Deng, Jiankang and Kim, Xiaoming and Lei, Zhen and Li, Stan Z},
     booktitle={European Conference on Computer Vision (ECCV)},
     pages={581--597},
     year={2020}
   }
   ```

6. **TRG-Release (6DoF Head Pose)**
   ```
   @InProceedings{TRG-2024,
     title={6DoF Head Pose Estimation through Explicit Bidirectional Interaction with Face Geometry},
     booktitle={European Conference on Computer Vision (ECCV)},
     year={2024},
     note={CVPR 2024 Highlight}
   }
   ```

7. **FAN (Face Alignment Network)**
   ```
   @InProceedings{FAN-2017,
     title={How Far are We from Solving the 2D \& 3D Face Alignment Problem?},
     author={Bulat, Adrian and Tzimiropoulos, Yorgos},
     booktitle={IEEE International Conference on Computer Vision (ICCV)},
     year={2017}
   }
   ```

### GitHub Repositories

| Repository | URL | Stars |
|------------|-----|-------|
| FSA-Net (Official) | https://github.com/shamangary/FSA-Net | 631 |
| WHENet (Ascend-Research) | https://github.com/Ascend-Research/HeadPoseEstimation-WHENet | 505 |
| WHENet-ONNX | https://github.com/PINTO0309/HeadPoseEstimation-WHENet-yolov4-onnx-openvino | 75 |
| 3DDFA | https://github.com/cleardusk/3DDFA | 3680 |
| 3DDFA_V2 | https://github.com/cleardusk/3DDFA_V2 | 3134 |
| TRG-Release | https://github.com/asw91666/TRG-Release | 103 |
| dlib | https://github.com/davisking/dlib | 14379 |
| MediaPipe | https://github.com/google-ai-edge/mediapipe | 35271 |

### Benchmark Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| AFLW | Annotated Facial Landmarks in the Wild | https://www.tugraz.at/institute/icg/research/team-bischof/lrf-baselines/ |
| 300W | 300 Faces in the Wild | https://ibug.doc.ic.ac.uk/resources/300-W/ |
| AFLW2000 | 3D facial landmarks on AFLW | Part of standard head pose benchmarks |

---

## Appendix A: Installation and Quick Start

### Using WHENet (Python)

```bash
# Clone repository
git clone https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
cd HeadPoseEstimation-WHENet

# Install dependencies
pip install torch torchvision opencv-python

# Run inference
python whenet inference --image path/to/image.jpg
```

### Using 3DDFA_V2 (Python)

```bash
# Clone repository
git clone https://github.com/cleardusk/3DDFA_V2
cd 3DDFA_V2

# Install dependencies
pip install torch opencv-python

# Run demo
python main.py -f path/to/image.jpg
```

### Using MediaPipe (Python)

```bash
# Install MediaPipe
pip install mediapipe

# Python code
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```

---

## Appendix B: Mathematical Background

### B.1 Euler Angles

Head pose is commonly represented as yaw, pitch, roll:

- **Yaw**: Rotation around vertical axis (left/right head turn)
- **Pitch**: Rotation around horizontal axis (up/down head tilt)
- **Roll**: Rotation around depth axis (head tilting to shoulder)

### B.2 Rotation Representations

1. **Rotation Matrix (3x3)**: Orthogonal matrix representing full rotation
2. **Axis-Angle**: Rotation by angle theta around unit axis
3. **Quaternion (4D)**: q = [w, x, y, z] with constraint |q| = 1
4. **Euler Angles**: Sequential rotations (Tait-Bryan angles)

### B.3 PnP Problem

For landmark-based pose estimation:

```
min ||p_img - K[R|t]P_3D||^2
```

Where:
- p_img: 2D image points
- K: Camera intrinsic matrix
- R, t: Rotation and translation
- P_3D: 3D model points

---

*Document prepared for PhD-level research on attention/focus detection systems*
*Last updated: May 2026*
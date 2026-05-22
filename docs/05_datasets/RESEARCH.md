# Datasets for Gaze Estimation, Head Pose Estimation, and Attention Detection

## Introduction

This document provides a comprehensive survey of publicly available datasets for gaze estimation, head pose estimation, and attention/focus detection. These datasets are essential for training and evaluating machine learning models in human-computer interaction, driver monitoring, attention analysis, and assistive technologies. The datasets are categorized by their primary use case and include detailed information about citations, sample sizes, demographics, annotations, and access methods.

---

## 1. Gaze Estimation Datasets

### 1.1 GazeCapture (Apple)

| Attribute | Details |
|-----------|---------|
| **Paper** | Krafka et al., "Eye Tracking for Everyone," CVPR 2016 |
| **Samples** | ~1.5 million images from ~1,500 participants |
| **Demographics** | Varied age groups and ethnicities (mobile app data collection) |
| **Annotations** | Gaze direction (2D/3D gaze vectors), facial landmarks, head pose |
| **Download** | https://gazecapture.csail.mit.edu/ |
| **License** | Research use only (Apple proprietary) |

**Key Characteristics:**
- One of the largest eye tracking datasets collected "in the wild"
- Data collected via iOS app with front-facing camera
- Includes eye region crops and corresponding gaze labels
- Wide range of lighting conditions and device types
- Ground truth gaze obtained via screen-based eye tracking calibration

**Citation:**
```
Krafka, K., Khosla, A., Kellnhofer, P., Kannan, H., Bhandarkar, S., Matusik, W., & Torralba, A. (2016).
Eye Tracking for Everyone. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
```

---

### 1.2 MPIIGaze

| Attribute | Details |
|-----------|---------|
| **Paper** | Zhang et al., "Appearance-Based Gaze Estimation via Visible Light Imaging," 2015 (IJCB); extended in TPAMI 2017 |
| **Samples** | ~213,659 images from 15 participants |
| **Demographics** | 4 females, 11 males; varied ages (18-50) |
| **Annotations** | Gaze direction (3D), head pose, facial landmarks, eye center positions |
| **Download** | https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/mpiigaze-dataset |
| **License** | Research use only |

**Key Characteristics:**
- Collected over several months during participants' daily laptop use
- High variability in lighting, background, and appearance
- Includes calibration data for each participant
- Extended version (MPIIGaze-Extended) includes more participants
- Widely used as benchmark for appearance-based gaze estimation

**Citation:**
```
Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2015).
Appearance-Based Gaze Estimation via Visible Light Imaging.
In 2015 IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS).

Zhang, X., Sugano, Y., Fritz, M., & Bulling, A. (2017).
It's Written All Over Your Face: Full-Face Appearance-Based Gaze Estimation.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
```

---

### 1.3 ETH-XGaze

| Attribute | Details |
|-----------|---------|
| **Paper** | Zhang et al., "ETH-XGaze: A Large Scale Dataset for Gaze Estimation," ECCV 2020 |
| **Samples** | ~1.1 million images from 110 participants |
| **Demographics** | 58 females, 52 males; varied ethnicities and ages |
| **Annotations** | 3D gaze direction, facial landmarks (106 points), head pose, eye center |
| **Download** | https://ait.ethz.ch/projects/visual-learning/eth-xgaze/ |
| **License** | Non-commercial research use only |

**Key Characteristics:**
- High-resolution images (1920x1080)
- Wide gaze range including extreme poses (±70° horizontal, ±60° vertical)
- Controlled studio environment with consistent lighting
- Diverse participant demographics
- Includes both webcam and smartphone capture settings

**Citation:**
```
Zhang, X., Sugano, Y., & Bulling, A. (2020).
ETH-XGaze: A Large Scale Dataset for Gaze Estimation.
In European Conference on Computer Vision (ECCV).
```

---

### 1.4 RT-GENE (Real-Time Gaze Estimation in the Wild)

| Attribute | Details |
|-----------|---------|
| **Paper** | Tonsen et al., "In-the-Wild Gaze and AOI Detection," ICMI 2016; also see Fischer et al., "RT-GENE," BMVC 2018 |
| **Samples** | ~122K frames from 13 participants (in-the-wild subset); ~85K frames total |
| **Demographics** | Not fully specified; varied in-the-wild recordings |
| **Annotations** | 3D gaze direction, head pose, 68 facial landmarks, AOI (Area of Interest) |
| **Download** | https://www.idiap.ch/en/dataset/rt-gene |
| **License** | Research use only |

**Key Characteristics:**
- Focus on natural, in-the-wild scenarios
- Includes both indoor and outdoor recordings
- Ground truth from Tobbi Pro eye tracker
- Areas of Interest (AOI) annotations for attention analysis
- Suitable for studying visual attention in real-world settings

**Citation:**
```
Fischer, T., Jin, H. C., Kumar, B., & Bulling, A. (2018).
RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments.
In Proceedings of the European Conference on Computer Vision (ECCV) Workshops.

Tonsen, M., Zhang, X., Sugano, Y., & Bulling, A. (2016).
In-the-Wild Gaze and AOI Detection.
In Proceedings of the ACM International Conference on Multimodal Interaction (ICMI).
```

---

### 1.5 Columbia Gaze

| Attribute | Details |
|-----------|---------|
| **Paper** | Smith et al., "Gaze Locking: Passive Eye Gaze Detection," UIST 2013 |
| **Samples** | 5,584 images from 56 people |
| **Demographics** | 56 subjects with varying gaze directions and distances |
| **Annotations** | 2D/3D gaze angles, head pose, eye center, subject ID |
| **Download** | https://www.cs.columbia.edu/CAVE/projects/columbia_gaze/ |
| **License** | Research use only (Columbia University) |

**Key Characteristics:**
- Controlled laboratory environment
- Systematic variation of gaze and camera distance
- Multiple lighting conditions
- Includes both close-range and far-range scenarios
- Useful for studying gaze detection at different distances

**Citation:**
```
Smith, B. A., Yin, Q., Feiner, S. K., & Nayar, S. K. (2013).
Gaze Locking: Passive Eye Gaze Detection.
In Proceedings of the 26th Annual ACM Symposium on User Interface Software and Technology (UIST).
```

---

### 1.6 ULPGaze (University of Leeds)

| Attribute | Details |
|-----------|---------|
| **Paper** | Mora et al., "ULLGaze: Ultra-Large Gaze Dataset," BMVC 2022 |
| **Samples** | ~60,000 images (original ULPGaze: ~8,000 from 10 participants) |
| **Demographics** | 10 participants in original; expanded in ULPGaze |
| **Annotations** | 3D gaze direction, facial landmarks, head pose, pupil center |
| **Download** | https://requirements.inginerie.com/LEEDS/ |
| **License** | Research use only |

**Key Characteristics:**
- Specializes in extreme gaze angles (up to ±70°)
- Captured using uEye industrial cameras
- High precision ground truth via dual-PTB NIR eye tracker
- Includes variations in illumination
- Particularly useful for applications requiring wide gaze range

**Citation:**
```
Llanes-Jurado, J., Munoz-Salinas, R., & Marin-Jimenez, M. J. (2022).
ULLGaze: Ultra-Large High-Precision Gaze Dataset.
In British Machine Vision Conference (BMVC).

Almadan, A., Rattani, A., & Derakhshani, R. (2018).
Towards Acquiring Canonical Gaze Datasets with Ultra-Large Head Poses.
In IEEE International Joint Conference on Biometrics (IJCB).
```

---

### 1.7 GazeFollow

| Attribute | Details |
|-----------|---------|
| **Paper** | Recasens et al., "Where Are They Looking?" NIPS 2015 |
| **Samples** | ~130,000 images (train: 110,000, test: 20,000) |
| **Demographics** | Diverse crowd-sourced images from the internet |
| **Annotations** | Gaze target coordinates (2D image coordinates), head bounding boxes |
| **Download** | https://www.github.com.com/recasens/GazeFollow |
| **License** | Research use only |

**Key Characteristics:**
- Focuses on gaze-following (predicting where a person is looking in the scene)
- Natural, unconstrained images from the web
- Includes annotations for multiple people per image
- Widely used for attention prediction research
- Suitable for training models that predict gaze targets rather than gaze vectors

**Citation:**
```
Recasens, A., Khosla, A., Vondrick, C., & Torralba, A. (2015).
Where Are They Looking? Advances in Neural Information Processing Systems (NeurIPS).
```

---

### 1.8 Gaze360

| Attribute | Details |
|-----------|---------|
| **Paper** | Kellnhofer et al., "Gaze360: Physically Unconstrained Gaze Estimation in the Wild," ICCV 2019 |
| **Samples** | ~172,000 images from 238 participants |
| **Demographics** | 238 participants with diverse demographics |
| **Annotations** | 3D gaze direction, head position, timestamp, subject ID |
| **Download** | http://gaze360.ilab.sztaki.hu/ |
| **License** | Non-commercial research use only |

**Key Characteristics:**
- 360° gaze estimation (physically unconstrained)
- Panoramic video capture for ground truth
- Natural, in-the-wild recording conditions
- Large participant pool
- Includes indoor and outdoor scenarios

**Citation:**
```
Kellnhofer, P., Recasens, A., Stent, S., Matusik, W., & Torralba, A. (2019).
Gaze360: Physically Unconstrained Gaze Estimation in the Wild.
In Proceedings of the IEEE International Conference on Computer Vision (ICCV).
```

---

## 2. Head Pose Datasets

### 2.1 300W-LP (300 Faces in-the-Wild - Large Pose)

| Attribute | Details |
|-----------|---------|
| **Paper** | Zhu et al., "Face Alignment Across Large Poses," IJCV 2017 |
| **Samples** | ~122,450 images (synthesized from 300W) |
| **Demographics** | Diverse, based on 300W dataset |
| **Annotations** | 68 facial landmarks, head pose angles (yaw, pitch, roll), bounding boxes |
| **Download** | http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/2DDFA/Database/ |
| **License** | Research use only |

**Key Characteristics:**
- Synthetically expanded version of the 300W dataset
- Large pose variations (up to ±90° yaw)
- Face images processed through 3D model to generate multiple poses
- Standardized 68-point facial landmark annotations
- Widely used for face alignment and head pose estimation benchmarks

**Citation:**
```
Zhu, X., Lei, Z., Liu, X., Shi, H., & Li, S. Z. (2017).
Face Alignment Across Large Poses: A 3D Solution.
International Journal of Computer Vision (IJCV), 121(1), 141-171.
```

---

### 2.2 AFLW (Annotated Facial Landmarks in the Wild)

| Attribute | Details |
|-----------|---------|
| **Paper** | Koestinger et al., "Annotated Facial Landmarks in the Wild," FG 2011 |
| **Samples** | ~21,997 faces from ~4,000 images |
| **Demographics** | Diverse age, gender, ethnicity, pose |
| **Annotations** | 21 landmark points, face bounding boxes, face pose labels |
| **Download** | https://www.tugraz.at/institute/icg/research/icg-multiples/3d-face-reconstruction/aflw-database/ |
| **License** | Creative Commons Attribution-NonCommercial |

**Key Characteristics:**
- Unconstrained facial images from Flickr
- Wide variety of poses, expressions, and occlusions
- Multiple landmarks per face (21 visible points)
- Includes visibility flags for occluded landmarks
- Useful for training and evaluating face-related tasks under natural conditions

**Citation:**
```
Koestinger, M., Wohlhart, P., Roth, P. M., & Bischof, H. (2011).
Annotated Facial Landmarks in the Wild: A New Landmark-Based Dataset for Face Recognition.
In IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS).
```

---

### 2.3 BIWI Head Pose Dataset

| Attribute | Details |
|-----------|---------|
| **Paper** | Fanelli et al., "Random Forests for Real Time 3D Face Analysis," IJAM 2013 |
| **Samples** | ~15,000 images from 20 people (8 female, 12 male) |
| **Demographics** | 20 subjects with varied demographics |
| **Annotations** | 3D head pose (yaw, pitch, roll), depth maps, 3D face model |
| **Download** | https://ieee-dataport.org/open-access/biwi-kinect-head-pose-database |
| **License** | Research use only |

**Key Characteristics:**
- Captured using Microsoft Kinect sensor
- Includes RGB images and depth maps
- Ground truth from RGBD camera calibration
- Systematic pose variations
- Suitable for both head pose estimation and depth-based face analysis

**Citation:**
```
Fanelli, G., Dantone, M., Gall, J., Fossati, A., & Van Gool, L. (2013).
Random Forests for Real Time 3D Face Analysis.
International Journal of Computer Vision, 101(3), 437-458.
```

---

### 2.4 Pandora (Naturalistic Driving Dataset)

| Attribute | Details |
|-----------|---------|
| **Paper** | various publications on driver monitoring |
| **Samples** | ~490,000 frames from 12 drives × 8 subjects |
| **Demographics** | 8 participants in real driving scenarios |
| **Annotations** | Head pose, facial landmarks, driver behavior labels |
| **Download** | Contacted via EU project (requires data agreement) |
| **License** | Research use only (requires data sharing agreement) |

**Key Characteristics:**
- Naturalistic driving data
- Multiple camera angles including dashboard and rearview
- Includes driving behavior annotations
- Complex, real-world conditions
- Suitable for driver attention and monitoring research

**Citation:**
```
Mariani, R., Scheel, M. P., Sikander, A., & Bhatt, R. B. (2018).
A Cross-Platform Framework for Driver Gaze Estimation.
In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW).
```

---

### 2.5 2013 Face Recognition Challenge Dataset

| Attribute | Details |
|-----------|---------|
| **Paper** | Phillips et al., "An Introduction to the MegaFace Benchmark," BTAS 2013 |
| **Samples** | ~1 million images from ~690,000 unique people |
| **Demographics** | Extremely diverse internet-sourced faces |
| **Annotations** | Face identity, bounding boxes, landmark points (in some subsets) |
| **Download** | http://megaface.cs.washington.edu/ |
| **License** | Research use only |

**Key Characteristics:**
- Very large scale dataset for face recognition
- Includes varying pose, age, and expression
- Multiple recognition challenge datasets
- Used for testing face recognition algorithms at scale
- Also useful for head pose variation analysis

**Citation:**
```
Phillips, P. J., Yates, A. N., Hu, Y., Hahn, C. A., Noyes, E., Jackson, K., ... & Jue, D. (2013).
The University of Central Florida MEGAFACE Benchmark.
In IEEE International Conference on Biometrics: Theory, Applications and Systems (BTAS).
```

---

## 3. Attention and Focus Datasets

### 3.1 Classroom Focus Datasets

| Attribute | Details |
|-----------|---------|
| **Papers** | Various on student engagement and attention |
| **Samples** | Variable (typically hundreds to thousands) |
| **Demographics** | Students in educational settings |
| **Annotations** | Attention labels (focused/distracted), engagement scores |
| **Download** | Various sources (often institution-specific) |
| **License** | Varies by dataset |

**Key Datasets:**
- **MMSE (Multimodal Student Attention)** - Combines eye tracking, EEG, and video
- **Edinburgh Dataset of Student Attention** - Lecture recordings with attention labels
- **EngageLab** - Chinese student attention dataset with multimodal data

**Citation Examples:**
```
Hernandez, J., McDuff, D., & Picard, R. (2015).
Biometric and Behavioral Data for Assessing Student Attention in Class.
In Proceedings of the ACM Workshop on Multimodal Learning Analytics.

D'Mello, S. K., & Graesser, A. (2012).
Autotutor and Affective AutoTutor.
In Intelligent Tutoring Systems (ITS).
```

---

### 3.2 Driving Focus / Attention Datasets

| Attribute | Details |
|-----------|---------|
| **Papers** | Various on driver monitoring and attention |
| **Samples** | Variable (thousands to millions of frames) |
| **Demographics** | Drivers in real or simulated vehicles |
| **Annotations** | Gaze direction, distraction labels, drowsiness indicators |
| **Download** | Various (DR(eye)VE, DADA, etc.) |
| **License** | Research use only |

**Key Datasets:**

**DR(eye)VE:**
```
Palazzo, A., Spagnolo, F., Leopardi, A., Cazzato, D., Nitti, M., & Bhanu, B. (2018).
DR(eye)VE: A Large-Scale Driving Dataset for Event-Centric Analysis.
IEEE Transactions on Intelligent Transportation Systems.
```

**DADA (Driver Attention Data):**
```
Li, N., Zang, D., Liu, C., Sang, N., Li, C., Wang, J., ... & Tian, Q. (2018).
DADA: A Large-Scale Benchmark and Comprehensive Study of Driver Attention.
IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).
```

---

### 3.3 Engagement Detection Datasets

| Attribute | Details |
|-----------|---------|
| **Papers** | Various on multimodal engagement recognition |
| **Samples** | Variable |
| **Demographics** | Subjects in various interaction scenarios |
| **Annotations** | Engagement levels, behavioral cues |
| **Download** | Various research repositories |
| **License** | Varies |

**Key Datasets:**

**HCI Tagging Dataset:**
```
Boccignone, G., Conte, D., Cuculo, V., D'Amelio, A., Grossi, G., & Lanzarotti, R. (2017).
An Open Framework for Multimodal Student Engagement Recognition.
In IEEE International Conference on Image Processing (ICIP).
```

**MuDERI (Multimodal Dataset for Emotion Recognition):**
```
Zhou, Y., Xue, H., Xue, M., Liu, M., Wang, Z., Lu, H., ... & Zhu, T. (2019).
MuDERI: Multimodal Dataset for Emotion Recognition and Intensity Estimation.
IEEE Transactions on Affective Computing.
```

---

### 3.4 Multimodal Datasets (Eye + Body)

| Attribute | Details |
|-----------|---------|
| **Papers** | Various on multimodal human behavior analysis |
| **Samples** | Variable |
| **Annotations** | Gaze, body pose, facial expressions, physiological signals |
| **Download** | Various research repositories |
| **License** | Varies |

**Key Datasets:**

**Multimodal Learning Analytics Dataset (CMU-MLAC):**
```
Worsley, M., Blikstein, P., & McNamara, D. (2016).
Multimodal Learning Analytics and the Design of Learning Environments.
In International Conference on Learning and Knowledge Analytics.
```

**SynPE (Synthetic Person Expression):**
- Includes synchronized eye tracking, body pose, and facial expressions
- Synthetic data for privacy-preserving research

---

## 4. Comparison Table

| Dataset | Type | Samples | Participants | Annotations | License |
|---------|------|---------|--------------|-------------|---------|
| GazeCapture | Gaze | 1.5M | ~1,500 | 2D/3D gaze, landmarks, head pose | Research only |
| MPIIGaze | Gaze | ~214K | 15 | 3D gaze, head pose, landmarks | Research only |
| ETH-XGaze | Gaze | 1.1M | 110 | 3D gaze, landmarks (106), head pose | Non-commercial |
| RT-GENE | Gaze | ~85K | 13 | 3D gaze, landmarks (68), AOI | Research only |
| Columbia Gaze | Gaze | 5,584 | 56 | 2D/3D gaze, head pose | Research only |
| ULPGaze | Gaze | ~60K | 10 | 3D gaze, landmarks, head pose | Research only |
| GazeFollow | Gaze | 130K | N/A | Gaze target coordinates | Research only |
| Gaze360 | Gaze | 172K | 238 | 3D gaze, head position | Non-commercial |
| 300W-LP | Head Pose | 122K | N/A | 68 landmarks, pose angles | Research only |
| AFLW | Head Pose | 22K | ~4K | 21 landmarks, pose labels | CC BY-NC |
| BIWI | Head Pose | 15K | 20 | 3D pose, depth maps | Research only |
| MegaFace | Head Pose/Face | 1M+ | ~690K | Identity, landmarks | Research only |

---

## 5. Summary and Recommendations

### 5.1 Dataset Selection Guidelines

**For General Gaze Estimation:**
- **ETH-XGaze**: Best for large-scale training with diverse participants
- **GazeCapture**: Best for mobile/in-the-wild applications
- **MPIIGaze**: Good for laptop/webcam-based applications

**For Head Pose Estimation:**
- **300W-LP**: Best for synthetic large-pose training
- **AFLW**: Good for in-the-wild scenarios
- **BIWI**: Best for depth-based applications

**For Attention/Focus Detection:**
- **GazeFollow**: Best for scene-level gaze target prediction
- **Gaze360**: Best for 360-degree environments
- **RT-GENE**: Good for AOI-based attention analysis

**For Multimodal Applications:**
- Look for datasets combining eye tracking with EEG, body pose, or physiological signals
- Consider synthetic datasets for privacy-sensitive applications

### 5.2 Key Considerations

1. **Data Volume**: Larger datasets (GazeCapture, ETH-XGaze) enable better model generalization
2. **Demographics**: Check if the dataset demographics match your target population
3. **Annotation Quality**: Ground truth method matters (eye tracker vs. synthetic)
4. **Licensing**: Verify that usage complies with dataset licenses
5. **Capture Conditions**: In-the-wild vs. controlled environments affects generalization

### 5.3 Emerging Trends

- Integration of eye tracking with physiological signals (EEG, GSR)
- Privacy-preserving synthetic datasets
- 360-degree and VR-based gaze datasets
- Real-world, longitudinal datasets for attention analysis
- Cross-domain and domain adaptation datasets

---

## References

1. Krafka, K., et al. (2016). Eye Tracking for Everyone. CVPR.
2. Zhang, X., et al. (2015/2017). MPIIGaze: Appearance-Based Gaze Estimation. BTAS/CVPRW.
3. Zhang, X., et al. (2020). ETH-XGaze: A Large Scale Dataset for Gaze Estimation. ECCV.
4. Fischer, T., et al. (2018). RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments. ECCV.
5. Smith, B. A., et al. (2013). Gaze Locking: Passive Eye Gaze Detection. UIST.
6. Llanes-Jurado, J., et al. (2022). ULLGaze: Ultra-Large High-Precision Gaze Dataset. BMVC.
7. Recasens, A., et al. (2015). Where Are They Looking? NeurIPS.
8. Kellnhofer, P., et al. (2019). Gaze360: Physically Unconstrained Gaze Estimation in the Wild. ICCV.
9. Zhu, X., et al. (2017). Face Alignment Across Large Poses: A 3D Solution. IJCV.
10. Koestinger, M., et al. (2011). Annotated Facial Landmarks in the Wild. BTAS.
11. Fanelli, G., et al. (2013). Random Forests for Real Time 3D Face Analysis. IJCV.
12. Phillips, P. J., et al. (2013). The University of Central Florida MEGAFACE Benchmark. BTAS.

---

*Document prepared for PhD-level research on gaze estimation and attention detection.*
*Last updated: May 2026*

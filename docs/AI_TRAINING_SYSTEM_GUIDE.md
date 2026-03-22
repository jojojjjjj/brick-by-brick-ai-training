# AI Training System Guide

> 项目：Brick by Brick（HarmonyOS 番茄钟专注度检测）
> 
> 目标：构建一个跨场景（写代码/阅读/看视频）、跨硬件（Huawei NPU / Apple ANE / Intel NPU / Qualcomm Hexagon）部署的统一专注度模型训练系统。

---

## 1. 系统目标与约束

### 1.1 业务目标
- 输入：
  - 前置摄像头视频帧（5 FPS）
  - 键盘/鼠标活动日志
- 输出：
  - 每 5 秒输出一个 0~100 的专注度分数
- 场景：
  - `0=写代码`
  - `1=阅读`
  - `2=看视频`

### 1.2 技术目标
- 统一模型架构 + 场景 embedding（非多模型切换）
- FP16 训练 + ONNX 导出 + INT8 PTQ
- 一套训练产物适配多端 NPU 后端

### 1.3 训练硬件
- GPU：RTX 5080（16GB GDDR7，Blackwell，360W）
- OS：Linux x86_64

---

## 2. 项目结构与模块职责

```text
brick-by-brick-ai-training/
├── docs/
│   └── AI_TRAINING_SYSTEM_GUIDE.md
├── src/
│   ├── data/
│   │   ├── video_processor.py      # FFmpeg 抽帧
│   │   ├── face_extractor.py       # MediaPipe Face Mesh
│   │   ├── action_recognizer.py    # MMAction2 SlowFast
│   │   ├── behavior_logger.py      # 键鼠日志采集
│   │   └── label_fusion.py         # 多信号融合 + HMM
│   ├── models/
│   │   ├── backbone.py             # MobileNetV3 feature extractor
│   │   ├── scene_embedding.py      # 场景 embedding
│   │   ├── focus_head.py           # 专注度输出头
│   │   ├── mobilenet_gru.py        # 完整模型
│   │   └── ONNX_export.py          # ONNX 导出
│   ├── training/
│   │   ├── dataset.py              # FocusDataset
│   │   ├── trainer.py              # 主训练脚本（FP16）
│   │   ├── active_learning.py      # 主动学习
│   │   ├── eval.py                 # 评估
│   │   └── quantize.py             # INT8 PTQ
│   ├── annotation/
│   │   └── auto_labeler.py         # 自动标注流水线
│   └── deployment/
│       ├── base.py                 # 统一推理接口
│       ├── onnx_inference.py       # ONNX Runtime
│       ├── huawei_cann.py          # 华为 CANN EP
│       ├── apple_coreml.py         # 苹果 CoreML EP
│       ├── intel_openvino.py       # 英特尔 OpenVINO EP
│       └── qualcomm_qnn.py         # 高通 QNN EP
├── configs/
│   ├── training.yaml
│   └── model.yaml
├── scripts/
│   ├── download_youtube.py
│   └── record_self.py
├── data/                           # gitignore
├── checkpoints/                    # gitignore
└── requirements.txt
```

---

## 3. 环境搭建

## 3.1 Python 环境

建议 Python 3.10/3.11。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 3.2 关键依赖建议
- PyTorch + torchvision（CUDA 12.x 对应版本）
- onnx / onnxruntime / onnxruntime-gpu
- mediapipe
- opencv-python
- ffmpeg-python
- mmcv / mmaction2
- scikit-learn
- pandas / numpy
- label-studio-sdk

## 3.3 系统依赖
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git
```

## 3.4 验证 GPU
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. 数据采集与标注流水线

## 4.1 数据来源
1. 自我录制（推荐主数据）：
   - `scripts/record_self.py`
   - FFmpeg + OBS 同步录制
2. YouTube 补充数据：
   - `scripts/download_youtube.py`
   - 使用 `yt-dlp` 下载
3. 行为日志：
   - `src/data/behavior_logger.py` 采集键鼠活动

## 4.2 自动标注流水线（AutoLabeler）
`src/annotation/auto_labeler.py`

执行顺序：
1. 抽帧（FFmpeg）
2. 人脸关键点与姿态估计（MediaPipe Face Mesh）
3. 动作识别（MMAction2 SlowFast）
4. 键鼠行为融合
5. HMM 时序平滑，生成伪标签

## 4.3 Label Studio 人工校正
- 将自动标签样本导入 Label Studio
- 按 session 回写人工标签到 `labels.json`
- 形成「伪标签 + 人工高质量标签」混合数据集

---

## 5. 数据格式规范

## 5.1 Session 目录格式

```text
data/
  session_001/
    frames/
      000001.jpg
      000002.jpg
      ...
    gaze.json
    pose.json
    behavior.json
    labels.json
```

## 5.2 建议字段

### gaze.json（每帧）
```json
[
  {"frame_id": 1, "gaze_x": 0.12, "gaze_y": -0.07, "ear": 0.29},
  {"frame_id": 2, "gaze_x": 0.09, "gaze_y": -0.05, "ear": 0.31}
]
```

### pose.json（每帧）
```json
[
  {"frame_id": 1, "yaw": 2.1, "pitch": -4.8, "roll": 0.3},
  {"frame_id": 2, "yaw": 1.8, "pitch": -5.0, "roll": 0.2}
]
```

### behavior.json（时间窗聚合）
```json
[
  {"frame_id": 1, "keyboard_rate": 0.8, "mouse_rate": 0.5, "behavior": 1.0},
  {"frame_id": 2, "keyboard_rate": 0.1, "mouse_rate": 0.2, "behavior": 0.2}
]
```

### labels.json（监督目标）
```json
[
  {"frame_id": 1, "focus": 82.0},
  {"frame_id": 2, "focus": 79.0}
]
```

## 5.3 时间窗约定（5秒输出）
- 原始帧率：5 FPS
- 5 秒窗口：25 帧
- 训练可用 `max_frames=30`（覆盖 6 秒）
- 线上输出建议：每 25 帧汇聚一次（均值/中位数/最后一步）

---

## 6. 模型设计

## 6.1 Backbone（MobileNetV3-Small）
`src/models/backbone.py`

```python
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

class MobileNetBackbone(nn.Module):
    """MobileNetV3-Small 视频特征提取器
    输入: batch x 3 x H x W
    输出: batch x 576
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)
```

## 6.2 Scene Embedding
`src/models/scene_embedding.py`

```python
class SceneEmbedding(nn.Module):
    """场景 embedding (0=写代码, 1=阅读, 2=看视频)
    输出: batch x 32
    """
    def __init__(self, num_scenes=3, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_scenes, embedding_dim)

    def forward(self, scene_id):
        return self.embedding(scene_id)
```

## 6.3 Focus Head（GRU + MLP）
`src/models/focus_head.py`

```python
class FocusHead(nn.Module):
    """专注度输出头
    输入: video_feat (batch, T, 576) + scene_emb (batch, 32)
    输出: focus_score (batch, T) 每帧 0-100
    """
    def __init__(self, video_dim=576, scene_dim=32, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(video_dim + scene_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, video_feat, scene_emb):
        B, T, _ = video_feat.shape
        scene_emb = scene_emb.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([video_feat, scene_emb], dim=-1)
        out, _ = self.gru(x)
        scores = self.fc(out)
        return scores.squeeze(-1) * 100
```

## 6.4 完整模型
`src/models/mobilenet_gru.py`

```python
class FocusDetectionModel(nn.Module):
    """完整专注度检测模型
    输入:
        frames: batch x T x 3 x H x W
        gaze_pose: batch x T x 7 (gaze_x, gaze_y, yaw, pitch, roll, EAR, behavior)
        scene_id: batch
    输出:
        focus_scores: batch x T
    """
    def __init__(self, num_scenes=3):
        super().__init__()
        self.backbone = MobileNetBackbone(pretrained=True)
        self.scene_emb = SceneEmbedding(num_scenes=num_scenes)
        self.focus_head = FocusHead()

    def forward(self, frames, gaze_pose, scene_id):
        B, T, C, H, W = frames.shape
        frames_flat = frames.view(B*T, C, H, W)
        video_feat = self.backbone(frames_flat)
        video_feat = video_feat.view(B, T, -1)
        scene_vec = self.scene_emb(scene_id)
        combined = torch.cat([video_feat, gaze_pose], dim=-1)
        focus_scores = self.focus_head(combined, scene_vec)
        return focus_scores
```

## 6.5 维度对齐注意事项（必须检查）
当前 `mobilenet_gru.py` 中传入 `FocusHead` 的 `video_feat` 实际是 `576 + 7 = 583` 维（拼接了 `gaze_pose`），因此建议两种实现二选一：

1) 将 `FocusHead(video_dim=583, scene_dim=32)`；或
2) 在 `FocusHead` 内部显式区分 `video_feat` 与 `gaze_pose` 并调整输入维度。

否则会出现 GRU 输入维度不一致错误。

---

## 7. 数据集与 Dataloader

`src/training/dataset.py`

```python
class FocusDataset(torch.utils.data.Dataset):
    """专注度检测数据集
    目录结构:
        data/
            session_001/
                frames/     # jpg
                gaze.json
                pose.json
                behavior.json
                labels.json
    """
    def __init__(self, data_root, scene_id, max_frames=30):
        ...
```

实现要点：
- 对齐 frame 与各 JSON 信号（按 `frame_id`）
- 输出：
  - `frames`: `(T, 3, H, W)`
  - `gaze_pose`: `(T, 7)`
  - `scene_id`: `()` 或 `(1,)`
  - `labels`: `(T,)`
- `max_frames` 内截断/补齐
- 图像归一化统一（训练/推理一致）

---

## 8. 训练流程

## 8.1 配置文件
`configs/training.yaml`

```yaml
model:
  name: mobilenet_gru
  num_scenes: 3
  max_frames: 30

training:
  batch_size: 8
  num_epochs: 50
  learning_rate: 1e-3
  optimizer: adamw
  fp16: true
  gradient_clip: 1.0
```

## 8.2 训练函数要求
`src/training/trainer.py`

```python
def train_one_epoch(model, dataloader, optimizer, device, scaler):
    """FP16 混合精度训练
    损失: MSELoss
    梯度裁剪: max_norm=1.0
"""
```

建议细节：
- AMP：`torch.cuda.amp.autocast` + `GradScaler`
- 梯度裁剪：`torch.nn.utils.clip_grad_norm_`
- 损失函数：`MSELoss`（回归 0~100）
- 监控：Train/Val MSE、MAE、Pearson

## 8.3 启动训练
```bash
python -m src.training.trainer --config configs/training.yaml
```

## 8.4 Checkpoint 策略
- 每 N step 保存 `last.pt`
- 每 epoch 保存 `epoch_{k}.pt`
- 以验证集最优保存 `best.pt`
- 记录随机种子、git commit、配置快照

---

## 9. 评估与验收

`src/training/eval.py`

建议指标：
- 回归指标：MSE、RMSE、MAE
- 相关性：Pearson / Spearman
- 时序稳定性：
  - 相邻帧差分均值
  - 5 秒窗口波动率
- 场景分桶评估：
  - code / reading / video 单独报表

推荐验收阈值（可按数据迭代调整）：
- 全局 MAE ≤ 8.0
- 5 秒窗口相关性 Pearson ≥ 0.75
- 各场景 MAE 差异 ≤ 2.0

---

## 10. 主动学习闭环

`src/training/active_learning.py`

```python
def select_uncertain_samples(model, unlabeled_dataset, threshold=0.15, n=1000):
    """MC Dropout 不确定性估计
    选择预测方差最大的 n 个样本供人工标注
"""
```

流程：
1. 在未标注池上启用 MC Dropout 多次前向
2. 计算每样本预测方差
3. 选取 top-n 高不确定样本
4. 人工标注后回流训练集
5. 重新训练并对比指标增益

---

## 11. ONNX 导出规范

`src/models/ONNX_export.py`

要求：
- PyTorch → ONNX
- 动态轴支持（可变 T）
- 场景 embedding 固化（可选）

建议导出参数：
- opset: 17（跨 EP 通常更稳）
- dynamic_axes:
  - `frames`: batch, time
  - `gaze_pose`: batch, time
  - `focus_scores`: batch, time

示例命令：
```bash
python -m src.models.ONNX_export \
  --ckpt checkpoints/best.pt \
  --output checkpoints/focus_model.onnx \
  --opset 17 \
  --dynamic-time true
```

导出后验证：
```bash
python - << 'PY'
import onnx
m = onnx.load('checkpoints/focus_model.onnx')
onnx.checker.check_model(m)
print('ONNX OK')
PY
```

---

## 12. INT8 PTQ 量化流程

`src/training/quantize.py`

```python
def ptq_quantize(model, calibration_data, output_path):
    """INT8 PTQ
    1. 读取 FP32
    2. calibration 采集激活值
    3. 导出 INT8 ONNX
"""
```

建议流程：
1. 使用代表性校准集（覆盖三场景）
2. 执行静态量化（per-channel 优先用于 Conv）
3. 对比 FP32/INT8 精度差
4. 记录每 EP 的量化兼容限制

建议验收：
- INT8 相对 FP32 MAE 劣化 ≤ 1.5
- 推理延迟下降显著（目标 > 25%）

---

## 13. 部署层设计（统一接口 + 多 EP）

## 13.1 统一接口
`src/deployment/base.py`

```python
class FocusDetector:
    def load(self, model_path): pass
    def detect(self, frames, gaze_pose, scene_id):
        """frames: list of np.ndarray (HWC, BGR)
           gaze_pose: np.ndarray (T, 7)
           scene_id: int (0/1/2)
           返回: list of float 0-100
        """
        pass
```

## 13.2 ONNX Runtime 基线实现
`src/deployment/onnx_inference.py`

```python
class ONNXFocusDetector(FocusDetector):
    """ONNX Runtime CPU/GPU
    支持: CPUExecutionProvider, CUDAExecutionProvider
"""
```

## 13.3 各 NPU 后端
- Huawei: `huawei_cann.py`（CANN EP）
- Apple: `apple_coreml.py`（CoreML EP）
- Intel: `intel_openvino.py`（OpenVINO EP）
- Qualcomm: `qualcomm_qnn.py`（QNN EP）

部署建议：
1. 先用 ORT CPU/GPU 跑通一致性
2. 再切换各 EP 做兼容修正
3. 固化预处理/后处理逻辑，确保跨端一致

---

## 14. 端到端执行清单（Runbook）

## 14.1 一次完整训练发布

1) 环境安装
```bash
pip install -r requirements.txt
```

2) 数据准备（录制 / 下载 / 自动标注 / 人工校正）

3) 训练
```bash
python -m src.training.trainer --config configs/training.yaml
```

4) 评估
```bash
python -m src.training.eval --ckpt checkpoints/best.pt
```

5) ONNX 导出
```bash
python -m src.models.ONNX_export --ckpt checkpoints/best.pt --output checkpoints/focus_model.onnx
```

6) INT8 PTQ
```bash
python -m src.training.quantize --input checkpoints/focus_model.onnx --output checkpoints/focus_model_int8.onnx
```

7) 多 EP 验证
```bash
python -m src.deployment.onnx_inference --model checkpoints/focus_model.onnx
python -m src.deployment.huawei_cann --model checkpoints/focus_model_int8.onnx
python -m src.deployment.apple_coreml --model checkpoints/focus_model_int8.onnx
python -m src.deployment.intel_openvino --model checkpoints/focus_model_int8.onnx
python -m src.deployment.qualcomm_qnn --model checkpoints/focus_model_int8.onnx
```

---

## 15. 质量保障与工程规范

## 15.1 可复现性
- 固定随机种子（Python/Numpy/Torch）
- 固定数据切分文件
- 所有实验保存 config 与 commit hash

## 15.2 日志与追踪
- 记录 loss、lr、梯度范数、GPU 显存
- 保存每场景评估报表
- 导出混淆样本（低分高专注 / 高分低专注）供复核

## 15.3 隐私与合规
- 原始视频仅本地安全存储
- 匿名化处理（人脸关键点替代原图用于共享）
- 数据最小化保留策略（仅保留训练所需特征）

---

## 16. 常见问题（FAQ）

1) **训练 loss 不下降**
- 检查 label 分布是否偏斜
- 检查 `frames` 与 `gaze_pose` 的时间对齐
- 检查学习率是否过高

2) **导出 ONNX 失败**
- 降低 opset 或升级 onnx/torch 版本
- 避免导出不支持的动态控制流

3) **NPU 端精度下降明显**
- 校准集覆盖不足
- 使用 per-channel 量化
- 检查预处理归一化是否一致

4) **推理延迟过高**
- 减小输入分辨率
- 减少序列长度 T
- 启用 INT8 + NPU EP

5) **维度报错（最常见）**
- 重点检查 `video_feat + gaze_pose + scene_emb` 维度与 GRU 输入是否一致

---

## 17. 推荐里程碑

- M1：打通数据采集 + 自动标注 + baseline 训练
- M2：完成三场景稳定收敛（FP32/FP16）
- M3：ONNX 导出 + ORT 跨平台一致性
- M4：INT8 PTQ + 四类 NPU 部署联调
- M5：主动学习闭环上线，持续提效

---

## 18. 最小可用命令集合

```bash
# 训练
python -m src.training.trainer --config configs/training.yaml

# 评估
python -m src.training.eval --ckpt checkpoints/best.pt

# 导出 ONNX
python -m src.models.ONNX_export --ckpt checkpoints/best.pt --output checkpoints/focus_model.onnx

# PTQ 量化
python -m src.training.quantize --input checkpoints/focus_model.onnx --output checkpoints/focus_model_int8.onnx
```

---

## 19. 附：关键实现片段（需求对齐）

### `src/training/active_learning.py`
```python
def select_uncertain_samples(model, unlabeled_dataset, threshold=0.15, n=1000):
    """MC Dropout 不确定性估计
    选择预测方差最大的 n 个样本供人工标注
"""
```

### `src/annotation/auto_labeler.py`
```python
class AutoLabeler:
    """自动标注流水线
    1. 抽帧 (FFmpeg)
    2. MediaPipe Face Mesh
    3. MMAction2 SlowFast
    4. 键鼠融合
    5. HMM 时序平滑
"""
```

### `src/deployment/base.py`
```python
class FocusDetector:
    def load(self, model_path): pass
    def detect(self, frames, gaze_pose, scene_id):
        """frames: list of np.ndarray (HWC, BGR)
           gaze_pose: np.ndarray (T, 7)
           scene_id: int (0/1/2)
           返回: list of float 0-100
        """
        pass
```

### `src/deployment/onnx_inference.py`
```python
class ONNXFocusDetector(FocusDetector):
    """ONNX Runtime CPU/GPU
    支持: CPUExecutionProvider, CUDAExecutionProvider
"""
```

---

本指南可直接作为项目训练与部署的标准执行文档。
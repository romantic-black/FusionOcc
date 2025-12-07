# Occupancy 预测头和 3D Encoder 架构图说明

本文档提供详细的架构图说明，包含所有张量的维度信息和处理流程。

## 1. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FusionOcc: Occupancy Prediction Architecture             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐
│ Image 3D Feat│         │ LiDAR Feat   │
│ [B,32,D,H,W] │         │ [B,32,D,H,W] │
│ D=128        │         │ D=128        │
│ H=W=1600     │         │ H=W=1600     │
└──────┬───────┘         └──────┬───────┘
       │                        │
       └────────┬───────────────┘
                │
         ┌──────▼──────┐
         │ Concatenate │
         │  (dim=1)    │
         │ torch.cat   │
         └──────┬──────┘
                │
    ┌───────────▼───────────┐
    │  3D Fused Features    │
    │  [B, 64, D, H, W]     │
    │  64 = 32(img) + 32(l) │
    │  ⚠️ Simple Concat     │
    │  No learned fusion    │
    └───────────┬───────────┘
                │
    ┌───────────▼─────────────────────────────────────────────┐
    │              3D Encoder (occ_encoder)                    │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  ┌──────────────────────────────────────────────────┐   │
    │  │ Backbone: CustomResNet3D                         │   │
    │  ├──────────────────────────────────────────────────┤   │
    │  │                                                  │   │
    │  │  Stage 0: BasicBlock3D × 1, stride=1            │   │
    │  │  └─→ [B, 64, 128, 1600, 1600]                   │   │
    │  │       │                                          │   │
    │  │  Stage 1: BasicBlock3D × 2, stride=2            │   │
    │  │  └─→ [B, 128, 64, 800, 800]                     │   │
    │  │       │                                          │   │
    │  │  Stage 2: BasicBlock3D × 3, stride=2            │   │
    │  │  └─→ [B, 256, 32, 400, 400]                     │   │
    │  │                                                  │   │
    │  └──────────────────────────────────────────────────┘   │
    │                     │                                    │
    │  ┌──────────────────▼──────────────────────────────────┐│
    │  │ Neck: LSSFPN3D                                      ││
    │  ├──────────────────────────────────────────────────────┤│
    │  │                                                      ││
    │  │  1. x_16: Trilinear Upsample ×2                     ││
    │  │     [B,128,64,800,800] → [B,128,128,1600,1600]     ││
    │  │                                                      ││
    │  │  2. x_32: Trilinear Upsample ×4                     ││
    │  │     [B,256,32,400,400] → [B,256,128,1600,1600]     ││
    │  │                                                      ││
    │  │  3. Concat: [B, 448, 128, 1600, 1600]              ││
    │  │     (64 + 128 + 256 = 448)                         ││
    │  │                                                      ││
    │  │  4. Conv3d(1×1×1) + BN3d + ReLU                    ││
    │  │     [B, 448, ...] → [B, 64, 128, 1600, 1600]      ││
    │  │                                                      ││
    │  └──────────────────────────────────────────────────────┘│
    │                                                          │
    └──────────────────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │  Encoded Features           │
                │  [B, 64, 128, 1600, 1600]   │
                └──────────────┬──────────────┘
                               │
    ┌──────────────────────────▼──────────────────────────────┐
    │          Occupancy Prediction Head                      │
    ├──────────────────────────────────────────────────────────┤
    │                                                          │
    │  ┌────────────────────────────────────────────┐         │
    │  │ Final Conv                                 │         │
    │  │ Conv3d(64→64, kernel=3, padding=1)        │         │
    │  │ [B, 64, 128, 1600, 1600]                  │         │
    │  └────────────────┬───────────────────────────┘         │
    │                   │                                     │
    │  ┌────────────────▼───────────────────────────┐         │
    │  │ Permute (0,4,3,2,1)                       │         │
    │  │ BCDHW → BWHDC                             │         │
    │  │ [B, 1600, 1600, 128, 64]                  │         │
    │  └────────────────┬───────────────────────────┘         │
    │                   │                                     │
    │  ┌────────────────▼───────────────────────────┐         │
    │  │ Predictor (if use_predicter=True)          │         │
    │  │ Linear(64→128) + Softplus                  │         │
    │  │ Linear(128→18)                             │         │
    │  │ [B, 1600, 1600, 128, 18]                   │         │
    │  └────────────────────────────────────────────┘         │
    │                                                          │
    └──────────────────────────┬───────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │  Occupancy Prediction       │
                │  [B, 1600, 1600, 128, 18]   │
                │                             │
                │  18 Classes:                │
                │  0-10: Objects              │
                │  11-16: Surfaces            │
                │  17: Free space             │
                └──────────────┬──────────────┘
                               │
                ┌──────────────▼──────────────┐
                │  Loss L_occ                 │
                │  CrossEntropyLoss           │
                │  Input: logits [N, 18]      │
                │  Target: labels [N]         │
                │  Optional: mask_camera      │
                └─────────────────────────────┘
```

## 2. 详细维度变化表

| 阶段 | 操作 | 输入维度 | 输出维度 | 说明 |
|------|------|---------|---------|------|
| **输入** | - | - | - | - |
| | Image Features | - | `[B, 32, 128, 1600, 1600]` | 从图像分支得到的3D特征 |
| | LiDAR Features | - | `[B, 32, 128, 1600, 1600]` | 从LiDAR分支得到的3D特征 |
| **融合** | Concatenate | `[B, 32, ...]` × 2 | `[B, 64, 128, 1600, 1600]` | 简单通道拼接 |
| **3D Encoder Backbone** | Stage 0 | `[B, 64, 128, 1600, 1600]` | `[B, 64, 128, 1600, 1600]` | stride=1, 保持尺寸 |
| | Stage 1 | `[B, 64, 128, 1600, 1600]` | `[B, 128, 64, 800, 800]` | stride=2, 下采样2倍 |
| | Stage 2 | `[B, 128, 64, 800, 800]` | `[B, 256, 32, 400, 400]` | stride=2, 下采样2倍 |
| **3D Encoder Neck** | Upsample x_16 | `[B, 128, 64, 800, 800]` | `[B, 128, 128, 1600, 1600]` | Trilinear ×2 |
| | Upsample x_32 | `[B, 256, 32, 400, 400]` | `[B, 256, 128, 1600, 1600]` | Trilinear ×4 |
| | Concat | 3个特征图 | `[B, 448, 128, 1600, 1600]` | 64+128+256=448 |
| | Conv3d(1×1×1) | `[B, 448, ...]` | `[B, 64, 128, 1600, 1600]` | 通道压缩 |
| **预测头** | Final Conv | `[B, 64, 128, 1600, 1600]` | `[B, 64, 128, 1600, 1600]` | Conv3d(kernel=3) |
| | Permute | `[B, 64, 128, 1600, 1600]` | `[B, 1600, 1600, 128, 64]` | BCDHW→BWHDC |
| | Predictor | `[B, 1600, 1600, 128, 64]` | `[B, 1600, 1600, 128, 18]` | Linear MLP |
| **输出** | Softmax | `[B, ..., 18]` | `[B, ..., 18]` | 类别概率 |
| | Argmax | `[B, ..., 18]` | `[B, ..., 1]` | 预测类别 |

## 3. 关键组件详解

### 3.1 BasicBlock3D 结构

```
输入 x [B, C_in, D, H, W]
  │
  ├─→ Conv3d(3×3×3, C_in→C_out, stride=s) + BN3d + ReLU
  │
  └─→ Conv3d(3×3×3, C_out→C_out, stride=1) + BN3d
      │
      └─→ + (残差连接)
          │
          └─→ ReLU
              │
              └─→ 输出 [B, C_out, D', H', W']
```

### 3.2 LSSFPN3D 详细流程

```
输入: 三个不同尺度的特征
  │
  ├─→ x_8:  [B, 64, 128, 1600, 1600]  (保持不变)
  │
  ├─→ x_16: [B, 128, 64, 800, 800]
  │         └─→ Trilinear Upsample ×2
  │             └─→ [B, 128, 128, 1600, 1600]
  │
  └─→ x_32: [B, 256, 32, 400, 400]
            └─→ Trilinear Upsample ×4
                └─→ [B, 256, 128, 1600, 1600]

拼接: [B, 448, 128, 1600, 1600]  (64+128+256)

  └─→ Conv3d(1×1×1, 448→64) + BN3d + ReLU
      └─→ 输出: [B, 64, 128, 1600, 1600]
```

### 3.3 Predictor 结构

```
输入: [B, 1600, 1600, 128, 64]
  │
  └─→ 对每个voxel (最后64维)应用:
      │
      ├─→ Linear(64 → 128)
      │
      ├─→ Softplus()  # 平滑的ReLU变体
      │
      └─→ Linear(128 → 18)
          │
          └─→ 输出: [B, 1600, 1600, 128, 18]
```

## 4. Loss 计算详解

### 4.1 输入准备

```
预测值 (preds):
  Shape: [B, 1600, 1600, 128, 18]
  Type: Logits (未归一化)
  ↓ reshape(-1, 18)
  Shape: [N, 18]  (N = B×1600×1600×128)

真实值 (voxel_semantics):
  Shape: [B, 1600, 1600, 128]
  Type: Long, 范围 [0, 17]
  ↓ reshape(-1)
  Shape: [N]

掩码 (mask_camera, 可选):
  Shape: [B, 1600, 1600, 128]
  Type: Bool
  ↓ reshape(-1)
  Shape: [N]
```

### 4.2 Loss 计算

**不使用掩码**:
```python
loss = CrossEntropyLoss(preds, labels)
# 对所有 N 个 voxel 计算损失
```

**使用掩码**:
```python
# 只计算掩码为 True 的 voxel
valid_preds = preds[mask_camera]  # [M, 18], M < N
valid_labels = labels[mask_camera]  # [M]
num_samples = mask_camera.sum()

loss = CrossEntropyLoss(valid_preds, valid_labels, avg_factor=num_samples)
```

### 4.3 数学公式

```
L_occ = -1/N * Σ log(exp(logits[i, y_i]) / Σ_j exp(logits[i, j]))

其中:
- N: 有效voxel数量
- logits[i, j]: 第i个voxel对第j个类别的logit
- y_i: 第i个voxel的真实类别
```

## 5. 关于 L+C 融合的说明

### 5.1 当前实现

**位置**: `mmdet3d/models/detectors/fusion_occ.py:156`

```python
fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
```

**特点**:
- ✅ **有拼接**: LiDAR 和 Camera 特征确实被组合在一起
- ❌ **无学习**: 没有可学习的融合权重或注意力机制
- ❌ **无交互**: 两个模态的特征在拼接前没有交互

### 5.2 为什么可能有效

尽管是简单拼接，但依然可能有效的理由：

1. **3D Encoder 的后处理**: 后续的 3D Encoder 可以通过卷积操作学习如何利用两个模态的信息
2. **空间对齐**: 两种特征已经在相同的 3D 空间网格中，拼接是合理的
3. **特征维度**: 每个模态提供 32 通道，拼接后 64 通道，给模型足够的表示空间

### 5.3 改进方向

如果想要更好的 L+C 融合，可以考虑：

1. **跨模态注意力**:
   ```python
   # 伪代码
   attention_weights = CrossModalAttention(img_feat, lidar_feat)
   fused_feat = attention_weights * img_feat + (1-attention_weights) * lidar_feat
   ```

2. **可学习的加权融合**:
   ```python
   # 伪代码
   alpha = torch.sigmoid(self.fusion_net(torch.cat([img_feat, lidar_feat], dim=1)))
   fused_feat = alpha * img_feat + (1 - alpha) * lidar_feat
   ```

3. **特征重标定**:
   ```python
   # 伪代码
   img_feat_calibrated = self.calibrate_img(img_feat, lidar_feat)
   lidar_feat_calibrated = self.calibrate_lidar(lidar_feat, img_feat)
   fused_feat = torch.cat([img_feat_calibrated, lidar_feat_calibrated], dim=1)
   ```

## 6. 使用 Draw.io 查看架构图

1. 打开 [draw.io](https://app.diagrams.net/)
2. 选择 "File" → "Open from" → "Device"
3. 选择 `docs/occupancy_head_structure.drawio` 文件
4. 查看详细的交互式架构图

架构图包含：
- 所有组件的详细标注
- 张量维度信息
- 处理流程箭头
- 重要说明和警告

## 7. 相关文件

- **模型实现**: `mmdet3d/models/detectors/fusion_occ.py`
- **3D Encoder Backbone**: `mmdet3d/models/backbones/resnet.py`
- **3D Encoder Neck**: `mmdet3d/models/necks/lss_fpn.py`
- **配置文件**: `configs/fusion_occ/fusion_occ.py`
- **详细文档**: `docs/occupancy_head_architecture.md`


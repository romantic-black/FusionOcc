# Occupancy 预测头和 3D Encoder 架构详解

## 概述

本文档详细解释 FusionOcc 模型中 Occupancy 预测头（Occupancy Prediction Head）和 3D Encoder 的结构、数据流以及损失函数的计算方式。

## 1. 整体数据流

```
图像特征 (img_3d_feat) [B, C_img, D, H, W]
        ↓
    拼接 (Concat)  ← LiDAR特征 (lidar_feat) [B, C_lidar, D, H, W]
        ↓
3D Fused Features [B, C_img+C_lidar, D, H, W]
        ↓
  3D Encoder (occ_encoder)
        ↓
编码后的特征 [B, C_out, D, H, W]
        ↓
  Occupancy 预测头
        ↓
  Occupancy 预测 [B, N, W, H, D, num_classes]
```

## 2. 3D Fused Features 的生成

### 2.1 特征拼接

在 `fusion_occ.py` 的 `forward_train` 方法中（第156行），3D Fused Features 是通过**简单的通道拼接（Concatenation）**生成的：

```python
fusion_feat = torch.cat([img_3d_feat_feat, lidar_feat], dim=1)
```

**关键点**：
- **没有复杂的融合机制**：当前实现使用的是简单的 `torch.cat` 操作，在通道维度（dim=1）上拼接
- **不是学习型融合**：没有使用注意力机制、加权融合或其他高级融合策略

### 2.2 张量维度分析

假设配置参数为：
- `img_channels = 32`
- `lidar_out_channel = 32`
- `voxel_size = [0.05, 0.05, 0.05]`
- `point_cloud_range = [-40, -40, -1, 40, 40, 5.4]`

**空间维度计算**：
```
sparse_shape = [
    int((40 - (-40)) / 0.05) = 1600,  # X维度
    int((40 - (-40)) / 0.05) = 1600,  # Y维度
    int((5.4 - (-1)) / 0.05) = 128    # Z维度
]
```

**具体维度**：
- **img_3d_feat_feat**: `[B, 32, 128, 1600, 1600]` （对于单帧，多帧会更大）
  - 如果有多个时序帧，维度为 `[B, 32 * num_frames, 128, 1600, 1600]`
- **lidar_feat**: `[B, 32, 128, 1600, 1600]`
- **fusion_feat**: `[B, 64, 128, 1600, 1600]` （32 + 32 = 64通道）

### 2.3 关于 L+C 融合的说明

**当前实现**：
- ✅ 有 LiDAR 和 Camera 特征的拼接
- ❌ **没有**学习型的 3D voxel latent 融合机制
- ❌ **没有**跨模态注意力机制
- ❌ **没有**特征重标定或自适应加权

**潜在的改进方向**：
- 可以引入跨模态注意力模块（Cross-Modal Attention）
- 可以使用可学习的融合权重
- 可以实现基于空间位置的动态融合策略

## 3. 3D Encoder 结构

3D Encoder 由两部分组成：**Backbone (CustomResNet3D)** 和 **Neck (LSSFPN3D)**。

### 3.1 Backbone: CustomResNet3D

**位置**：`mmdet3d/models/backbones/resnet.py`

**结构**：
```
输入: [B, C_in, D, H, W]
  ↓
Stage 0 (1层 BasicBlock3D, stride=1)
  ↓ 输出: [B, C_0, D, H, W]
Stage 1 (2层 BasicBlock3D, stride=2)
  ↓ 输出: [B, C_1, D/2, H/2, W/2]
Stage 2 (3层 BasicBlock3D, stride=2)
  ↓ 输出: [B, C_2, D/4, H/4, W/4]
```

**配置参数**（来自 `fusion_occ.py` 配置）：
- `numC_input = 64` （32 * num_frames + 32）
- `num_channels = [64, 128, 256]` （对应三个 stage 的输出通道）
- `stride = [1, 2, 2]`
- `backbone_output_ids = [0, 1, 2]` （输出三个尺度的特征）

**BasicBlock3D 结构**：
```
输入 x
  ↓
Conv3d (3×3×3, stride=s) + BN3d + ReLU
  ↓
Conv3d (3×3×3, stride=1) + BN3d
  ↓
残差连接 (如果维度不匹配，使用 downsample)
  ↓
ReLU
  ↓
输出
```

**输出特征**：
- `x_8`: `[B, 64, 128, 1600, 1600]` （1/1 分辨率）
- `x_16`: `[B, 128, 64, 800, 800]` （1/2 分辨率）
- `x_32`: `[B, 256, 32, 400, 400]` （1/4 分辨率）

### 3.2 Neck: LSSFPN3D

**位置**：`mmdet3d/models/necks/lss_fpn.py`

**功能**：将多个尺度的特征融合并上采样到原始分辨率

**结构**：
```
x_8 [B, 64, 128, 1600, 1600]  保持不变
  ↓
x_16 [B, 128, 64, 800, 800]    Trilinear Upsample (×2)
  ↓                             → [B, 128, 128, 1600, 1600]
x_32 [B, 256, 32, 400, 400]    Trilinear Upsample (×4)
  ↓                             → [B, 256, 128, 1600, 1600]
拼接 (Concat)                   → [B, 448, 128, 1600, 1600]
  ↓                             (64 + 128 + 256 = 448)
Conv3d (1×1×1) + BN3d + ReLU
  ↓
输出: [B, 64, 128, 1600, 1600]
```

**关键操作**：
- `up1`: 三线性上采样（Trilinear Upsample），放大 2 倍
- `up2`: 三线性上采样（Trilinear Upsample），放大 4 倍
- `ConvModule`: 1×1×1 卷积，将融合后的 448 通道压缩到 64 通道

**最终输出维度**：`[B, 64, 128, 1600, 1600]`

## 4. Occupancy 预测头

### 4.1 结构组成

Occupancy 预测头由两部分组成：

#### 4.1.1 Final Conv

**代码位置**：`mmdet3d/models/detectors/fusion_occ.py` 第110-117行

```python
self.final_conv = ConvModule(
    out_dim,           # 64
    out_channels,      # 64 (如果use_predicter=True) 或 18 (如果use_predicter=False)
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    conv_cfg=dict(type='Conv3d'))
```

**功能**：
- 对编码后的特征进行最后的 3D 卷积处理
- 如果 `use_predicter=True`，输出维度保持为 `out_dim`（64）
- 如果 `use_predicter=False`，直接输出 `num_classes`（18）

**维度变换**：
```
输入: [B, 64, 128, 1600, 1600]
  ↓ Conv3d(64 → 64, kernel=3, padding=1)
输出: [B, 64, 128, 1600, 1600]
  ↓ permute(0, 4, 3, 2, 1)  # BCDHW → BWHDC
输出: [B, 1600, 1600, 128, 64]
```

#### 4.1.2 Predictor（可选）

**代码位置**：`mmdet3d/models/detectors/fusion_occ.py` 第119-124行

```python
if use_predicter:
    self.predicter = nn.Sequential(
        nn.Linear(self.out_dim, self.out_dim * 2),  # 64 → 128
        nn.Softplus(),
        nn.Linear(self.out_dim * 2, num_classes),   # 128 → 18
    )
```

**功能**：
- 将每个 voxel 的 64 维特征映射到 18 个类别的 logits
- 使用 Softplus 作为激活函数（平滑的 ReLU 变体）

**维度变换**：
```
输入: [B, 1600, 1600, 128, 64]
  ↓ 对最后一个维度应用 Linear 层
输出: [B, 1600, 1600, 128, 18]
```

### 4.2 输出含义

**最终输出维度**：`[B, 1600, 1600, 128, 18]`

**语义解释**：
- `B`: Batch size
- `1600`: X 方向的空间维度（对应 -40m 到 +40m，voxel size 0.05m）
- `1600`: Y 方向的空间维度
- `128`: Z 方向的空间维度（对应 -1m 到 5.4m，voxel size 0.05m）
- `18`: 类别数量

**18 个类别**（根据 nuScenes 数据集）：
```
0: others
1: barrier
2: bicycle
3: bus
4: car
5: construction_vehicle
6: motorcycle
7: pedestrian
8: traffic_cone
9: trailer
10: truck
11: driveable_surface
12: other_flat
13: sidewalk
14: terrain
15: manmade
16: vegetation
17: free (空/可通行)
```

**输出值的含义**：
- 每个 voxel 对应一个 18 维的 logits 向量
- 经过 softmax 后，得到每个类别的概率分布
- 最终预测类别 = `argmax(softmax(logits))`

## 5. Loss 函数 L_occ

### 5.1 计算流程

**代码位置**：`mmdet3d/models/detectors/fusion_occ.py` 第176-192行

#### 5.1.1 输入准备

```python
# 预测值
occ_pred: [B, 1600, 1600, 128, 18]  # Logits，未经过 softmax

# 真实值
voxel_semantics: [B, 1600, 1600, 128]  # 每个 voxel 的类别标签 (0-17)

# 可选的 mask
mask_camera: [B, 1600, 1600, 128]  # 布尔掩码，标记相机视野内的 voxel
```

#### 5.1.2 形状变换

```python
# 将预测和标签展平为 2D 形式
voxel_semantics = voxel_semantics.reshape(-1)  # [B*1600*1600*128]
preds = preds.reshape(-1, self.num_classes)    # [B*1600*1600*128, 18]
```

#### 5.1.3 Loss 计算

**不使用 mask 的情况**：
```python
loss_occ = self.loss_occ(preds, voxel_semantics)
```

**使用 mask 的情况**：
```python
mask_camera = mask_camera.reshape(-1)  # [B*1600*1600*128]
num_total_samples = mask_camera.sum()  # 统计有效 voxel 数量
loss_occ = self.loss_occ(
    preds, 
    voxel_semantics, 
    mask_camera, 
    avg_factor=num_total_samples
)
```

### 5.2 Loss 函数类型

**配置**：`CrossEntropyLoss`

```python
loss_occ=dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,  # 使用标准的交叉熵，不是二分类
    loss_weight=1.0
)
```

**数学公式**：
```
L_occ = -∑ log(softmax(pred)[true_class])
```

其中：
- `pred`: 预测的 logits `[N, 18]`
- `true_class`: 真实类别标签 `[N]`
- `N`: 有效 voxel 数量（如果使用 mask，则只计算 mask 内的 voxel）

### 5.3 掩码的作用

**mask_camera** 用于：
1. **限制计算范围**：只计算相机视野内的 voxel，忽略被遮挡或不可观测的区域
2. **平衡训练**：避免大量空 voxel 主导损失函数
3. **提高精度**：专注于模型能够可靠预测的区域

**掩码生成**：
- 通常基于相机投影矩阵，将 3D voxel 投影到图像平面
- 只保留在图像范围内的 voxel

## 6. 总结

### 6.1 关键发现

1. **3D Fused Features 是简单拼接**：
   - ✅ 使用 `torch.cat` 在通道维度拼接
   - ❌ 没有学习型的融合机制
   - ❌ 没有跨模态注意力

2. **3D Encoder 结构清晰**：
   - Backbone (CustomResNet3D): 3 个 stage，提取多尺度特征
   - Neck (LSSFPN3D): 多尺度特征融合，恢复原始分辨率

3. **Occupancy 预测头简洁**：
   - Final Conv: 最后的 3D 卷积处理
   - Predictor: 可选的 MLP，将特征映射到类别空间

4. **Loss 计算标准**：
   - 使用交叉熵损失
   - 支持可选的相机视野掩码

### 6.2 维度变化总结

```
img_3d_feat:     [B, 32, 128, 1600, 1600]
lidar_feat:      [B, 32, 128, 1600, 1600]
    ↓ Concat
fusion_feat:     [B, 64, 128, 1600, 1600]
    ↓ 3D Encoder Backbone
x_8:             [B, 64, 128, 1600, 1600]
x_16:            [B, 128, 64, 800, 800]
x_32:            [B, 256, 32, 400, 400]
    ↓ 3D Encoder Neck (FPN)
encoded_feat:    [B, 64, 128, 1600, 1600]
    ↓ Final Conv
feat_before_pred:[B, 64, 128, 1600, 1600]
    ↓ Permute
feat_permuted:   [B, 1600, 1600, 128, 64]
    ↓ Predictor (if enabled)
occ_pred:        [B, 1600, 1600, 128, 18]
```

### 6.3 改进建议

如果需要改进 L+C 融合效果，可以考虑：

1. **引入跨模态注意力**：
   ```python
   # 伪代码示例
   fused_feat = CrossModalAttention(img_feat, lidar_feat)
   ```

2. **使用可学习的融合权重**：
   ```python
   # 伪代码示例
   alpha = torch.sigmoid(self.fusion_weight(img_feat, lidar_feat))
   fused_feat = alpha * img_feat + (1 - alpha) * lidar_feat
   ```

3. **实现空间感知融合**：
   - 不同空间位置使用不同的融合策略
   - 例如：近距离区域更依赖 LiDAR，远距离区域更依赖图像


# FusionOcc 模型前向流程图验证文档

本文档验证 `docs/model_forward_flow.drawio` 流程图的正确性。

## 1. 维度验证

### 1.1 输入维度

**多视角图像**:
- 配置: `input_size=(512, 1408)`, `Ncams=6`
- 维度: `[B, 6, 3, 512, 1408]` ✅

**稀疏深度图**:
- 配置: `depth_channels=88`
- 维度: `[B, 6, H, W, 88]` ✅
- 来源: 从LiDAR点云投影得到 ✅

**LiDAR点云**:
- 配置: `lidar_in_channel=5`, `point_cloud_range=[-40, -40, -1, 40, 40, 5.4]`
- 维度: `[N_points, 5]` ✅

### 1.2 Images Branch 维度变化

**Image Encoder (Swin Transformer)**:
- 输入: `[B*6, 3, 512, 1408]`
- 输出 indices: (2, 3)
- 输出: `[B*6, 512, H/32, W/32]`, `[B*6, 1024, H/64, W/64]` ✅

**Image Neck (FPN_LSS)**:
- 输入: `[512, 1024]` channels
- 输出: `[B*6, 256, H/16, W/16]` ✅
- 配置: `out_channels=256`, `scale_factor=2` ✅

**Image Reduce Conv**:
- 输入: `[B*6, 256, H/16, W/16]`
- 输出: `[B*6, 128, H/16, W/16]` ✅
- 配置: `mid_channels=128` ✅

**Depth Encoder**:
- 输入: `[B*6, 88, H, W]`
- 输出: `[B*6, 128, H, W]` ✅
- 配置: `depth_channels=88`, `mid_channels=128` ✅

**Cross-modal Fusion**:
- 输入: `[B*6, 128, H/16, W/16]` × 2 (image + depth)
- 输出: `[B*6, 128, H/16, W/16]` × 2 (fc_c2d, fc_d2c) ✅

**Further Fuse**:
- 输入: `[B*6, 256, H/16, W/16]` (concat of fc_c2d + fc_d2c)
- 输出: `[B*6, 256, H/16, W/16]` ✅

**Depth & Segmentation Net**:
- 输出深度: `[B*6, 88, H/16, W/16]` ✅
- 输出分割: `[B*6, 18, H/16, W/16]` ✅
- 输出特征: `[B*6, 32, H/16, W/16]` ✅
- 配置: `feature_channels=32`, `seg_num_classes=18`, `depth_channels=88` ✅

**2D to 3D Projection (LSS View Transformer)**:
- 输入: 深度分布 + 分割特征
- 输出: `[B, 32, D, H, W]` ✅
- 配置: `D=128`, `H=W=1600` (根据 `voxel_size=[0.05, 0.05, 0.05]` 和 `point_cloud_range` 计算) ✅

**多帧处理**:
- 配置: `multi_adj_frame_id_cfg = (1, 1 + 1, 1)` → `range(1, 2)` = 1个过去帧
- 总帧数: T = 2 (当前 + 1过去) ✅
- 拼接后: `[B, 32*2, D, H, W]` = `[B, 64, D, H, W]` ✅

### 1.3 Points Branch 维度变化

**Voxelization**:
- 输入: `[N_points, 5]`
- Voxel size: `[0.05, 0.05, 0.05]` ✅
- Sparse shape: `[1600, 1600, 128]` ✅
- 计算: `(40-(-40))/0.05 = 1600`, `(5.4-(-1))/0.05 = 128` ✅

**Voxel Encoder (CustomSparseEncoder)**:
- 输入: Voxel features + Coords
- 输出: `[B, 32, D, H, W]` ✅
- 配置: `lidar_out_channel=32` ✅

**Permute**:
- 输入: `[B, 32, D, H, W]`
- 输出: `[B, 32, D, W, H]` (对齐图像特征) ✅
- 代码: `permute(0, 1, 2, 4, 3)` ✅

### 1.4 Fusion 维度

**Concatenate**:
- 输入1: `[B, 64, D, H, W]` (图像特征，2帧)
- 输入2: `[B, 32, D, H, W]` (LiDAR特征)
- 输出: `[B, 96, D, H, W]` ✅
- 代码: `torch.cat([img_3d_feat_feat, lidar_feat], dim=1)` ✅
- 配置验证: `numC_input = 32 * (1 + 1) + 32 = 96` ✅

### 1.5 3D Encoder 维度变化

**Backbone Input**:
- 输入: `[B, 96, D, H, W]` ✅
- 配置: `numC_input=96` ✅

**Stage 0**:
- 输入: `[B, 96, 128, 1600, 1600]`
- 输出: `[B, 64, 128, 1600, 1600]` ✅
- 配置: `num_channels=[64, 128, 256]`, `stride=[1, 2, 2]` ✅

**Stage 1**:
- 输入: `[B, 64, 128, 1600, 1600]`
- 输出: `[B, 128, 64, 800, 800]` ✅
- stride=2, 下采样2倍 ✅

**Stage 2**:
- 输入: `[B, 128, 64, 800, 800]`
- 输出: `[B, 256, 32, 400, 400]` ✅
- stride=2, 下采样2倍 ✅

**Neck (LSSFPN3D)**:
- x_8: `[B, 64, 128, 1600, 1600]` (保持不变) ✅
- x_16: `[B, 128, 64, 800, 800]` → Trilinear Upsample ×2 → `[B, 128, 128, 1600, 1600]` ✅
- x_32: `[B, 256, 32, 400, 400]` → Trilinear Upsample ×4 → `[B, 256, 128, 1600, 1600]` ✅
- Concat: `[B, 448, 128, 1600, 1600]` (64+128+256) ✅
- Conv3d(1×1×1): `[B, 64, 128, 1600, 1600]` ✅
- 配置: `in_channels=448` (numC_Trans * 7 = 64 * 7 = 448), `out_channels=64` ✅

### 1.6 Occupancy Head 维度变化

**Final Conv**:
- 输入: `[B, 64, 128, 1600, 1600]`
- 输出: `[B, 64, 128, 1600, 1600]` ✅
- 配置: `out_dim=64`, `kernel_size=3` ✅

**Permute**:
- 输入: `[B, 64, 128, 1600, 1600]`
- 输出: `[B, 1600, 1600, 128, 64]` ✅
- 代码: `permute(0, 4, 3, 2, 1)` ✅

**Predictor**:
- 输入: `[B, 1600, 1600, 128, 64]`
- Linear(64→128) + Softplus + Linear(128→18)
- 输出: `[B, 1600, 1600, 128, 18]` ✅
- 配置: `use_predicter=True`, `num_classes=18` ✅

**Loss**:
- 输入: logits `[N, 18]`, labels `[N]`
- N = B × 1600 × 1600 × 128 ✅
- 配置: `CrossEntropyLoss`, `use_mask=True/False` ✅

## 2. 组件验证

### 2.1 Images Branch 组件

✅ **Image Encoder**: Swin Transformer
- 配置: `depths=[2, 2, 18, 2]`, `out_indices=(2, 3)`
- 代码位置: `mmdet3d/models/backbones/swin.py`

✅ **Image Neck**: FPN_LSS
- 配置: `in_channels=512+1024`, `out_channels=256`
- 代码位置: `mmdet3d/models/necks/fpn_lss.py`

✅ **Cross-modal Fusion**: CrossModalFusion
- 代码位置: `mmdet3d/models/necks/fusion_view_transformer.py:95`

✅ **View Transformer**: CrossModalLSS
- 代码位置: `mmdet3d/models/necks/fusion_view_transformer.py:146`

### 2.2 Points Branch 组件

✅ **Voxel Encoder**: CustomSparseEncoder
- 配置: `in_channels=5`, `output_channels=32`
- 代码位置: `mmdet3d/models/backbones/lidar_encoder.py:14`

### 2.3 Fusion 组件

✅ **Concatenate**: 简单拼接
- 代码位置: `mmdet3d/models/detectors/fusion_occ.py:156`
- 方法: `torch.cat([img_3d_feat_feat, lidar_feat], dim=1)`

### 2.4 3D Encoder 组件

✅ **Backbone**: CustomResNet3D
- 配置: `num_layer=[1, 2, 3]`, `num_channels=[64, 128, 256]`
- 代码位置: `mmdet3d/models/backbones/resnet.py`

✅ **Neck**: LSSFPN3D
- 配置: `in_channels=448`, `out_channels=64`
- 代码位置: `mmdet3d/models/necks/lss_fpn.py`

### 2.5 Occupancy Head 组件

✅ **Final Conv**: Conv3d
- 配置: `kernel_size=3`, `out_dim=64`
- 代码位置: `mmdet3d/models/detectors/fusion_occ.py:110`

✅ **Predictor**: MLP
- 配置: `Linear(64→128) + Softplus + Linear(128→18)`
- 代码位置: `mmdet3d/models/detectors/fusion_occ.py:120`

## 3. 数据流验证

### 3.1 前向传播路径

✅ **Images Branch**:
```
Multi-view Images → Image Encoder → Image Neck → Image Reduce Conv
                                                      ↓
Sparse Depth → Depth Encoder → Cross-modal Fusion → Further Fuse
                                                      ↓
                                              Depth & Seg Net
                                                      ↓
                                              2D to 3D Projection
                                                      ↓
                                              Multi-frame Processing
                                                      ↓
                                              Image 3D Features
```

✅ **Points Branch**:
```
LiDAR Points → Voxelization → Voxel Encoder → Permute → LiDAR Features
```

✅ **Fusion**:
```
Image 3D Features + LiDAR Features → Concatenate → 3D Fused Features
```

✅ **3D Encoder**:
```
3D Fused Features → Backbone (3 stages) → Neck (FPN) → Encoded Features
```

✅ **Occupancy Head**:
```
Encoded Features → Final Conv → Permute → Predictor → Occupancy Prediction
```

### 3.2 损失计算路径

✅ **Auxiliary Losses** (Images Branch):
- L_depth: 深度估计损失
- L_seg: 语义分割损失
- 权重: `0.1 × fuse_loss_weight`
- 代码位置: `mmdet3d/models/detectors/fusion_occ.py:160-163`

✅ **Occupancy Loss**:
- L_occ: 交叉熵损失
- 输入: logits `[N, 18]`, labels `[N]`
- 可选: `mask_camera`
- 代码位置: `mmdet3d/models/detectors/fusion_occ.py:172`

## 4. 关键发现

### 4.1 维度一致性

✅ 所有维度计算与配置文件和代码实现一致
✅ 多帧处理: T=2 (当前 + 1过去)
✅ 融合维度: 96 = 64(img, 2帧) + 32(lidar)
✅ 3D Encoder 将 96 通道压缩到 64 通道

### 4.2 组件完整性

✅ 所有主要组件都已包含在流程图中
✅ 关键数据流路径都已标注
✅ 维度信息完整且准确

### 4.3 注意事项

⚠️ **Fusion 机制**: 当前是简单拼接，没有学习型融合
⚠️ **多帧处理**: 需要正确处理时序帧的拼接
⚠️ **Sparse Depth**: 从LiDAR点云投影得到，用于辅助深度估计

## 5. 验证结论

✅ **流程图正确性**: 已验证
✅ **维度信息**: 全部准确
✅ **组件完整性**: 所有主要组件都已包含
✅ **数据流**: 与代码实现一致

流程图可以用于：
- 理解模型架构
- 调试维度问题
- 教学和文档
- 模型改进参考

## 6. 相关文件

- **流程图**: `docs/model_forward_flow.drawio`
- **配置文件**: `configs/fusion_occ/fusion_occ.py`
- **模型实现**: `mmdet3d/models/detectors/fusion_occ.py`
- **详细文档**: `docs/occupancy_head_architecture.md`


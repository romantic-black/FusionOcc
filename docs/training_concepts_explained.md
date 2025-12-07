# 训练相关概念详解

本文档解释 FusionOcc 训练过程中的两个重要概念：
1. **不使用掩码的版本（version without using mask）**
2. **BEVDet 预训练权重的作用**

---

## 1. 不使用掩码的版本（Version Without Using Mask）

### 1.1 什么是掩码（Mask）？

在 FusionOcc 中，**掩码（mask）**指的是 `mask_camera`，它是一个布尔张量，用于标记哪些 3D voxel 位于相机视野内。

- **形状**：`[B, 1600, 1600, 128]`，与 occupancy 预测的空间维度相同
- **含义**：
  - `True`：该 voxel 在相机视野内，可以被相机观测到
  - `False`：该 voxel 不在相机视野内，无法被相机观测到（可能被遮挡或超出视野范围）

### 1.2 掩码在损失计算中的作用

在训练过程中，`use_mask` 参数控制是否使用掩码来限制损失计算的范围。

#### 使用掩码（`use_mask=True`）

```python
# 代码位置：mmdet3d/models/detectors/fusion_occ.py:179-186
if self.use_mask:
    mask_camera = mask_camera.to(torch.int32)
    voxel_semantics = voxel_semantics.reshape(-1)
    preds = preds.reshape(-1, self.num_classes)
    mask_camera = mask_camera.reshape(-1)
    num_total_samples = mask_camera.sum()  # 只统计有效 voxel 数量
    loss_occ = self.loss_occ(preds, voxel_semantics, mask_camera, avg_factor=num_total_samples)
```

**特点**：
- ✅ 只计算相机视野内的 voxel 的损失
- ✅ 忽略被遮挡或不可观测的区域
- ✅ 专注于模型能够可靠预测的区域
- ⚠️ 训练样本数量减少（只使用视野内的 voxel）

#### 不使用掩码（`use_mask=False`）

```python
# 代码位置：mmdet3d/models/detectors/fusion_occ.py:187-191
else:
    voxel_semantics = voxel_semantics.reshape(-1)
    preds = preds.reshape(-1, self.num_classes)
    loss_occ = self.loss_occ(preds, voxel_semantics)  # 计算所有 voxel
```

**特点**：
- ✅ 计算所有 voxel 的损失，包括相机视野外的区域
- ✅ 训练样本数量更多
- ✅ 模型需要学习预测整个 3D 空间的 occupancy
- ⚠️ 视野外的区域可能缺乏监督信号（因为相机无法观测到）

### 1.3 如何切换到不使用掩码的版本？

在配置文件 `configs/fusion_occ/fusion_occ.py` 中修改：

```python
# 第 38 行
use_mask = False  # 从 True 改为 False
```

然后重新训练几个 epoch：

```bash
./tools/dist_train.sh configs/fusion_occ/fusion_occ.py num_gpu
```

### 1.4 两种版本的性能对比

根据 README.md 中的 Model Zoo 表格：

| 版本 | Mask | mIoU | Checkpoints |
| :---: | :---: | :---: | :---: |
| BaseWoMask | ✖️ | **56.62** | [BaseWoMask](https://drive.google.com/file/d/16ELoDLoDkCYheREJUPiBz2905MHhuVHv/view) |
| BaseMask | ✔️ | 35.94 | (已注释) |

**观察**：
- 不使用掩码的版本（BaseWoMask）性能更好（mIoU: 56.62）
- 这可能是因为：
  1. 更多的训练样本有助于模型学习
  2. 模型需要预测整个 3D 空间，而不仅仅是相机视野内
  3. 在评估时，通常也会评估整个空间，而不仅仅是视野内

### 1.5 为什么需要两种版本？

1. **研究目的**：比较使用掩码和不使用掩码对模型性能的影响
2. **应用场景**：
   - 如果只关心相机视野内的 occupancy，可以使用掩码版本
   - 如果需要预测整个 3D 空间的 occupancy（如用于路径规划），应该使用无掩码版本
3. **训练策略**：可以先使用掩码训练，然后切换到无掩码进行微调

---

## 2. BEVDet 预训练权重的作用

### 2.1 什么是 BEVDet？

**BEVDet**（Bird's Eye View Detection）是一个用于多相机 3D 目标检测的模型。FusionOcc 的架构基于 BEVDet，特别是：

- **图像编码器（Image Backbone）**：Swin Transformer
- **视图变换器（View Transformer）**：将多相机图像特征转换为 BEV 特征
- **BEV 编码器（BEV Encoder）**：处理 BEV 特征

### 2.2 为什么使用 BEVDet 预训练权重？

#### 2.2.1 迁移学习（Transfer Learning）

BEVDet 在 3D 目标检测任务上预训练，学习到了：
- ✅ 如何从多相机图像中提取有效特征
- ✅ 如何将图像特征转换为 BEV 表示
- ✅ 如何理解 3D 空间结构

这些知识对 FusionOcc 的 occupancy 预测任务同样有用，因为：
- 两者都使用多相机输入
- 两者都需要理解 3D 空间
- 两者都使用 BEV 表示

#### 2.2.2 加速训练

使用预训练权重可以：
- 🚀 **减少训练时间**：模型已经有了良好的初始化，不需要从零开始学习基础特征
- 🎯 **提高性能**：预训练权重提供了更好的起点，通常能达到更高的最终性能
- 💰 **节省资源**：减少训练所需的计算资源和时间

### 2.3 如何使用 BEVDet 预训练权重？

#### 步骤 1：下载 BEVDet 预训练权重

从 BEVDet 官方仓库下载预训练权重：
- 仓库地址：https://github.com/HuangJunJie2017/BEVDet
- 通常可以在 Model Zoo 或 Releases 中找到预训练权重文件

#### 步骤 2：修改配置文件

在 `configs/fusion_occ/fusion_occ.py` 文件末尾修改 `load_from` 字段：

```python
# 第 293 行
load_from = "path/to/bevdet_pretrained.pth"  # 替换为你的 BEVDet 权重路径
```

#### 步骤 3：开始训练

```bash
./tools/dist_train.sh configs/fusion_occ/fusion_occ.py num_gpu
```

训练脚本会自动加载预训练权重，初始化模型参数。

### 2.4 预训练权重的加载机制

在训练过程中，MMDetection3D 框架会：

1. **加载权重文件**：从 `load_from` 指定的路径加载 `.pth` 文件
2. **匹配参数名称**：根据参数名称匹配，只加载名称匹配的层
3. **跳过不匹配的层**：如果某些层在预训练权重中不存在（如 FusionOcc 特有的 occupancy head），这些层会使用随机初始化

**代码位置**：`mmdet3d/apis/train.py:175-176`

```python
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
```

### 2.5 哪些部分可以使用 BEVDet 预训练权重？

通常可以复用的部分：

| 模块 | 是否可复用 | 说明 |
| :---: | :---: | :--- |
| Image Backbone (Swin) | ✅ | 图像特征提取，完全兼容 |
| Image Neck (FPN) | ✅ | 特征金字塔网络，完全兼容 |
| View Transformer | ✅ | 视图变换，完全兼容 |
| BEV Encoder | ⚠️ | 部分兼容，取决于架构差异 |
| Occupancy Head | ❌ | FusionOcc 特有，需要随机初始化 |
| LiDAR Encoder | ❌ | FusionOcc 特有，需要随机初始化 |

### 2.6 从零开始训练 vs 使用预训练权重

| 方式 | 优点 | 缺点 |
| :---: | :--- | :--- |
| **从零开始** | 不依赖外部资源，完全自主 | 训练时间长，可能需要更多 epoch |
| **使用预训练** | 训练快，性能通常更好 | 需要下载预训练权重，可能有架构兼容性问题 |

**建议**：
- 🎯 **推荐使用预训练权重**：特别是对于图像相关的模块
- 🔧 **如果架构差异较大**：可以只加载部分权重（如只加载 backbone）
- 📊 **实验对比**：可以同时训练两个版本，对比性能差异

---

## 3. 总结

### 3.1 关于掩码版本

- **不使用掩码**：计算所有 voxel 的损失，性能更好（mIoU: 56.62）
- **使用掩码**：只计算相机视野内的损失，训练样本更少
- **切换方法**：修改配置文件中的 `use_mask = False`

### 3.2 关于预训练权重

- **作用**：加速训练，提高性能
- **来源**：BEVDet 等 3D 检测模型的预训练权重
- **使用方法**：在配置文件中设置 `load_from` 路径
- **优势**：复用图像特征提取和视图变换的知识

### 3.3 实际训练建议

1. **首次训练**：使用 BEVDet 预训练权重 + `use_mask=False`
2. **实验对比**：尝试不同的配置组合，找到最佳设置
3. **性能优化**：根据实际需求选择是否使用掩码

---

## 4. 相关文件

- **配置文件**：`configs/fusion_occ/fusion_occ.py`
- **模型实现**：`mmdet3d/models/detectors/fusion_occ.py`
- **训练脚本**：`tools/dist_train.sh`
- **BEVDet 仓库**：https://github.com/HuangJunJie2017/BEVDet


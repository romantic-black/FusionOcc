# Draw.io 架构图使用说明

## 问题解决

如果遇到 "d.setId is not a function" 错误，可以尝试以下方法：

### 方法 1：重新生成文件（推荐）

1. 打开 [draw.io](https://app.diagrams.net/)
2. 创建新文件
3. 参考 `docs/architecture_diagram.md` 中的文本版架构图
4. 手动绘制（这样可以确保兼容性）

### 方法 2：使用文本版架构图

直接查看 `docs/architecture_diagram.md`，其中包含：
- 完整的 ASCII 流程图
- 详细的维度变化表
- 所有组件说明

### 方法 3：修复现有文件

如果文件无法打开：

1. **打开 draw.io**
2. **File → Import from → Device**
3. 选择 `occupancy_head_structure.drawio`
4. 如果仍有错误，尝试：
   - 使用旧版本的 draw.io
   - 或者将文件内容复制到一个新的 draw.io 文件中

## 文件说明

- **occupancy_head_structure.drawio**: Draw.io 格式的架构图（可能需要修复）
- **architecture_diagram.md**: 文本版架构图（推荐使用）
- **occupancy_head_architecture.md**: 完整的架构文档

## 推荐阅读顺序

1. 先看 `occupancy_head_summary.md`（快速问答）
2. 再看 `occupancy_head_architecture.md`（详细文档）
3. 最后看 `architecture_diagram.md`（架构图说明）

文本版文档包含了所有必要的信息，不需要 Draw.io 文件也能完全理解架构。


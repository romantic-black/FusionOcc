# Draw.io 文件问题解决指南

## 如果 Draw.io 文件仍然报错

### ✅ 推荐方案：使用文本版文档

**所有信息都在文本版文档中，不需要 Draw.io 文件！**

```bash
# 1. 快速问答（最推荐先看这个）
cat docs/occupancy_head_summary.md

# 2. 完整架构文档
cat docs/occupancy_head_architecture.md

# 3. 架构图说明（包含 ASCII 流程图）
cat docs/architecture_diagram.md
```

文本版文档包含：
- ✅ 完整的架构说明
- ✅ ASCII 流程图
- ✅ 详细的维度变化表
- ✅ 所有代码位置和引用
- ✅ 关键问题的答案

### 🔧 尝试修复 Draw.io 文件

如果一定要使用 Draw.io 文件，可以尝试：

1. **使用在线版本重新打开**：
   - 访问 https://app.diagrams.net/
   - 点击 "Open Existing Diagram"
   - 选择 "Device" → 上传 `occupancy_head_structure.drawio`

2. **导入为新文件**：
   - 在 draw.io 中创建新文件
   - File → Import from → Device
   - 选择 `occupancy_head_structure.drawio`

3. **手动创建**（最简单）：
   - 打开 draw.io 创建新文件
   - 参考 `docs/architecture_diagram.md` 中的 ASCII 流程图
   - 按照说明手动绘制

### 📝 最简单的解决方案

**直接看文本版文档就足够了！**

所有重要信息都在：
- `docs/occupancy_head_summary.md` - 快速问答
- `docs/occupancy_head_architecture.md` - 详细说明
- `docs/architecture_diagram.md` - 架构图和流程图

这些文档包含了比 Draw.io 文件更多的信息，包括：
- 完整的代码引用
- 详细的维度说明
- 数学公式
- 改进建议

### 💡 建议

**不需要修复 Draw.io 文件！** 文本版文档更详细、更容易搜索，并且包含了所有必要的信息。


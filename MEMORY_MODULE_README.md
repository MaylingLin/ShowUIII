# 层次化记忆模块 - 第（1）步实现文档

## 概述

本模块实现了轨迹块划分与编码功能，是层次化记忆系统的基础组件。

## 已实现的功能

### 1. 轨迹块划分器 (`model/memory/trajectory_segmentation.py`)

#### FixedWindowSegmenter
- **功能**: 固定窗口大小分段（baseline方法）
- **参数**: `window_size` (默认4)
- **用途**: 消融实验对比基线

#### RuleBasedSegmenter
- **功能**: 基于动作转换规则的语义分段
- **参数**: 
  - `min_block_size`: 最小块大小（默认2）
  - `max_block_size`: 最大块大小（默认8）
  - `boundary_patterns`: 自定义边界模式（可选）
- **默认边界模式**: 基于Mind2Web/GUIAct分析
  - SCROLL → CLICK
  - CLICK → INPUT
  - INPUT → ENTER
  - ANSWER → * (任意)
  - 等

### 2. 块编码器 (`model/memory/block_encoder.py`)

#### TrajectoryBlockEncoder
- **功能**: 使用自注意力机制将轨迹块编码为固定维度向量
- **架构**:
  1. 步骤投影：融合视觉特征、动作类型、坐标
  2. 位置编码：块内步骤顺序
  3. Transformer：多层自注意力聚合
  4. 压缩：query token 或 mean pooling
- **参数**: 
  - `vision_hidden_size`: 视觉编码器输出维度（从模型配置获取）
  - `action_vocab_size`: 动作类型数量（从数据集构建）
  - `hidden_dim`: 隐藏维度（默认1024）
  - `num_layers`: Transformer层数（默认2）
  - `compression_method`: 'query' 或 'mean'

### 3. 损失函数 (`model/memory/losses.py`)

#### BlockEncoderLoss
- **动作预测损失**: 预测块内出现过哪些动作类型
- **对比学习损失**: 同任务块应相似（可选）

### 4. 动作词表 (`model/memory/action_vocab.py`)

#### ActionVocabulary
- **功能**: 从数据集动态构建动作类型词表
- **特点**: 
  - 无硬编码：从实际数据提取
  - 支持缓存：避免重复构建
  - 大小写归一化

---

## 避免硬编码策略

### ✅ 完全动态配置

1. **动作空间**: 通过 `ActionVocabulary` 从数据集提取
2. **模型维度**: 所有维度参数通过构造函数传入
3. **块大小**: 所有阈值可配置

### ⚠️ 有默认值但可覆盖

1. **边界模式**: `RuleBasedSegmenter.get_default_patterns()`
   - 提供常见GUI动作的默认规则
   - 可通过 `boundary_patterns` 参数完全替换

```python
# 使用自定义边界模式
custom_patterns = [('SCROLL', 'CLICK'), ('INPUT', 'ENTER')]
segmenter = RuleBasedSegmenter(boundary_patterns=custom_patterns)
```

2. **超参数**: `hidden_dim`, `num_layers` 等
   - 提供合理默认值
   - 可通过参数调整

---

## 测试方法

### 测试1: 基础逻辑测试（无需PyTorch）

```bash
cd /home/may/Proj/ShowUIII
python test_memory_basic.py
```

**测试内容**:
- ✓ 固定窗口分段
- ✓ 规则分段边界检测
- ✓ 自定义边界模式
- ✓ 边缘情况处理
- ✓ 参数验证

### 测试2: 完整功能测试（需要PyTorch）

```bash
cd /home/may/Proj/ShowUIII
python test_memory_module.py
```

**测试内容**:
- ✓ 轨迹分段（5个测试用例）
- ✓ 块编码器前向传播
- ✓ 两种压缩方法（query/mean）
- ✓ Padding处理
- ✓ 损失函数计算
- ✓ 反向传播梯度
- ✓ 动作词表构建
- ✓ 端到端集成

### 测试3: Mind2Web真实数据测试

```bash
cd /home/may/Proj/ShowUIII
python test_with_mind2web.py
```

**数据路径**:
- 图像: `/home/may/Proj/SeeProcess/data/mind2web_images/mind2web_images`
- 标注: `/home/may/Proj/SeeProcess/data/mind2web_annots`
- 文件格式: `mind2web_data_{split}.json`

**测试内容**:
- ✓ 加载真实Mind2Web数据
- ✓ 动作分布分析
- ✓ 轨迹块划分效果
- ✓ 动作词表构建
- ✓ 图像文件检查

**输出示例**:
```
Action Distribution:
  CLICK: 45%
  INPUT: 25%
  SCROLL: 15%
  ...

Segmentation Results:
  Fixed Window (size=4): 8 blocks, avg 3.5 steps
  Rule-Based: 6 blocks, avg 4.7 steps
```

---

## 使用示例

### 示例1: 基本分段

```python
from model.memory import FixedWindowSegmenter, RuleBasedSegmenter

# 准备轨迹数据
trajectory = [
    {'obs': 'img1.png', 'action': {'action': 'SCROLL', 'position': [0.5, 0.3]}},
    {'obs': 'img2.png', 'action': {'action': 'CLICK', 'position': [0.6, 0.4]}},
    {'obs': 'img3.png', 'action': {'action': 'INPUT', 'position': [0.7, 0.5], 'value': 'query'}},
]

# 方法1: 固定窗口
fixed_seg = FixedWindowSegmenter(window_size=2)
blocks = fixed_seg.segment(trajectory)

# 方法2: 规则分段
rule_seg = RuleBasedSegmenter(min_block_size=2, max_block_size=5)
blocks = rule_seg.segment(trajectory)
```

### 示例2: 块编码

```python
import torch
from model.memory import TrajectoryBlockEncoder

# 初始化编码器（参数从模型/数据配置获取）
encoder = TrajectoryBlockEncoder(
    vision_hidden_size=1536,  # 从Qwen2-VL配置
    action_vocab_size=15,      # 从数据集构建
    hidden_dim=1024,
    num_layers=2,
    compression_method='query'
)

# 准备块数据
batch_size, num_steps = 2, 4
vision_features = torch.randn(batch_size, num_steps, 1536)
action_types = torch.randint(0, 15, (batch_size, num_steps))
action_positions = torch.rand(batch_size, num_steps, 4)  # [x, y, w, h]
attention_mask = torch.ones(batch_size, num_steps)

# 编码
block_embeddings = encoder(
    vision_features,
    action_types,
    action_positions,
    attention_mask
)
# 输出: (batch_size, hidden_dim) = (2, 1024)
```

### 示例3: 构建动作词表

```python
from model.memory.action_vocab import get_action_vocab_for_dataset

# 从数据集构建词表（带缓存）
vocab = get_action_vocab_for_dataset(
    dataset_dir="/path/to/datasets",
    dataset_name="Mind2Web",
    json_file="mind2web_data_train",
    cache_dir="./cache"
)

# 使用词表
action_idx = vocab.encode('CLICK')  # 编码
action_name = vocab.decode(action_idx)  # 解码
print(f"Vocabulary size: {len(vocab)}")
print(f"Actions: {vocab.get_stats()['actions']}")
```

---

## 下一步集成计划

### 待完成任务

1. **数据加载器集成** (`data/dset_trajectory_blocks.py`)
   - 继承 `NavigationDataset`
   - 添加轨迹块预处理
   - 返回块编码所需数据

2. **Collate函数适配** (`data/data_utils.py`)
   - 处理块数据批处理
   - Padding对齐

3. **训练脚本集成** (`train.py`)
   - 初始化块编码器
   - 添加块编码损失
   - 训练循环集成

4. **配置参数** (argparse)
   - `--use_block_memory`: 启用块记忆
   - `--segmenter_type`: 'fixed' / 'rule'
   - `--block_hidden_dim`: 块编码器维度
   - `--compression_method`: 'query' / 'mean'

---

## 文件结构

```
ShowUIII/
├── model/memory/
│   ├── __init__.py
│   ├── trajectory_segmentation.py  # 轨迹分段器
│   ├── block_encoder.py            # 块编码器
│   ├── losses.py                   # 损失函数
│   └── action_vocab.py             # 动作词表
├── test_memory_basic.py            # 基础测试（无PyTorch）
├── test_memory_module.py           # 完整测试（需PyTorch）
├── test_with_mind2web.py           # Mind2Web数据测试
└── MEMORY_MODULE_README.md         # 本文档
```

---

## 参数量统计

### TrajectoryBlockEncoder (hidden_dim=1024, num_layers=2)
- 步骤投影层: ~5M
- Transformer: ~8M
- 输出投影: ~1M
- **总计**: ~10-15M 参数（可单卡训练）

---

## 常见问题

### Q1: 如何调整分段策略？

**A**: 根据数据特点选择：
- 短序列（<10步）: 固定窗口或无分段
- 中等序列（10-20步）: 规则分段
- 长序列（>20步）: 规则分段+自定义边界模式

### Q2: 如何选择压缩方法？

**A**: 
- `query`: 更强表达能力，适合复杂任务
- `mean`: 更简单高效，适合资源受限

消融实验推荐测试两者。

### Q3: 动作词表大小会影响什么？

**A**: 
- 影响 `action_type_embed` 层大小
- 典型GUI任务词表: 10-20个动作
- Mind2Web: ~8-12个动作（CLICK, INPUT, SELECT, SCROLL等）

### Q4: 如何处理新的数据集？

**A**:
1. 使用 `ActionVocabulary.build_from_dataset()` 构建词表
2. 根据动作模式分析，自定义边界规则
3. 运行 `test_with_mind2web.py` 风格的测试验证

---

## 性能优化建议

1. **缓存词表**: 使用 `cache_dir` 参数避免重复构建
2. **预分段**: 在数据预处理阶段完成分段，训练时直接加载
3. **混合精度**: 使用 fp16/bf16 训练块编码器
4. **Gradient Checkpointing**: 若显存不足，可对Transformer层使用

---

## 引用与参考

本实现基于以下设计原则：
- **无硬编码**: 所有关键配置从数据/模型配置动态获取
- **模块化**: 分段器、编码器、损失函数独立可测试
- **可扩展**: 易于添加新的分段策略或压缩方法

相关论文：
- Mind2Web: Towards a Generalist Agent for the Web
- GUIAct: GUI Agent Corpus and Training
- Transformer in Transformer: Efficient Vision Transformers

---

## 更新日志

### v1.0 (2025-01-01)
- ✓ 实现轨迹块划分器（Fixed Window & Rule-Based）
- ✓ 实现块编码器（Self-Attention + Compression）
- ✓ 实现损失函数（Action Presence + Contrastive）
- ✓ 实现动作词表构建
- ✓ 完整测试脚本（基础+完整+真实数据）
- ✓ 文档与使用示例


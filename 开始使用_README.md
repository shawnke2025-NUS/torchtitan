# 🎉 完成！Qwen3.5 可配置训练方案已就绪

## ✅ 验证结果

所有 14 项检查通过！工具链已完全就绪，可以开始训练。

---

## 📦 你得到了什么

### 核心训练工具
1. **[run_qwen35_configurable.sh](run_qwen35_configurable.sh)** ⭐⭐⭐
   - 功能最强大的可配置训练脚本
   - 支持调整 GPU 数量、所有并行策略
   - 自动输出精度、显存、性能信息

2. **[examples_qwen35_training.sh](examples_qwen35_training.sh)** ⭐⭐
   - 12 种预设配置示例
   - 新手友好的菜单选择

3. **[monitor_training.sh](monitor_training.sh)** ⭐⭐
   - 实时监控训练状态
   - 7 种监控模式（Loss、显存、性能等）

### 完整文档
4. **[README_可配置训练方案.md](README_可配置训练方案.md)** - 总览文档
5. **[可配置训练脚本使用指南.md](可配置训练脚本使用指南.md)** - 详细使用说明
6. **[快速开始.md](快速开始.md)** - 新手入门指南
7. **[QWEN35_运行指南.md](QWEN35_运行指南.md)** - 完整配置参考
8. **[QWEN35_环境准备指南.md](QWEN35_环境准备指南.md)** - 环境配置详解

### 测试工具
9. **[test_toolchain.sh](test_toolchain.sh)** - 工具链验证脚本
10. **[check_qwen35_ready.sh](check_qwen35_ready.sh)** - 环境检查脚本
11. **[setup_and_run_qwen35.sh](setup_and_run_qwen35.sh)** - 一键安装运行

---

## 🚀 立即开始（三选一）

### 方式 1: 使用示例选择器（推荐新手）
```bash
bash examples_qwen35_training.sh
```
会显示 12 种预设配置，选择一个即可开始。

### 方式 2: 直接运行可配置脚本
```bash
# 单 GPU 测试（1 步，快速验证）
STEPS=1 bash run_qwen35_configurable.sh

# 单 GPU 标准训练（10 步）
bash run_qwen35_configurable.sh

# 4 GPU 数据并行
NGPU=4 bash run_qwen35_configurable.sh

# 8 GPU 混合并行
NGPU=8 DP_SHARD=4 TP=2 bash run_qwen35_configurable.sh
```

### 方式 3: 自定义所有参数
```bash
NGPU=8 \
STEPS=1000 \
BATCH_SIZE=2 \
SEQ_LEN=2048 \
DP_SHARD=4 \
TP=2 \
COMPILE=true \
CHECKPOINT_INTERVAL=100 \
bash run_qwen35_configurable.sh
```

---

## 📊 如何实时查看精度、显存、性能？

### ✅ 训练时自动显示（最直接）

运行训练后，终端会**实时显示**：
```
[2026-07-16 10:00:01] Step 1/10 | Loss: 10.5234 | Tokens/sec: 1234.5 | GPU Mem: 45.2 GB
[2026-07-16 10:00:02] Step 2/10 | Loss: 10.2156 | Tokens/sec: 1245.8 | GPU Mem: 45.3 GB
```

**包含的信息**：
- 🎯 **Loss**：训练精度（越低越好）
- ⚡ **Tokens/sec**：训练速度（越高越好）
- 💾 **GPU Mem**：显存使用
- 📈 **Grad Norm**：梯度稳定性

### ✅ 使用监控脚本（最方便）

**在另一个终端运行**：
```bash
bash monitor_training.sh
```

然后选择：
- **选项 1**: 查看所有日志（实时）
- **选项 2**: 只看 Loss（精度变化）
- **选项 3**: 只看性能（Tokens/sec）
- **选项 4**: 只看显存（GPU Memory）
- **选项 5**: 只看梯度（Grad Norm）
- **选项 7**: 启动 TensorBoard

### ✅ TensorBoard 可视化（最直观）

```bash
# 方式 1: 使用监控脚本
bash monitor_training.sh  # 选择 7

# 方式 2: 直接启动
tensorboard --logdir ./outputs/
```

浏览器打开 `http://localhost:6006`，可以看到：
- 📈 Loss 曲线
- 💾 显存曲线
- ⚡ 性能曲线（Tokens/sec、TFLOPS、MFU）
- 📊 学习率变化
- 🔢 梯度统计

### 输出位置总结

| 信息类型 | 位置 1（终端） | 位置 2（监控脚本） | 位置 3（TensorBoard） |
|---------|---------------|-------------------|---------------------|
| **Loss** | 自动显示 | 选项 2 | Loss 标签页 |
| **显存** | 自动显示 | 选项 4 | Memory 标签页 |
| **性能** | 自动显示 | 选项 3 | Performance 标签页 |
| **梯度** | 自动显示 | 选项 5 | Gradients 标签页 |

---

## ⚙️ GPU 数量和并行策略配置

### 快速参考表

| GPU 数量 | 推荐命令 | 并行策略 |
|---------|---------|---------|
| 1 个 | `bash run_qwen35_configurable.sh` | 单 GPU |
| 2-4 个 | `NGPU=4 bash run_qwen35_configurable.sh` | 数据并行 |
| 8 个 | `NGPU=8 DP_SHARD=4 TP=2 bash run_qwen35_configurable.sh` | DP + TP |
| 8 个 | `NGPU=8 DP_SHARD=2 TP=2 PP=2 bash run_qwen35_configurable.sh` | DP + TP + PP |
| 16 个 | `NGPU=16 DP_SHARD=4 TP=2 PP=2 bash run_qwen35_configurable.sh` | 三维并行 |
| 32 个 | `NGPU=32 DP_SHARD=8 TP=2 PP=2 bash run_qwen35_configurable.sh` | 大规模并行 |

### 并行参数说明

| 参数 | 含义 | 默认值 | 示例 |
|------|------|--------|------|
| `NGPU` | GPU 总数 | 1 | `NGPU=8` |
| `DP_SHARD` | FSDP 分片度 | -1（自动） | `DP_SHARD=4` |
| `DP_REPLICATE` | DDP 复制度 | 1 | `DP_REPLICATE=2` |
| `TP` | 张量并行度 | 1 | `TP=2` |
| `PP` | 流水线并行度 | 1 | `PP=2` |
| `EP` | 专家并行度（MoE） | 1 | `EP=4` |
| `CP` | 上下文并行度 | 1 | `CP=2` |

**并行度计算规则**：`NGPU = DP_SHARD × TP × PP × EP × CP`

---

## 📖 推荐阅读顺序

### 第 1 步：快速上手
- 阅读：**[README_可配置训练方案.md](README_可配置训练方案.md)**（本文档）
- 运行：`STEPS=1 bash run_qwen35_configurable.sh`
- 监控：`bash monitor_training.sh`

### 第 2 步：尝试不同配置
- 运行：`bash examples_qwen35_training.sh`
- 尝试不同的 GPU 数量和并行策略

### 第 3 步：深入学习
- 阅读：**[可配置训练脚本使用指南.md](可配置训练脚本使用指南.md)**
- 了解所有可配置参数和使用场景

### 第 4 步：故障排查（如果遇到问题）
- 运行：`bash check_qwen35_ready.sh`（环境检查）
- 阅读：**[QWEN35_环境准备指南.md](QWEN35_环境准备指南.md)**

---

## 💡 典型使用流程

### 场景 1: 首次使用
```bash
# 1. 验证工具链
bash test_toolchain.sh

# 2. 快速测试（1 步）
STEPS=1 bash run_qwen35_configurable.sh

# 3. 查看监控（在另一个终端）
bash monitor_training.sh

# 4. 标准训练（10 步）
bash run_qwen35_configurable.sh
```

### 场景 2: 多 GPU 训练
```bash
# 1. 选择预设配置
bash examples_qwen35_training.sh
# 选择 4 或 5（4 或 8 GPU）

# 2. 或直接运行
NGPU=8 DP_SHARD=4 TP=2 bash run_qwen35_configurable.sh

# 3. 监控（在另一个终端）
bash monitor_training.sh
```

### 场景 3: 长时间训练
```bash
# 1. 启动训练
NGPU=8 STEPS=10000 CHECKPOINT_INTERVAL=100 \
bash run_qwen35_configurable.sh

# 2. 监控 Loss
bash monitor_training.sh  # 选择 2

# 3. 查看 TensorBoard
tensorboard --logdir ./outputs/
```

---

## 🔧 所有可配置参数速查

### 基本参数
```bash
NGPU=8                    # GPU 数量
STEPS=1000                # 训练步数
BATCH_SIZE=2              # Batch size
SEQ_LEN=2048              # 序列长度
COMPILE=true              # torch.compile
CONFIG=qwen35_2b          # 模型配置
```

### 并行策略
```bash
DP_SHARD=4                # FSDP 分片
TP=2                      # 张量并行
PP=2                      # 流水线并行
EP=4                      # 专家并行
CP=2                      # 上下文并行
```

### 监控配置
```bash
METRICS_LOG_FREQ=1        # 输出频率
TENSORBOARD=true          # TensorBoard
ENABLE_PROFILING=true     # 性能分析
MEMORY_SNAPSHOT=true      # 显存快照
```

### 检查点
```bash
CHECKPOINT_ENABLE=true    # 启用检查点
CHECKPOINT_INTERVAL=100   # 保存间隔
CHECKPOINT_FOLDER=./ckpt  # 保存路径
```

**完整列表**：见 [可配置训练脚本使用指南.md](可配置训练脚本使用指南.md)

---

## 🎯 回答你的核心问题

### ✅ 问题 1: 如何调整 GPU 数量？
**答案**：使用 `NGPU` 参数
```bash
# 单 GPU
bash run_qwen35_configurable.sh

# 4 GPU
NGPU=4 bash run_qwen35_configurable.sh

# 8 GPU
NGPU=8 bash run_qwen35_configurable.sh
```

### ✅ 问题 2: 如何调整并行特性？
**答案**：使用并行策略参数
```bash
# 只用数据并行
NGPU=8 bash run_qwen35_configurable.sh

# 数据并行 + 张量并行
NGPU=8 DP_SHARD=4 TP=2 bash run_qwen35_configurable.sh

# 三维并行
NGPU=8 DP_SHARD=2 TP=2 PP=2 bash run_qwen35_configurable.sh
```

### ✅ 问题 3: 哪里可以实时输出精度、显存、性能信息？
**答案**：有 3 个地方

1. **终端自动输出**（训练时实时显示）
   ```
   Step 1/10 | Loss: 10.52 | Tokens/sec: 1234 | GPU Mem: 45.2 GB
   ```

2. **监控脚本**（专门查看特定指标）
   ```bash
   bash monitor_training.sh
   # 选择 2 查看 Loss
   # 选择 3 查看性能
   # 选择 4 查看显存
   ```

3. **TensorBoard**（可视化曲线）
   ```bash
   tensorboard --logdir ./outputs/
   # 浏览器打开 http://localhost:6006
   ```

---

## 🎉 总结

你现在拥有：
- ✅ **3 个强大的训练脚本**（可配置、示例选择、监控）
- ✅ **8 份完整文档**（从新手到进阶）
- ✅ **3 个测试工具**（验证、检查、一键安装）
- ✅ **完全控制**：GPU 数量、并行策略、监控方式
- ✅ **实时可观测**：精度、显存、性能随时查看

---

## 📞 下一步

**立即开始**：
```bash
# 快速验证
STEPS=1 bash run_qwen35_configurable.sh

# 查看监控
bash monitor_training.sh
```

**遇到问题**：
1. 运行 `bash check_qwen35_ready.sh` 检查环境
2. 查看 [QWEN35_环境准备指南.md](QWEN35_环境准备指南.md)
3. 使用 `COMM_MODE="fake_backend"` 验证配置

**需要更多帮助**：
- 详细文档：[可配置训练脚本使用指南.md](可配置训练脚本使用指南.md)
- 快速入门：[快速开始.md](快速开始.md)

---

**🚀 祝你训练顺利！有任何问题随时查看文档或重新运行测试脚本！**

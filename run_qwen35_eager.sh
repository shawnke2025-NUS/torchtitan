#!/bin/bash
# Qwen3.5 动态图模式（Eager Mode）运行脚本
# 动态图模式 = 不使用 torch.compile，保持 PyTorch 原生执行方式

set -e

echo "=========================================="
echo "运行 Qwen3.5 动态图模式训练"
echo "=========================================="

# 配置参数
NUM_GPUS=${NUM_GPUS:-1}  # 默认使用 1 个 GPU，可通过环境变量修改
CONFIG_NAME=${CONFIG_NAME:-qwen35_debugmodel}  # 默认使用 debugmodel 配置

echo "GPU 数量: $NUM_GPUS"
echo "配置名称: $CONFIG_NAME"
echo ""

# 检查 tokenizer 是否存在
if [ ! -d "./tests/assets/tokenizer" ]; then
    echo "警告: tokenizer 目录不存在，请确保已准备好 tokenizer 文件"
    echo "目录: ./tests/assets/tokenizer"
fi

# 检查测试数据集是否存在
if [ ! -d "./tests/assets/cc12m_test" ]; then
    echo "警告: 测试数据集目录不存在"
    echo "目录: ./tests/assets/cc12m_test"
fi

echo "开始训练..."
echo ""

# 根据 GPU 数量选择运行方式
if [ "$NUM_GPUS" -eq 1 ]; then
    # 单 GPU 运行
    python -m torchtitan.train \
        --module torchtitan.models.qwen3_5.config_registry \
        --config "$CONFIG_NAME" \
        --training.compile.enable False \
        --debug.seed 42 \
        --debug.deterministic True
else
    # 多 GPU 运行
    torchrun --nproc_per_node="$NUM_GPUS" -m torchtitan.train \
        --module torchtitan.models.qwen3_5.config_registry \
        --config "$CONFIG_NAME" \
        --training.compile.enable False \
        --debug.seed 42 \
        --debug.deterministic True
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="

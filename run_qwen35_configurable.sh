#!/usr/bin/bash
# Qwen3.5 可配置训练脚本
# 支持灵活调整 GPU 数量和并行策略
#
# 使用示例:
#   # 单 GPU，动态图模式
#   bash run_qwen35_configurable.sh
#
#   # 4 GPU，数据并行
#   NGPU=4 bash run_qwen35_configurable.sh
#
#   # 8 GPU，混合并行（DP=2, TP=2, PP=2）
#   NGPU=8 DP_SHARD=2 TP=2 PP=2 bash run_qwen35_configurable.sh
#
#   # 使用编译模式
#   COMPILE=true bash run_qwen35_configurable.sh
#
#   # 调试模式（单 GPU 模拟多 GPU）
#   NGPU=4 COMM_MODE="local_tensor" bash run_qwen35_configurable.sh

set -ex

# ============================================
# 基本配置
# ============================================
NGPU=${NGPU:-"1"}                           # GPU 数量，默认 1
export LOG_RANK=${LOG_RANK:-0}             # 日志输出的 rank，默认 0
MODULE=${MODULE:-"torchtitan.models.qwen3_5.config_registry"}  # 模型模块
CONFIG=${CONFIG:-"qwen35_debugmodel"}      # 配置名称
COMM_MODE=${COMM_MODE:-""}                 # 通信模式：空=正常，fake_backend=配置验证，local_tensor=单GPU调试

# ============================================
# 训练参数配置
# ============================================
STEPS=${STEPS:-"10"}                       # 训练步数
BATCH_SIZE=${BATCH_SIZE:-"1"}              # 本地 batch size
SEQ_LEN=${SEQ_LEN:-"512"}                  # 序列长度
COMPILE=${COMPILE:-"false"}                # 是否启用 torch.compile（true/false）
SEED=${SEED:-"42"}                         # 随机种子
DETERMINISTIC=${DETERMINISTIC:-"true"}     # 是否确定性训练

# ============================================
# 并行策略配置
# ============================================
# 数据并行（Data Parallel）
DP_REPLICATE=${DP_REPLICATE:-"1"}         # 数据并行复制度（DDP/HSDP 的 replicate 维度）
DP_SHARD=${DP_SHARD:-"-1"}                # 数据并行分片度（FSDP 分片维度，-1 表示使用剩余 GPU）

# 张量并行（Tensor Parallel）
TP=${TP:-"1"}                             # 张量并行度（1 表示禁用）
ENABLE_ASYNC_TP=${ENABLE_ASYNC_TP:-"false"}  # 是否启用异步 TP

# 流水线并行（Pipeline Parallel）
PP=${PP:-"1"}                             # 流水线并行度（1 表示禁用）
PP_SCHEDULE=${PP_SCHEDULE:-"gpipe"}       # PP 调度策略：gpipe, 1f1b, looped_bps

# 专家并行（Expert Parallel，仅 MoE 模型）
EP=${EP:-"1"}                             # 专家并行度（1 表示禁用）

# 上下文并行（Context Parallel）
CP=${CP:-"1"}                             # 上下文并行度（1 表示禁用）

# ============================================
# 监控和日志配置
# ============================================
METRICS_LOG_FREQ=${METRICS_LOG_FREQ:-"1"} # 指标输出频率（每 N 步）
ENABLE_PROFILING=${ENABLE_PROFILING:-"false"}  # 是否启用性能分析
PROFILE_FREQ=${PROFILE_FREQ:-"5"}         # 性能分析频率
TENSORBOARD=${TENSORBOARD:-"true"}        # 是否启用 TensorBoard
MEMORY_SNAPSHOT=${MEMORY_SNAPSHOT:-"false"}  # 是否启用显存快照

# ============================================
# 检查点配置
# ============================================
CHECKPOINT_ENABLE=${CHECKPOINT_ENABLE:-"true"}   # 是否启用检查点
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-"100"} # 检查点保存间隔
CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-"./checkpoints/qwen35_run_$(date +%Y%m%d_%H%M%S)"}

# ============================================
# 输出目录配置
# ============================================
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/qwen35_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CHECKPOINT_FOLDER}"

# ============================================
# 打印配置信息
# ============================================
echo "=========================================="
echo "Qwen3.5 训练配置"
echo "=========================================="
echo "基本配置:"
echo "  GPU 数量: ${NGPU}"
echo "  模型配置: ${CONFIG}"
echo "  训练步数: ${STEPS}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  序列长度: ${SEQ_LEN}"
echo "  编译模式: ${COMPILE}"
echo ""
echo "并行策略:"
echo "  数据并行 (DP Replicate): ${DP_REPLICATE}"
echo "  数据并行 (DP Shard): ${DP_SHARD}"
echo "  张量并行 (TP): ${TP}"
echo "  流水线并行 (PP): ${PP}"
echo "  专家并行 (EP): ${EP}"
echo "  上下文并行 (CP): ${CP}"
echo ""
echo "监控配置:"
echo "  指标输出频率: ${METRICS_LOG_FREQ}"
echo "  性能分析: ${ENABLE_PROFILING}"
echo "  TensorBoard: ${TENSORBOARD}"
echo ""
echo "输出目录:"
echo "  日志: ${OUTPUT_DIR}"
echo "  检查点: ${CHECKPOINT_FOLDER}"
echo "=========================================="
echo ""

# ============================================
# 构建训练参数
# ============================================
TRAIN_ARGS=(
    # 基本训练参数
    --training.steps "${STEPS}"
    --training.local_batch_size "${BATCH_SIZE}"
    --training.seq_len "${SEQ_LEN}"
    --training.compile.enable "${COMPILE}"

    # 并行配置
    --parallelism.data_parallel_replicate_degree "${DP_REPLICATE}"
    --parallelism.data_parallel_shard_degree "${DP_SHARD}"
    --parallelism.tensor_parallel_degree "${TP}"
    --parallelism.pipeline_parallel_degree "${PP}"
    --parallelism.enable_async_tensor_parallel "${ENABLE_ASYNC_TP}"

    # 调试配置
    --debug.seed "${SEED}"
    --debug.deterministic "${DETERMINISTIC}"

    # 指标配置
    --metrics.log_freq "${METRICS_LOG_FREQ}"
    --metrics.enable_tensorboard "${TENSORBOARD}"

    # 检查点配置
    --checkpoint.enable "${CHECKPOINT_ENABLE}"
    --checkpoint.interval "${CHECKPOINT_INTERVAL}"
    --checkpoint.folder "${CHECKPOINT_FOLDER}"

    # 输出目录
    --dump_folder "${OUTPUT_DIR}"
)

# MoE 模型专家并行配置
if [ "${EP}" != "1" ]; then
    TRAIN_ARGS+=(--parallelism.expert_parallel_degree "${EP}")
fi

# 上下文并行配置
if [ "${CP}" != "1" ]; then
    TRAIN_ARGS+=(--parallelism.context_parallel_degree "${CP}")
fi

# 流水线并行调度策略
if [ "${PP}" != "1" ]; then
    TRAIN_ARGS+=(--parallelism.pipeline_parallel_schedule "${PP_SCHEDULE}")
fi

# 性能分析配置
if [ "${ENABLE_PROFILING}" = "true" ]; then
    TRAIN_ARGS+=(
        --profiling.enable_profiling true
        --profiling.profile_freq "${PROFILE_FREQ}"
    )
fi

# 显存快照配置
if [ "${MEMORY_SNAPSHOT}" = "true" ]; then
    TRAIN_ARGS+=(
        --profiling.enable_memory_snapshot true
        --profiling.save_memory_snapshot_folder "${OUTPUT_DIR}/memory_snapshots"
    )
fi

# ============================================
# 执行训练
# ============================================
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

if [ -n "$COMM_MODE" ]; then
    # 调试模式：验证配置或单 GPU 模拟
    echo "运行调试模式: comm_mode=${COMM_MODE}"
    echo "这将在单 GPU 上模拟 ${NGPU} GPU 的训练"
    echo ""

    NGPU="${NGPU}" LOCAL_RANK=0 python -m torchtitan.train \
        --module ${MODULE} \
        --config ${CONFIG} \
        --comm.mode="${COMM_MODE}" \
        "${TRAIN_ARGS[@]}"
else
    # 正常训练模式
    echo "开始正常训练..."
    echo "命令: torchrun --nproc_per_node=${NGPU} ..."
    echo ""

    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun \
        --nproc_per_node="${NGPU}" \
        --rdzv_backend c10d \
        --rdzv_endpoint="localhost:0" \
        --local-ranks-filter ${LOG_RANK} \
        --role rank \
        --tee 3 \
        -m torchtitan.train \
        --module ${MODULE} \
        --config ${CONFIG} \
        "${TRAIN_ARGS[@]}"
fi

# ============================================
# 训练完成后的信息
# ============================================
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 训练成功完成！"
    echo "=========================================="
    echo ""
    echo "查看结果:"
    echo "  日志文件: ${OUTPUT_DIR}/"
    echo "  检查点: ${CHECKPOINT_FOLDER}/"
    if [ "${TENSORBOARD}" = "true" ]; then
        echo "  TensorBoard: tensorboard --logdir ${OUTPUT_DIR}"
    fi
    echo ""
    echo "实时监控命令:"
    echo "  tail -f ${OUTPUT_DIR}/train.log"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 训练失败"
    echo "=========================================="
    echo "检查日志: ${OUTPUT_DIR}/"
fi

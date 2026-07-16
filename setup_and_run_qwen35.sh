#!/bin/bash
# Qwen3.5 一键安装和运行脚本
# 自动检测环境并提供修复建议

set -e

echo "=========================================="
echo "Qwen3.5 一键运行助手"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检测 Python 命令
detect_python() {
    if command -v python &> /dev/null; then
        PYTHON_CMD="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v py &> /dev/null; then
        PYTHON_CMD="py"
    else
        echo -e "${RED}错误: 未找到 Python${NC}"
        echo "请先安装 Python 3.10 或更高版本"
        echo "推荐使用 Conda: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    echo -e "${GREEN}✓${NC} 检测到 Python: $PYTHON_CMD"
}

# 检查 PyTorch
check_pytorch() {
    echo ""
    echo "检查 PyTorch..."
    if $PYTHON_CMD -c "import torch" 2>/dev/null; then
        VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
        echo -e "${GREEN}✓${NC} PyTorch 已安装: $VERSION"

        # 检查 CUDA
        if $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
            CUDA_VER=$($PYTHON_CMD -c "import torch; print(torch.version.cuda)")
            GPU_COUNT=$($PYTHON_CMD -c "import torch; print(torch.cuda.device_count())")
            echo -e "${GREEN}✓${NC} CUDA 可用: $CUDA_VER"
            echo -e "${GREEN}✓${NC} GPU 数量: $GPU_COUNT"
            HAS_GPU=true
        else
            echo -e "${YELLOW}!${NC} CUDA 不可用，将使用 CPU 训练（速度较慢）"
            HAS_GPU=false
        fi
        return 0
    else
        echo -e "${RED}✗${NC} PyTorch 未安装"
        echo ""
        echo "请安装 PyTorch："
        echo ""
        echo "方法 1 - 使用 Conda（推荐）："
        echo "  conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia"
        echo ""
        echo "方法 2 - 使用 pip（GPU 版本）："
        echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        echo ""
        echo "方法 3 - 使用 pip（CPU 版本）："
        echo "  pip install torch torchvision torchaudio"
        echo ""
        return 1
    fi
}

# 安装依赖
install_dependencies() {
    echo ""
    echo "检查 TorchTitan 依赖..."

    # 检查 torchtitan 是否可导入
    if $PYTHON_CMD -c "import torchtitan" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} TorchTitan 模块可用"
        return 0
    fi

    echo -e "${YELLOW}!${NC} TorchTitan 依赖未完全安装"
    echo ""
    read -p "是否现在安装依赖？(y/n) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "安装依赖中..."
        if [ -f ".ci/docker/requirements.txt" ]; then
            $PYTHON_CMD -m pip install -r .ci/docker/requirements.txt
            echo -e "${GREEN}✓${NC} 依赖安装完成"
        else
            echo -e "${RED}✗${NC} 找不到 requirements.txt"
            echo "请手动运行: pip install -r .ci/docker/requirements.txt"
            return 1
        fi
    else
        echo "跳过依赖安装"
        return 1
    fi
}

# 检查必需文件
check_files() {
    echo ""
    echo "检查必需文件..."

    all_files_ok=true

    # 检查 tokenizer
    if [ -f "tests/assets/tokenizer/tokenizer.json" ]; then
        echo -e "${GREEN}✓${NC} Tokenizer: tests/assets/tokenizer/tokenizer.json"
    else
        echo -e "${RED}✗${NC} Tokenizer 缺失"
        all_files_ok=false
    fi

    # 检查数据集
    if [ -f "tests/assets/cc12m_test/cc12m-train-0000.tar" ]; then
        echo -e "${GREEN}✓${NC} 数据集: tests/assets/cc12m_test/cc12m-train-0000.tar"
    else
        echo -e "${RED}✗${NC} 数据集缺失"
        all_files_ok=false
    fi

    if [ "$all_files_ok" = false ]; then
        echo ""
        echo -e "${RED}错误: 缺少必需文件${NC}"
        echo "请确保已完整克隆 git 仓库"
        return 1
    fi

    return 0
}

# 设置 PYTHONPATH
setup_pythonpath() {
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    echo ""
    echo -e "${GREEN}✓${NC} PYTHONPATH 已设置"
}

# 运行训练
run_training() {
    echo ""
    echo "=========================================="
    echo "开始训练 Qwen3.5"
    echo "=========================================="
    echo ""

    # 询问运行模式
    echo "选择运行模式："
    echo "  1) 快速测试（1 步训练，验证环境）"
    echo "  2) 标准训练（10 步训练，debugmodel 配置）"
    echo "  3) 自定义训练"
    echo ""
    read -p "请选择 (1/2/3): " -n 1 -r mode
    echo ""
    echo ""

    # 基础参数
    BASE_CMD="$PYTHON_CMD -m torchtitan.train --module torchtitan.models.qwen3_5.config_registry --config qwen35_debugmodel --training.compile.enable False"

    case $mode in
        1)
            echo "运行快速测试（1 步）..."
            $BASE_CMD --training.steps 1 --debug.seed 42
            ;;
        2)
            echo "运行标准训练（10 步）..."
            $BASE_CMD --debug.seed 42 --debug.deterministic True
            ;;
        3)
            echo "请手动运行命令，例如："
            echo "$BASE_CMD --training.steps 20"
            exit 0
            ;;
        *)
            echo "无效选择，运行默认测试（1 步）..."
            $BASE_CMD --training.steps 1 --debug.seed 42
            ;;
    esac

    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo -e "${GREEN}✓ 训练成功完成！${NC}"
        echo "=========================================="
        echo ""
        echo "下一步："
        echo "  - 查看日志: ls -la outputs/"
        echo "  - 运行更多步数: $BASE_CMD --training.steps 100"
        echo "  - 查看完整文档: cat QWEN35_运行指南.md"
    else
        echo ""
        echo "=========================================="
        echo -e "${RED}✗ 训练失败${NC}"
        echo "=========================================="
        echo ""
        echo "故障排查："
        echo "  1. 检查错误信息"
        echo "  2. 运行环境检查: bash check_qwen35_ready.sh"
        echo "  3. 查看准备指南: cat QWEN35_环境准备指南.md"
    fi
}

# 主流程
main() {
    # 1. 检测 Python
    detect_python

    # 2. 检查 PyTorch
    if ! check_pytorch; then
        exit 1
    fi

    # 3. 安装依赖
    install_dependencies

    # 4. 检查文件
    if ! check_files; then
        exit 1
    fi

    # 5. 设置 PYTHONPATH
    setup_pythonpath

    # 6. 运行训练
    run_training
}

# 执行主流程
main

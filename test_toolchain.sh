#!/bin/bash
# 一键测试脚本 - 验证所有工具是否正常工作

echo "=========================================="
echo "Qwen3.5 工具链验证测试"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass_count=0
fail_count=0

# 检查函数
check() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓${NC} $1"
        ((pass_count++))
    else
        echo -e "${RED}✗${NC} $1"
        ((fail_count++))
    fi
}

echo "1. 检查脚本文件..."
[ -f "run_qwen35_configurable.sh" ]
check "run_qwen35_configurable.sh 存在"

[ -f "examples_qwen35_training.sh" ]
check "examples_qwen35_training.sh 存在"

[ -f "monitor_training.sh" ]
check "monitor_training.sh 存在"

[ -x "run_qwen35_configurable.sh" ]
check "run_qwen35_configurable.sh 可执行"

[ -x "examples_qwen35_training.sh" ]
check "examples_qwen35_training.sh 可执行"

[ -x "monitor_training.sh" ]
check "monitor_training.sh 可执行"

echo ""
echo "2. 检查文档文件..."
[ -f "README_可配置训练方案.md" ]
check "README_可配置训练方案.md 存在"

[ -f "可配置训练脚本使用指南.md" ]
check "可配置训练脚本使用指南.md 存在"

[ -f "快速开始.md" ]
check "快速开始.md 存在"

echo ""
echo "3. 检查必需文件..."
[ -f "tests/assets/tokenizer/tokenizer.json" ]
check "Tokenizer 文件存在"

[ -f "tests/assets/cc12m_test/cc12m-train-0000.tar" ]
check "测试数据集存在"

echo ""
echo "4. 检查脚本语法..."
bash -n run_qwen35_configurable.sh 2>/dev/null
check "run_qwen35_configurable.sh 语法正确"

bash -n examples_qwen35_training.sh 2>/dev/null
check "examples_qwen35_training.sh 语法正确"

bash -n monitor_training.sh 2>/dev/null
check "monitor_training.sh 语法正确"

echo ""
echo "=========================================="
echo "测试结果"
echo "=========================================="
echo -e "通过: ${GREEN}${pass_count}${NC} 项"
echo -e "失败: ${RED}${fail_count}${NC} 项"
echo ""

if [ $fail_count -eq 0 ]; then
    echo -e "${GREEN}✓ 所有检查通过！工具链已就绪。${NC}"
    echo ""
    echo "下一步："
    echo "  1. 快速测试: STEPS=1 bash run_qwen35_configurable.sh"
    echo "  2. 查看示例: bash examples_qwen35_training.sh"
    echo "  3. 阅读文档: cat README_可配置训练方案.md"
    exit 0
else
    echo -e "${RED}✗ 发现 ${fail_count} 个问题${NC}"
    echo ""
    echo "建议："
    echo "  1. 重新运行脚本创建过程"
    echo "  2. 检查文件权限: chmod +x *.sh"
    exit 1
fi

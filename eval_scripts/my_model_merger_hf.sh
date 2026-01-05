#!/bin/bash

# model_merger_hf.sh - Model merger script for HuggingFace format
# Usage: ./model_merger_hf.sh <base_path>
# Example: ./model_merger_hf.sh /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/rllm-mipo-local/exp/main_exp_0922/gpu_zw_stage1_train_dpscaler_1kq_8kr_xnode_smoothReward_1e-4/global_step_500

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -ne 1 ]; then
    echo "错误: 请提供基础路径参数"
    echo "用法: $0 <base_path>"
    exit 1
fi

# 获取基础路径参数
BASE_PATH="$1"

# 构建完整路径
LOCAL_DIR="${BASE_PATH}/actor/"
TARGET_DIR="${BASE_PATH}/actor/huggingface"

# 检查源目录是否存在
if [ ! -d "$LOCAL_DIR" ]; then
    echo "错误: 源目录不存在: $LOCAL_DIR"
    exit 1
fi

# 创建目标目录（如果不存在）
mkdir -p "$TARGET_DIR"

echo "========================================="
echo "Model Merger 脚本开始执行"
echo "========================================="
echo "基础路径: $BASE_PATH"
echo "源目录:   $LOCAL_DIR"
echo "目标目录: $TARGET_DIR"
echo "========================================="

# 执行 model merger 命令
echo "正在执行模型合并..."
python legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR" \
    --hf_model_path "$TARGET_DIR"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "✅ 模型合并成功完成!"
    echo "合并后的模型保存在: $TARGET_DIR"
    echo "========================================="
    
    # 显示目标目录内容
    echo "目标目录内容:"
    ls -la "$TARGET_DIR"
else
    echo "========================================="
    echo "❌ 模型合并失败!"
    echo "========================================="
    exit 1
fi

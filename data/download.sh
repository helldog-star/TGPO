#!/bin/bash

# Conda环境配置
CONDA_SH_PATH="/mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh"  # Conda初始化脚本路径
CONDA_ENV_NAME="your_env_name"  # Conda环境名称
BASE_DIR="your_model_dataset_save_dir"

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"  # 当前脚本所在目录，用于调用 align_tokenizer.py

source $CONDA_SH_PATH
conda activate $CONDA_ENV_NAME

SYMLINKS="False"

# ================= 1. 下载 Qwen2.5-Math-1.5B =================
echo "正在下载 Qwen2.5-Math-1.5B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-Math-1.5B-Instruct \
    --local-dir "$BASE_DIR/Qwen2.5-Math-1.5B-Instruct" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# ================= 2. 下载 Qwen2.5-Math-7B =================
echo "正在下载 Qwen2.5-Math-7B-Instruct..."
huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
    --local-dir "$BASE_DIR/Qwen2.5-Math-7B-Instruct" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# ================= 3. 下载 Qwen3-30B-A3B-Thinking-2507 =================
echo "正在下载 Qwen3-30B-A3B-Thinking-2507..."
huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
    --local-dir "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# ================= 4. 下载 Dataset (Openr1-Math-46k-8192) =================
echo "正在下载数据集 Elliott/Openr1-Math-46k-8192..."
huggingface-cli download Elliott/Openr1-Math-46k-8192 \
    --repo-type dataset \
    --local-dir "$BASE_DIR/datasets/Openr1-Math-46k-8192" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# ================= 5. 对齐 tokenizer（以 Qwen3-30B 为 teacher）并为 1.5B/7B 的 assistant 后加 <think> =================
echo "正在对齐 tokenizer: 1.5B、7B 以 30B 为 teacher, 并为 1.5B/7B 的 chat_template assistant 后加 <think> ..."
python "$DATA_DIR/align_tokenizer.py" \
    --student "$BASE_DIR/Qwen2.5-Math-1.5B-Instruct" \
    --teacher "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \
    --add-think
python "$DATA_DIR/align_tokenizer.py" \
    --student "$BASE_DIR/Qwen2.5-Math-7B-Instruct" \
    --teacher "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \
    --add-think

echo "✅ 所有下载与 tokenizer 对齐已完成！"
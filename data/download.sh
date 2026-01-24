#!/bin/bash

# Conda环境配置
CONDA_SH_PATH="/mnt/nvme3/liuxinyu/miniconda3/etc/profile.d/conda.sh"  # Conda初始化脚本路径
BASE_DIR="/mnt/nvme3/liuxinyu/models"

CONDA_ENV_NAME="tgpo"  # Conda环境名称
export HF_ENDPOINT=https://hf-mirror.com

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"  # 当前脚本所在目录，用于调用 align_tokenizer.py

source $CONDA_SH_PATH
conda activate $CONDA_ENV_NAME

# SYMLINKS="False"

# ================= 1. 下载 Qwen2.5-Math-1.5B =================
echo "正在下载 Qwen2.5-Math-1.5B..."
huggingface-cli download Qwen/Qwen2.5-Math-1.5B \
    --local-dir "$BASE_DIR/Qwen2.5-Math-1.5B" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# ================= 2. 下载 Qwen2.5-Math-7B =================
echo "正在下载 Qwen2.5-Math-7B..."
huggingface-cli download Qwen/Qwen2.5-Math-7B \
    --local-dir "$BASE_DIR/Qwen2.5-Math-7B" \
    --local-dir-use-symlinks $SYMLINKS \
    --resume-download

# # ================= 3. 下载 Qwen3-30B-A3B-Thinking-2507 =================
# echo "正在下载 Qwen3-30B-A3B-Thinking-2507..."
# huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
#     --local-dir "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \
#     --local-dir-use-symlinks $SYMLINKS \
#     --resume-download

# # ================= 4. 下载 Dataset (Openr1-Math-46k-8192) =================
# echo "正在下载数据集 Elliott/Openr1-Math-46k-8192..."
# huggingface-cli download Elliott/Openr1-Math-46k-8192 \
#     --repo-type dataset \
#     --local-dir "$BASE_DIR/datasets/Openr1-Math-46k-8192" \
#     --local-dir-use-symlinks $SYMLINKS \
#     --resume-download

# ================= 5. 对齐 tokenizer（以 Qwen3-30B 为 teacher）=================
echo "正在对齐 tokenizer: 1.5B、7B 以 30B 为 teacher..."
python "$DATA_DIR/align_tokenizer.py" \
    --student "$BASE_DIR/Qwen2.5-Math-1.5B" \
    --teacher "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \

python "$DATA_DIR/align_tokenizer.py" \
    --student "$BASE_DIR/Qwen2.5-Math-7B" \
    --teacher "$BASE_DIR/Qwen3-30B-A3B-Thinking-2507" \

echo "✅ 所有下载与 tokenizer 对齐已完成！"

# ================= 6. 手动调整config.json中的max_position_embeddings和rope_theta，调整chat_template的assistant后加<think>\n =================

# huggingface-cli download Qwen/Qwen3-0.6B \
#     --local-dir "/mnt/lxy/hf_models/TGPO/Qwen3-0.6B" \
#     --local-dir-use-symlinks False \
#     --resume-download

# huggingface-cli download Elliott/Openr1-Math-46k-8192 \
#     --repo-type dataset \
#     --local-dir "/mnt/lxy/hf_models/datasets/Openr1-Math-46k-8192" \
#     --local-dir-use-symlinks False \
#     --resume-download

# huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 \
#     --local-dir "/mnt/lxy/hf_models/Qwen3-30B-A3B-Thinking-2507" \
#     --local-dir-use-symlinks False \
#     --resume-download
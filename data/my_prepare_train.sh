#!/bin/bash

# Conda环境配置
CONDA_SH_PATH="/mnt/zhaorunsong/anaconda3/etc/profile.d/conda.sh"  # Conda初始化脚本路径
CONDA_ENV_NAME="tgpo"  # Conda环境名称
BASE_DIR="/tmp/hx/models"

source $CONDA_SH_PATH
conda activate $CONDA_ENV_NAME

MODEL_PATH="$BASE_DIR/Qwen3-30B-A3B-Thinking-2507-aligned"
LOG_FILE="./prepare_train_vllm_server.log"

# ---------- 传入 Python 的配置（可按需修改） ----------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR}"
DATASET_PATH="${DATASET_PATH:-$BASE_DIR/datasets/Openr1-Math-46k-8192}"
OUTPUT_PARQUET_PATH_ORI="$OUTPUT_DIR/openr1.parquet"
OUTPUT_PARQUET_PATH="$OUTPUT_DIR/openr1.tgpo.parquet"
TEMP_CACHE_JSON_PATH="$OUTPUT_DIR/openr1_tgpo_cache.json"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
MAX_WORKERS="${MAX_WORKERS:-32}"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "🚀 数据预处理..."
python prepare_train.py --dataset $DATASET_PATH --output $OUTPUT_PARQUET_PATH_ORI

echo "🚀 正在启动 vLLM 服务..."

# 2. 后台启动 vLLM (注意末尾的 & 和日志重定向)
# > $LOG_FILE 2>&1 把日志写到文件，避免和 Python 脚本输出混在一起
vllm serve $MODEL_PATH \
    --max-model-len 16384 \
    --enable-reasoning \
    --reasoning-parser deepseek_r1 \
    --tensor-parallel-size 8 \
    --trust-remote-code \
    > $LOG_FILE 2>&1 &

# 获取 vLLM 进程的 PID，用于稍后关闭
SERVER_PID=$!
echo "vLLM PID: $SERVER_PID"

# 3. 设置清理陷阱：无论脚本是正常结束还是被 Ctrl+C 中断，都会执行 kill
trap "echo '🛑 关闭 vLLM 服务...'; kill $SERVER_PID" EXIT

# 4. 循环检查 vLLM 是否启动完成
echo "⏳ 等待模型加载 (这可能需要几分钟)..."
echo "你可以通过 'tail -f $LOG_FILE' 查看加载进度"

while true; do
    # 尝试访问 vLLM 的健康检查接口
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "✅ vLLM 服务已就绪！"
        break
    fi
    sleep 10
done

# 5. 运行 Python 数据处理脚本（vLLM 采样）
echo "🐍 开始运行 Python 处理脚本（vLLM 采样）..."
python "$SCRIPT_DIR/my_prepare_train.py" \
    --dataset-path "$OUTPUT_PARQUET_PATH_ORI" \
    --model-local-path "$MODEL_PATH" \
    --output-parquet-path "$OUTPUT_PARQUET_PATH" \
    --temp-cache-json-path "$TEMP_CACHE_JSON_PATH" \
    --vllm-base-url "$VLLM_BASE_URL" \
    --max-workers "$MAX_WORKERS" \
    --batch-size "$BATCH_SIZE"

# 6. 验证采样结果（math_verify，输出 _correct / _wrong）
echo "🔍 验证采样结果..."
python "$SCRIPT_DIR/my_verify.py" --input "$OUTPUT_PARQUET_PATH"

# 7. 去掉 target 前的 <think>\n，输出 _correct_nothink.parquet
CORRECT_PARQUET="${OUTPUT_PARQUET_PATH%.parquet}_correct.parquet"
NOTHINK_PARQUET="${OUTPUT_PARQUET_PATH%.parquet}_correct_nothink.parquet"
if [[ -f "$CORRECT_PARQUET" ]]; then
    echo "📝 去掉 target 前的 <think>\\n ..."
    python "$SCRIPT_DIR/my_post_process.py" --input "$CORRECT_PARQUET" --output "$NOTHINK_PARQUET"
    echo "🎉 任务全部完成！最终训练数据: $NOTHINK_PARQUET"
else
    echo "⚠️ 未找到正确样本文件 $CORRECT_PARQUET，跳过 post_process。"
    echo "🎉 任务完成。"
fi

# 脚本退出时会自动触发 trap 里的 kill 命令
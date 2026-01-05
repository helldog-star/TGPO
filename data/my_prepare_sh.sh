#!/bin/bash

# 1. 环境初始化
eval "$(/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/miniconda3/bin/conda shell.bash hook)"
conda activate verl

MODEL_PATH="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen3-30B-A3B-Thinking-2507"
LOG_FILE="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/luffy/data/vllm_server.log"

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

# 5. 运行 Python 数据处理脚本
echo "🐍 开始运行 Python 处理脚本..."
python my_prepare_train.py

echo "🎉 任务全部完成！"

# 脚本退出时会自动触发 trap 里的 kill 命令
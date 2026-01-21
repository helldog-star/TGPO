# 加载环境配置（从项目根目录的 config/env.sh）
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TGPO_ROOT="$(cd "$_SCRIPT_DIR/../.." && pwd)"
. "$TGPO_ROOT/config/env.sh"

setup_all_env
which conda
which python

# 支持通过环境变量覆盖，便于 run_train_merge_eval.sh 等脚本调用
ROOT=${ROOT:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/luffy}
DATA=${DATA:-$ROOT/data/valid.all.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/luffy/eval_results/sh_train_qwen3_dpscaler_grpo_1kq_8kr_step300}
MODEL_PATH=${MODEL_PATH:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/rllm-mipo-local/exp/main_exp_0922/sh_train_qwen3_dpscaler_grpo_1kq_8kr/global_step_300/actor/huggingface}
MODEL_NAME=${MODEL_NAME:-luffy}

mkdir -p $OUTPUT_DIR

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system False \
  --force_generate False \
  --any_true True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
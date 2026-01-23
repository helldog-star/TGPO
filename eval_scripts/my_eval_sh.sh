# 若由上层脚本准备环境（如 run_train_merge_eval.sh），则无需重复设置
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="${PROJECT_ROOT:-$(cd "$_SCRIPT_DIR/.." && pwd)}"
if [[ -z "$PROJECT_ROOT" ]]; then
  . "$PROJ_ROOT/config/env.sh"
  setup_all_env
  which conda
  which python
fi

# 支持通过环境变量覆盖，便于 run_train_merge_eval.sh 等脚本调用
SAVE_DIR=${SAVE_DIR:-$PROJ_ROOT}
DATA=${DATA:-$PROJ_ROOT/data/valid.all.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-${SAVE_DIR}/checkpoints/${EXP_NAME}/eval_results/${EXP_NAME}_step${EVAL_STEP}}
MODEL_PATH=${MODEL_PATH:-${SAVE_DIR}/checkpoints/${EXP_NAME}/global_step_${EVAL_STEP}/actor/huggingface}
MODEL_NAME=${MODEL_NAME:-${EXP_NAME}}

mkdir -p "$OUTPUT_DIR"

if [ "$MODEL_NAME" == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ "$MODEL_NAME" == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" python generate_vllm.py \
  --model_path "$MODEL_PATH" \
  --input_file "$DATA" \
  --remove_system False \
  --force_generate False \
  --any_true True \
  --add_oat_evaluate True \
  --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
  --template "$TEMPLATE" > "$OUTPUT_DIR/$MODEL_NAME.log"
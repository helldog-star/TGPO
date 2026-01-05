eval "$(/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/miniconda/bin/conda shell.bash hook)"
which conda
conda activate luffy
which python

export VLLM_WORKER_MULTIPROC_METHOD=spawn

ROOT=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY
DATA=$ROOT/data/valid.all.parquet

# If you want to evaluate other models, you can change the model path and name.
# MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-Math-1.5B-Instruct

MODEL_PATH=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/checkpoints/luffy_qwen2d5_math_7b/global_step_450/actor/huggingface
OUTPUT_DIR=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/eval_results/luffy_qwen2d5_math_7b

mkdir -p $OUTPUT_DIR

MODEL_NAME=luffy

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0 python generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system False \
  --force_generate False \
  --any_true True \
  --add_oat_evaluate True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log
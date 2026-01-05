eval "$(/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/miniconda3/bin/conda shell.bash hook)"
which conda
conda activate luffy
which python

ROOT=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/luffy
DATA=$ROOT/data/valid.all.parquet

OUTPUT_DIR=$ROOT/luffy/eval_results/sh_train_qwen3_dpscaler_grpo_1kq_8kr_step300
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/rllm-mipo-local/exp/main_exp_0922/sh_train_qwen3_dpscaler_grpo_1kq_8kr/global_step_300/actor/huggingface
MODEL_NAME=luffy

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
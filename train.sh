#!/bin/bash
#
# 统一训练脚本：通过 --algo 切换，ALGO 列表与差异化参数见 config/tasks.sh
# 用法: ./train.sh --algo ALGO [--exp-name NAME] [--save-dir DIR] [--model-path P] [--teacher-model-path P] [--data-dir D] [--train-files F] [--val-files F] [其他 hydra 覆盖...]
#
# 示例:
#   ./train.sh --algo grpo
#   ./train.sh --algo luffy --exp-name luffy_7b_exp1
#   ./train.sh --algo kdrl --teacher-model-path /path/to/teacher
#   ./train.sh --algo tipo_adv data.train_batch_size=64  # 末尾可加任意 hydra 覆盖
#
# 各 algo 的 hydra 差异化覆盖、EXP_NAME、需否 teacher 等见 config/tasks.sh
#

set -x
export HF_ENDPOINT=https://hf-mirror.com

# -------------------- 加载 config 文件 --------------------
_SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$_SCRIPT_DIR"
. "${_SCRIPT_DIR}/config/tasks.sh"
. "${_SCRIPT_DIR}/config/env.sh"
# . "$PROJECT_ROOT/config/env_run.sh" # runsong use

# -------------------- 解析本脚本参数 --------------------
ALGO=""
EXP_NAME=""
SAVE_DIR=""
MODEL_PATH=""
TEACHER_MODEL_PATH=""
DATA_DIR=""
TRAIN_FILES=""
VAL_FILES=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --algo)                ALGO="$2"; shift 2 ;;
        --exp-name)            EXP_NAME="$2"; shift 2 ;;
        --save-dir)            SAVE_DIR="$2"; shift 2 ;;
        --root)                SAVE_DIR="$2"; shift 2 ;; # 兼容旧参数
        --model-path)          MODEL_PATH="$2"; shift 2 ;;
        --teacher-model-path)  TEACHER_MODEL_PATH="$2"; shift 2 ;;
        --data-dir)            DATA_DIR="$2"; shift 2 ;;
        --train-files)         TRAIN_FILES="$2"; shift 2 ;;
        --val-files)           VAL_FILES="$2"; shift 2 ;;
        *) break ;;
    esac
done

if [[ -z "$ALGO" ]]; then
    echo "用法: $0 --algo ALGO [--exp-name NAME] [--save-dir DIR] [--model-path P] [--teacher-model-path P] [--data-dir D] [--train-files F] [--val-files F] [hydra 覆盖...]"
    echo "  ALGO: $ALGOS （见 config/tasks.sh）"
    exit 1
fi

# -------------------- 默认推导 --------------------
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT}"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
TRAIN_FILES="${TRAIN_FILES:-$DATA_DIR/openr1.a3b_correct_35k.parquet}"
VAL_FILES="${VAL_FILES:-$DATA_DIR/valid.parquet}"
EXP_NAME="${EXP_NAME:-$(get_exp_name "$ALGO")}"

# 需要 teacher 的 algo：--teacher-model-path 须为有效路径（config: NEED_TEACHER_ALGOS）
if algo_needs_teacher "$ALGO"; then
    if [[ -z "$TEACHER_MODEL_PATH" || "$TEACHER_MODEL_PATH" == "none" || "$TEACHER_MODEL_PATH" == "None" ]]; then
        echo "错误: --algo $ALGO 需要有效的 --teacher-model-path（不能为 none）"
        exit 1
    fi
fi

# -------------------- 环境与目录 --------------------
setup_all_env
ray stop

export WANDB_PROJECT="tgpo"
export WANDB_MODE="offline"
export WANDB_API_KEY="b6d66b4632451b4d1908d9286fdafc46553519a7"
export WANDB_DIR=$SAVE_DIR/checkpoints/$EXP_NAME/wandb
export PROJ_DIR=$SAVE_DIR/checkpoints/$EXP_NAME

mkdir -p $WANDB_DIR

mkdir -p $PROJ_DIR/logs
LOG_FILE=$PROJ_DIR/logs/training_$(date +%Y%m%d_%H%M%S).log

cd $PROJECT_ROOT/luffy/verl/

# -------------------- 按 algo 设置差异化 hydra 覆盖（来自 config/tasks.sh get_algo_extra） --------------------
EXTRA=$(get_algo_extra "$ALGO")
if [[ -z "$EXTRA" ]]; then
    echo "不支持的 --algo: $ALGO（可选: $ALGOS）"
    exit 1
fi

# -------------------- 公共 hydra 参数 + 差异化 EXTRA + 用户覆盖 "$@" --------------------
python3 -m verl.mix_src.main_mix_ppo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.max_prefix_len=8192 \
    algorithm.kl_ctrl.kl_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXP_NAME" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.default_local_dir=$PROJ_DIR \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.use_sft_prefix_reward=False \
    actor_rollout_ref.rollout.prefix_share_across_samples=False \
    actor_rollout_ref.rollout.prefix_strategy=random \
    actor_rollout_ref.rollout.n_prefix=1 \
    actor_rollout_ref.rollout.prefix_reward_weight_alpha=1.0 \
    actor_rollout_ref.ref.use_ref=False \
    data.reward_impl_version=3 \
    trainer.max_optim_to_keep=2 \
    data.shuffle=True \
    trainer.default_hdfs_dir=null \
    trainer.total_training_steps=300 \
    $EXTRA \
    "$@" > >(tee $LOG_FILE) 2> >(tee ${LOG_FILE}.err >&2)

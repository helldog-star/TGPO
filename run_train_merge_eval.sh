#!/bin/bash
#
# 训练 -> 转 HF -> 评估 全自动流水线
# 用法: ./run_train_merge_eval.sh --task TASK [--step N] [--root ROOT] [--no-train] [--no-merge] [--no-eval]
#
# 示例:
#   ./run_train_merge_eval.sh --task grpo
#   ./run_train_merge_eval.sh --task luffy --step 300
#   ./run_train_merge_eval.sh --task kdrl --no-train --step 300   # 仅 merge + eval
#

set -e

TGPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=config/tasks.sh
. "$TGPO_ROOT/config/tasks.sh"
# shellcheck source=config/env.sh
. "$TGPO_ROOT/config/env.sh"

# 统一设置 conda 环境（所有步骤共用）
setup_all_env
# 须与 train.sh 中的 ROOT 一致，否则 merge 找不到 checkpoint
ROOT="${ROOT:-/tmp/hx/tgpo}"
# on-policy distill 方法用的 teacher 模型路径；不需要 teacher 的 algo 可传 none
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/tmp/hx/models/Qwen3-30B-A3B-Thinking-2507-aligned}"
EVAL_SCRIPTS_DIR="${EVAL_SCRIPTS_DIR:-$TGPO_ROOT/eval_scripts}"

# 解析参数
TASK=""
STEP=""
NO_TRAIN=false
NO_MERGE=false
NO_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)   TASK="$2"; shift 2 ;;
        --step)   STEP="$2"; shift 2 ;;
        --root)   ROOT="$2"; shift 2 ;;
        --no-train) NO_TRAIN=true; shift ;;
        --no-merge) NO_MERGE=true; shift ;;
        --no-eval)  NO_EVAL=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "用法: $0 --task TASK [--step N] [--root ROOT] [--no-train] [--no-merge] [--no-eval]"
    echo "  TASK: $ALGOS （见 config/tasks.sh）"
    echo "  --step: 指定 checkpoint 的 global_step，不传则用最新一次保存"
    exit 1
fi

algo_exists "$TASK" || { echo "不支持的 TASK: $TASK（可选: $ALGOS）"; exit 1; }
EXP_NAME=$(get_exp_name "$TASK")

PROJ_DIR="$ROOT/checkpoints/$EXP_NAME"
export PYTHONPATH="${TGPO_ROOT}:${PYTHONPATH}"

# ---------- 1. 训练：train.sh --algo（项目根目录 train.sh） ----------
if [[ "$NO_TRAIN" != true ]]; then
    echo "========== 1/3 训练: train.sh --algo $TASK =========="
    if [[ ! -f "$TGPO_ROOT/train.sh" ]]; then
        echo "训练脚本不存在: $TGPO_ROOT/train.sh"
        exit 1
    fi
    # 统一传 --teacher-model-path；不需要 teacher 的 algo 在 train.sh 中会忽略
    bash "$TGPO_ROOT/train.sh" --algo "$TASK" --root "$TGPO_ROOT" --teacher-model-path "$TEACHER_MODEL_PATH" || { echo "训练失败"; exit 1; }
else
    echo "========== 1/3 训练: 已跳过 (--no-train) =========="
fi

# ---------- 2. 转 HuggingFace：legacy_model_merger ----------
if [[ "$NO_MERGE" != true ]]; then
    echo "========== 2/3 转 HuggingFace: legacy_model_merger =========="
    if [[ -z "$STEP" ]]; then
        # 取最新 global_step_*
        LATEST="$(ls -d "$PROJ_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)"
        if [[ -z "$LATEST" ]]; then
            echo "未找到 checkpoint: $PROJ_DIR/global_step_*"
            exit 1
        fi
        STEP="$(basename "$LATEST" | sed 's/global_step_//')"
        echo "使用最新 checkpoint: global_step_$STEP"
    fi

    ACTOR_DIR="$PROJ_DIR/global_step_$STEP/actor"
    HF_DIR="$PROJ_DIR/global_step_$STEP/actor_hf"

    if [[ ! -d "$ACTOR_DIR" ]]; then
        echo "actor 目录不存在: $ACTOR_DIR"
        exit 1
    fi

    python "$EVAL_SCRIPTS_DIR/legacy_model_merger.py" merge \
        --backend fsdp \
        --local_dir "$ACTOR_DIR" \
        --target_dir "$HF_DIR" \
        || { echo "legacy_model_merger 失败"; exit 1; }
    echo "已保存 HF 模型: $HF_DIR"
else
    # 未执行 merge 时，需要用 --step 推断 HF 目录（可能之前已 merge 到 actor_hf 或 actor/huggingface）
    if [[ -z "$STEP" ]]; then
        LATEST="$(ls -d "$PROJ_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)"
        STEP="$(basename "$LATEST" | sed 's/global_step_//')"
    fi
    HF_DIR="$PROJ_DIR/global_step_$STEP/actor_hf"
    if [[ ! -d "$HF_DIR" ]]; then
        echo "未找到 HF 模型目录，请先运行 merge: $HF_DIR"
        exit 1
    fi
    echo "========== 2/3 转 HuggingFace: 已跳过，使用已有 $HF_DIR =========="
fi

# ---------- 3. 评估：my_eval_sh.sh ----------
if [[ "$NO_EVAL" != true ]]; then
    echo "========== 3/3 评估: my_eval_sh.sh =========="
    export MODEL_PATH="$HF_DIR"
    export MODEL_NAME="${EXP_NAME}"
    export ROOT="$ROOT"
    export OUTPUT_DIR="${OUTPUT_DIR:-$TGPO_ROOT/eval_results/${EXP_NAME}_step${STEP}}"
    export DATA="${DATA:-$TGPO_ROOT/data/valid.all.parquet}"
    mkdir -p "$OUTPUT_DIR"

    (cd "$EVAL_SCRIPTS_DIR" && bash my_eval_sh.sh) || { echo "评估失败"; exit 1; }
    echo "评估结果: $OUTPUT_DIR"
else
    echo "========== 3/3 评估: 已跳过 (--no-eval) =========="
fi

echo ""
echo "✅ 全流程完成. HF 模型: $HF_DIR"


# cd /path/to/TGPO
# chmod +x run_train_merge_eval.sh

# # 完整：训练 → 转 HF → 评估（step 自动取最新）
# ./run_train_merge_eval.sh --task grpo

# # 指定 step
# ./run_train_merge_eval.sh --task luffy --step 300

# # 只做 merge + 评估（例如训练已在别处跑完）
# ./run_train_merge_eval.sh --task kdrl --no-train --step 300

# # 使用自己的 ROOT
# ./run_train_merge_eval.sh --task grpo --root /your/project/root
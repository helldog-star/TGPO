#!/bin/bash
#
# 训练 -> 转 HF -> 评估 全自动流水线
# 用法: ./run_train_merge_eval.sh --task TASK [--eval-step N] [--save-dir DIR] [--no-train] [--no-merge] [--no-eval]
#
# 示例:
#   ./run_train_merge_eval.sh --task grpo
#   ./run_train_merge_eval.sh --task luffy --eval-step 300
#   ./run_train_merge_eval.sh --task kdrl --no-train --eval-step 300   # 仅 merge + eval
#

set -e

# 运行环境配置
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
# 训练/评估产物根目录
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT}"

export PYTHONPATH="$PROJECT_ROOT/luffy:$PROJECT_ROOT/luffy/verl:$PYTHONPATH"
# shellcheck source=config/tasks.sh
. "$PROJECT_ROOT/config/tasks.sh"
# shellcheck source=config/env.sh
. "$PROJECT_ROOT/config/env.sh"

# . "$PROJECT_ROOT/config/env_run.sh" # runsong use

# 统一设置 conda 环境（所有步骤共用）
setup_all_env


# 解析参数
TASK=""
EVAL_STEP=""
NO_TRAIN=false
NO_MERGE=false
NO_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --task)   TASK="$2"; shift 2 ;;
        --eval-step)   EVAL_STEP="$2"; shift 2 ;;
        --save-dir) SAVE_DIR="$2"; shift 2 ;;
        --root)   SAVE_DIR="$2"; shift 2 ;; # 兼容旧参数
        --no-train) NO_TRAIN=true; shift ;;
        --no-merge) NO_MERGE=true; shift ;;
        --no-eval)  NO_EVAL=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$TASK" ]]; then
    echo "用法: $0 --task TASK [--step N] [--save-dir DIR] [--no-train] [--no-merge] [--no-eval]"
    echo "  TASK: $ALGOS （见 config/tasks.sh）"
    echo "  --eval-step: 指定 checkpoint 的 global_step，不传则用最新一次保存"
    exit 1
fi

algo_exists "$TASK" || { echo "不支持的 TASK: $TASK（可选: $ALGOS）"; exit 1; }
EXP_NAME=$(get_exp_name "$TASK")

CKPT_DIR="$SAVE_DIR/checkpoints/$EXP_NAME"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# ---------- 1. 训练：train.sh --algo（项目根目录 train.sh） ----------
if [[ "$NO_TRAIN" != true ]]; then
    echo "========== 1/3 训练: train.sh --algo $TASK =========="
    if [[ ! -f "$PROJECT_ROOT/train.sh" ]]; then
        echo "训练脚本不存在: $PROJECT_ROOT/train.sh"
        exit 1
    fi
    # 统一传 --teacher-model-path；不需要 teacher 的 algo 在 train.sh 中会忽略
    bash "$PROJECT_ROOT/train.sh" \
        --algo "$TASK" \
        --save-dir "$SAVE_DIR" \
        --model-path "$MODEL_PATH" \
        --teacher-model-path "$TEACHER_MODEL_PATH" \
        || { echo "训练失败"; exit 1; }
else
    echo "========== 1/3 训练: 已跳过 (--no-train) =========="
fi

# ---------- 2. 转 HuggingFace：legacy_model_merger ----------
if [[ "$NO_MERGE" != true ]]; then
    echo "========== 2/3 转 HuggingFace: legacy_model_merger =========="
    if [[ -z "$EVAL_STEP" ]]; then
        # 取最新 global_step_*
        LATEST="$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)"
        if [[ -z "$LATEST" ]]; then
            echo "未找到 checkpoint: $CKPT_DIR/global_step_*"
            exit 1
        fi
        EVAL_STEP="$(basename "$LATEST" | sed 's/global_step_//')"
        echo "使用最新 checkpoint: global_step_$EVAL_STEP"
    fi

    ACTOR_DIR="$CKPT_DIR/global_step_$EVAL_STEP/actor"
    HF_DIR="$CKPT_DIR/global_step_$EVAL_STEP/actor_hf"

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
    # 未执行 merge 时，需要用 --eval-step 推断 HF 目录（可能之前已 merge 到 actor_hf 或 actor/huggingface）
    if [[ -z "$EVAL_STEP" ]]; then
        LATEST="$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)"
        EVAL_STEP="$(basename "$LATEST" | sed 's/global_step_//')"
    fi
    HF_DIR="$CKPT_DIR/global_step_$EVAL_STEP/actor_hf"
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
    export PROJECT_ROOT="$PROJECT_ROOT"
    export SAVE_DIR="$SAVE_DIR"
    OUTPUT_BASE="${OUTPUT_DIR:-$SAVE_DIR/checkpoints/${EXP_NAME}/eval_results/${EXP_NAME}_step${EVAL_STEP}}"
    mkdir -p "$OUTPUT_BASE"

    eval_sets=("valid.arc_c.parquet" "valid.gpqa.parquet" "valid.mmlu_pro.parquet" "valid.all.parquet")
    # eval_sets=("valid.gpqa.parquet")
    gpu_sets=("0,1,2,3" "4,5,6,7")

    run_eval_one() {
        local ds="$1"
        local gpus="$2"
        export DATA="$PROJECT_ROOT/data/${ds}"
        export OUTPUT_DIR="$OUTPUT_BASE/$ds"
        export CUDA_VISIBLE_DEVICES="$gpus"
        mkdir -p "$OUTPUT_DIR"
        echo "---- 评估数据集: $ds ($DATA) GPUs=$gpus ----"
        (cd "$EVAL_SCRIPTS_DIR" && bash my_eval_sh.sh)
    }

    idx=0
    running=0
    free_gpus=("${gpu_sets[@]}")
    declare -A pid_to_ds
    declare -A pid_to_gpu

    while [[ $idx -lt ${#eval_sets[@]} || $running -gt 0 ]]; do
        while [[ $running -lt ${#gpu_sets[@]} && $idx -lt ${#eval_sets[@]} ]]; do
            ds="${eval_sets[$idx]}"
            gpu="${free_gpus[0]}"
            free_gpus=("${free_gpus[@]:1}")

            run_eval_one "$ds" "$gpu" &
            pid=$!
            pid_to_ds[$pid]="$ds"
            pid_to_gpu[$pid]="$gpu"
            running=$((running + 1))
            idx=$((idx + 1))
        done

        if [[ $running -gt 0 ]]; then
            wait -n -p finished_pid
            status=$?
            finished_ds="${pid_to_ds[$finished_pid]}"
            if [[ $status -ne 0 ]]; then
                echo "评估失败: $finished_ds"
                exit 1
            fi
            free_gpus+=("${pid_to_gpu[$finished_pid]}")
            unset pid_to_ds[$finished_pid]
            unset pid_to_gpu[$finished_pid]
            running=$((running - 1))
        fi
    done
    echo "评估结果: $OUTPUT_BASE"
else
    echo "========== 3/3 评估: 已跳过 (--no-eval) =========="
fi

echo ""
echo "✅ 全流程完成. HF 模型: $HF_DIR"


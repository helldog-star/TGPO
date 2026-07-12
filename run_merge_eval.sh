#!/bin/bash
#
# Merge + Eval 一键脚本（不含训练；训练请用 exp_scripts_*/*.sh 单独跑）
#
# 用法:
#   ./run_merge_eval.sh --ckpt /path/to/EXP_NAME/global_step_100 [选项]
#
# 必填:
#   --ckpt DIR          指向某个 global_step_N 目录（其下应有 actor/ 分片）。
#                       脚本自动推导：
#                         源分片   = <ckpt>/actor
#                         HF 目标  = <ckpt>/actor_hf
#                         STEP     = N（从目录名解析）
#                         EXP_NAME = <ckpt> 的父目录名
#                         名字标签 = <EXP_NAME>_step<N>（用于输出目录与 MODEL_NAME）
#
# 可选:
#   --data FILE         评测集 parquet，默认 <TGPO_ROOT>/data/valid.all.parquet
#   --output-dir DIR    评测输出目录，默认 <TGPO_ROOT>/eval_results/<EXP_NAME>_step<N>
#   --no-merge          跳过 merge，直接用已存在的 <ckpt>/actor_hf 评测
#   --force-merge       即使 actor_hf 已存在也重新 merge
#   --no-eval           只 merge，不评测
#
# 示例:
#   ./run_merge_eval.sh --ckpt $ROOT/checkpoints/rkl_reg_dist_qwen2d5_math_1d5b_a3b35k/global_step_100
#   ./run_merge_eval.sh --ckpt .../global_step_100 --data /path/to/valid.all.parquet
#   ./run_merge_eval.sh --ckpt .../global_step_100 --no-merge   # actor_hf 已就绪，只评测

set -e

TGPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
# 仅需环境（conda / 代理 / vllm backend）；不再 source config/tasks.sh
# shellcheck source=config/env.sh
. "$TGPO_ROOT/config/env.sh"

EVAL_SCRIPTS_DIR="${EVAL_SCRIPTS_DIR:-$TGPO_ROOT/eval_scripts}"

# ---------- 解析参数 ----------
CKPT=""
DATA=""
OUTPUT_DIR=""
NO_MERGE=false
FORCE_MERGE=false
NO_EVAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt)         CKPT="$2"; shift 2 ;;
        --data)         DATA="$2"; shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --no-merge)     NO_MERGE=true; shift ;;
        --force-merge)  FORCE_MERGE=true; shift ;;
        --no-eval)      NO_EVAL=true; shift ;;
        -h|--help)
            sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

if [[ -z "$CKPT" ]]; then
    echo "错误: 必须提供 --ckpt <global_step_N 目录>"
    echo "用法: $0 --ckpt DIR [--data FILE] [--output-dir DIR] [--no-merge] [--force-merge] [--no-eval]"
    exit 1
fi

# 去掉可能的结尾斜杠
CKPT="${CKPT%/}"
if [[ ! -d "$CKPT" ]]; then
    echo "错误: ckpt 目录不存在: $CKPT"
    exit 1
fi

# ---------- 从 ckpt 路径推导命名 ----------
STEP_DIRNAME="$(basename "$CKPT")"          # e.g. global_step_100
STEP="${STEP_DIRNAME##*global_step_}"       # e.g. 100
if [[ "$STEP" == "$STEP_DIRNAME" ]]; then
    echo "警告: ckpt 目录名不是 global_step_N 形式（$STEP_DIRNAME），STEP 将留空"
    STEP=""
fi
EXP_NAME="$(basename "$(dirname "$CKPT")")" # 父目录名
MODEL_NAME="${EXP_NAME}${STEP:+_step${STEP}}"

ACTOR_DIR="$CKPT/actor"
HF_DIR="$CKPT/actor_hf"

# 默认评测集与输出目录
DATA="${DATA:-$TGPO_ROOT/data/valid.all.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$TGPO_ROOT/eval_results/${MODEL_NAME}}"

setup_all_env
which conda
which python

echo "========== 配置 =========="
echo "  ckpt        : $CKPT"
echo "  EXP_NAME    : $EXP_NAME"
echo "  STEP        : ${STEP:-<未解析>}"
echo "  MODEL_NAME  : $MODEL_NAME"
echo "  actor 分片  : $ACTOR_DIR"
echo "  HF 目标     : $HF_DIR"
echo "  评测集 DATA : $DATA"
echo "  输出目录    : $OUTPUT_DIR"
echo "=========================="

# ---------- 1. Merge: FSDP 分片 -> HuggingFace ----------
if [[ "$NO_MERGE" == true ]]; then
    echo "========== 1/2 Merge: 已跳过 (--no-merge) =========="
    if [[ ! -d "$HF_DIR" ]]; then
        echo "错误: --no-merge 但 HF 目录不存在: $HF_DIR"
        exit 1
    fi
else
    if [[ -f "$HF_DIR/config.json" && "$FORCE_MERGE" != true ]]; then
        echo "========== 1/2 Merge: 检测到已存在 $HF_DIR，跳过（--force-merge 可强制重做） =========="
    else
        echo "========== 1/2 Merge: FSDP -> HuggingFace =========="
        if [[ ! -d "$ACTOR_DIR" ]]; then
            echo "错误: actor 分片目录不存在: $ACTOR_DIR"
            exit 1
        fi
        python "$EVAL_SCRIPTS_DIR/legacy_model_merger.py" merge \
            --backend fsdp \
            --local_dir "$ACTOR_DIR" \
            --target_dir "$HF_DIR" \
            || { echo "merge 失败"; exit 1; }
        echo "已保存 HF 模型: $HF_DIR"
    fi
fi

# ---------- 2. Eval: vLLM 生成 + grader 判分 ----------
if [[ "$NO_EVAL" == true ]]; then
    echo "========== 2/2 Eval: 已跳过 (--no-eval) =========="
else
    echo "========== 2/2 Eval: my_eval_sh.sh =========="
    if [[ ! -f "$DATA" ]]; then
        echo "错误: 评测集不存在: $DATA（用 --data 指定正确的 valid.all.parquet）"
        exit 1
    fi
    export MODEL_PATH="$HF_DIR"
    export MODEL_NAME="$MODEL_NAME"
    export ROOT="$TGPO_ROOT"           # my_eval_sh 仅用作默认值兜底，实际路径均已被上面 export 覆盖
    export OUTPUT_DIR="$OUTPUT_DIR"
    export DATA="$DATA"
    mkdir -p "$OUTPUT_DIR"

    (cd "$EVAL_SCRIPTS_DIR" && bash my_eval_sh.sh) || { echo "评估失败"; exit 1; }
    echo "评估结果: $OUTPUT_DIR （逐 benchmark 准确率见 $OUTPUT_DIR/${MODEL_NAME}.log）"
fi

echo ""
echo "✅ 完成. HF 模型: $HF_DIR"

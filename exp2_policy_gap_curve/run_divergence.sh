#!/bin/bash
#
# 离线师生策略分歧一键流水线(回应 reviewer)。
# student = Qwen2.5-Math-1.5B; teachers = Qwen2.5-Math-7B(小 gap) + Qwen3-30B-A3B(大 gap)。
#
# 流程: student 采样 rollout -> 三模型(含 student 自己)teacher-forcing 打分
#        -> 三模型独立生成测长度/EOS -> 汇总(表 + 图 + json)。
#
# 关键约束: token 级 KL/top-k 只在共享词表下成立 => 三个模型都用 align_tokenizer.py 产出的
#           *-aligned 版本; prompt 的 token-ids 一律用 student tokenizer 构造(脚本已保证)。
#
# 用法:
#   bash run_divergence.sh                       # 用下方默认路径
#   OUT_DIR=/path NUM_PROMPTS=128 bash run_divergence.sh
#   只重算汇总: STAGE=aggregate bash run_divergence.sh
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="$(cd "$(dirname "$0")" && pwd)/measure_divergence.py"

# ---------- 环境(与训练一致) ----------
if [ -f "$REPO/config/env.sh" ]; then . "$REPO/config/env.sh"; setup_all_env || true; fi
export no_proxy="127.0.0.1,localhost"; export NO_PROXY="127.0.0.1,localhost"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
PY="${PY:-python}"

# ---------- 模型(改成你 align 后的真实路径) ----------
BASE="${BASE:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/models}"
STUDENT_PATH="${STUDENT_PATH:-$BASE/Qwen2.5-Math-1.5B-aligned}"
T7B_PATH="${T7B_PATH:-$BASE/Qwen2.5-Math-7B-aligned}"
T30B_PATH="${T30B_PATH:-$BASE/Qwen3-30B-A3B-Thinking-2507-aligned}"

# 张量并行(30B-A3B 在 24G 卡上需 TP>=4)
STUDENT_TP="${STUDENT_TP:-1}"
T7B_TP="${T7B_TP:-1}"
T30B_TP="${T30B_TP:-4}"

# ---------- 数据 / 采样 ----------
# 建议用 teacher-中立集(未按 A3B 筛)做主结果; 默认用全量 openr1(未筛)。
PARQUET="${PARQUET:-$REPO/data/openr1.parquet}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"       # 训练 rollout 温度
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_SCORE_TOKENS="${MAX_SCORE_TOKENS:-4096}"
TOPK="${TOPK:-20}"
MAX_LEN="${MAX_LEN:-10240}"
SEED="${SEED:-1234}"

OUT_DIR="${OUT_DIR:-$REPO/exp2_policy_gap_curve/divergence_out}"
STAGE="${STAGE:-all}"                    # all | generate | score | aggregate
mkdir -p "$OUT_DIR"

echo "=================== 配置 ==================="
echo " student : $STUDENT_PATH (tp=$STUDENT_TP)"
echo " 7B      : $T7B_PATH (tp=$T7B_TP)"
echo " 30B     : $T30B_PATH (tp=$T30B_TP)"
echo " parquet : $PARQUET"
echo " prompts=$NUM_PROMPTS n=$N_SAMPLES temp=$TEMPERATURE topk=$TOPK"
echo " out     : $OUT_DIR   stage=$STAGE"
echo "============================================"

gen() {  # $1=path $2=tag $3=tp
  echo ">>> [generate] $2"
  "$PY" "$SCRIPT" --mode generate --model "$1" --model-tag "$2" --tp "$3" \
    --out-dir "$OUT_DIR" --parquet "$PARQUET" \
    --prompt-tokenizer "$STUDENT_PATH" \
    --num-prompts "$NUM_PROMPTS" --n-samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" \
    --max-prompt-len "$MAX_PROMPT_LEN" --max-len "$MAX_LEN" --seed "$SEED"
}

score() {  # $1=path $2=tag $3=tp
  echo ">>> [score] $2"
  "$PY" "$SCRIPT" --mode score --model "$1" --model-tag "$2" --tp "$3" \
    --out-dir "$OUT_DIR" --student-tag student \
    --topk "$TOPK" --max-score-tokens "$MAX_SCORE_TOKENS" --max-len "$MAX_LEN" --seed "$SEED"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "generate" ]; then
  gen "$STUDENT_PATH" student "$STUDENT_TP"      # 必须先跑: 产出 prompts.jsonl + student rollouts
  gen "$T7B_PATH"     teacher7b "$T7B_TP"        # 仅为长度/EOS 独立生成
  gen "$T30B_PATH"    teacher30b "$T30B_TP"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "score" ]; then
  score "$STUDENT_PATH" student "$STUDENT_TP"    # student 自打分(取 student top-k / logp)
  score "$T7B_PATH"     teacher7b "$T7B_TP"
  score "$T30B_PATH"    teacher30b "$T30B_TP"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "aggregate" ]; then
  echo ">>> [aggregate]"
  "$PY" "$SCRIPT" --mode aggregate --out-dir "$OUT_DIR" \
    --student-tag student --teacher-tags teacher7b teacher30b --seed "$SEED"
fi

echo ""
echo "✅ 完成. 汇总: $OUT_DIR/divergence_summary.md (+ .json, plots/)"

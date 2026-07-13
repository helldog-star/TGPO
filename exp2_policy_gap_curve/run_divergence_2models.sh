#!/bin/bash
#
# 2-模型版离线师生策略分歧流水线(一个 student × 一个 teacher)。
# 换 TEACHER_PATH/TEACHER_TAG 跑多次即可覆盖「一 student 对多 teacher」的所有对比。
#
# 约束:token 级 KL / top-k overlap 只在共享词表下成立 => student 与 teacher 必须是
#       data/align_tokenizer.py 产出的 *-aligned 版本(pad 到统一 vocab + 对齐特殊 token)。
# 环境:统一用 conda vllm084 里的 vLLM 0.8.4(Qwen3-30B-A3B 是 MoE,0.6.3 加载不了;
#       Qwen2.5 系在 0.8.4 也原生支持,故三实验统一 vllm084)。
#
# 本仓库计划的三个实验(见 README「实验清单」):
#   ① student=qwen2.5_1.5b_math_aligned / teacher=qwen2.5_7b_math_aligned  divergence(小 gap)
#   ② student=qwen2.5_1.5b_math_aligned / teacher=qwen3_30b_a3b_aligned    divergence(大 gap)
#   ③ teacher=qwen3_30b_a3b_aligned     top-{5,10,100,1000} mass 覆盖率
#
# 用法:
#   TEACHER_PATH=... TEACHER_TAG=... bash run_divergence_2models.sh              # 全流程(generate→score→aggregate)
#   STAGE=coverage COVERAGE_TOPK=1000 TEACHER_PATH=... TEACHER_TAG=... bash ...   # 只跑覆盖率(实验③)
#   STAGE=generate|score|aggregate ...                                            # 分阶段调试/复算
#
# OUT_DIR 默认按 TEACHER_TAG 分目录,换 teacher 跑多次不互相覆盖;
# SEED 固定 => 同一 student 的 rollouts 每次一致、跨 teacher 可比。
set -eu

REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT="$(cd "$(dirname "$0")" && pwd)/measure_divergence.py"

# ---------- 环境(统一 vllm084) ----------
if [ -f "$REPO/config/env.sh" ]; then . "$REPO/config/env.sh"; setup_all_env || true; fi
export no_proxy="127.0.0.1,localhost"; export NO_PROXY="127.0.0.1,localhost"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 建议在集群/本机显式传:PY=/mnt/lxy/miniconda3/envs/vllm084/bin/python
PY="${PY:-python}"

# ---------- 模型(必须是 align 后的 *-aligned 版本) ----------
# BASE = 你存放 aligned 权重的目录(本机若无, 指向 dolphinfs 对应路径)。
BASE="${BASE:-/mnt/lxy/hf_models}"
STUDENT_PATH="${STUDENT_PATH:-$BASE/qwen2.5_1.5b_math_aligned}"
TEACHER_PATH="${TEACHER_PATH:-$BASE/qwen2.5_7b_math_aligned}"
TEACHER_TAG="${TEACHER_TAG:-teacher7b}"
STUDENT_TP="${STUDENT_TP:-1}"
TEACHER_TP="${TEACHER_TP:-1}"        # 30B-A3B(MoE)bf16 需 TP>=4(24G 卡);1.5B/7B 单卡即可
# score/coverage 的 prompt_logprobs 会在全词表上 materialize 大张量;
# gpu_mem 太高(默认0.85)会把显存全给 KV cache 导致 OOM。0.5 留足头寸;
# 覆盖率取 top-1000 时更吃显存,OOM 就再降到 0.4。
GPU_MEM="${GPU_MEM:-0.5}"

# ---------- 数据 / 采样(正式规模,对齐训练 n=8) ----------
# 建议用 teacher-中立集(未按 A3B 筛)做主结果; 默认用全量 openr1(未筛)。
PARQUET="${PARQUET:-$REPO/data/openr1.parquet}"
NUM_PROMPTS="${NUM_PROMPTS:-256}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"        # 训练 rollout 温度
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
MAX_SCORE_TOKENS="${MAX_SCORE_TOKENS:-4096}"
TOPK="${TOPK:-20}"                       # score 阶段 prompt_logprobs 的 k(top-k overlap / 截断 KL)
COVERAGE_TOPK="${COVERAGE_TOPK:-1000}"   # coverage 阶段 k;=1000 => 输出 top-{1,5,10,20,50,100,1000}
MAX_LEN="${MAX_LEN:-10240}"
SEED="${SEED:-1234}"

# 默认按 teacher 分目录:同一 student 换 teacher 跑多次时,各自结果不互相覆盖。
OUT_DIR="${OUT_DIR:-$REPO/exp2_policy_gap_curve/divergence_out_${TEACHER_TAG}}"
STAGE="${STAGE:-all}"                    # all | generate | score | aggregate | coverage
mkdir -p "$OUT_DIR"

echo "=================== 配置(2-模型) ==================="
echo " student : $STUDENT_PATH (tp=$STUDENT_TP)"
echo " teacher : $TEACHER_PATH (tp=$TEACHER_TP, tag=$TEACHER_TAG)"
echo " parquet : $PARQUET"
echo " prompts=$NUM_PROMPTS n=$N_SAMPLES temp=$TEMPERATURE topk=$TOPK cov_topk=$COVERAGE_TOPK max_tokens=$MAX_TOKENS max_len=$MAX_LEN"
echo " out     : $OUT_DIR   stage=$STAGE"
echo "===================================================="

gen() {  # $1=path $2=tag $3=tp
  echo ">>> [generate] $2"
  "$PY" "$SCRIPT" --mode generate --model "$1" --model-tag "$2" --tp "$3" \
    --gpu-mem "$GPU_MEM" \
    --out-dir "$OUT_DIR" --parquet "$PARQUET" \
    --prompt-tokenizer "$STUDENT_PATH" \
    --num-prompts "$NUM_PROMPTS" --n-samples "$N_SAMPLES" \
    --temperature "$TEMPERATURE" --max-tokens "$MAX_TOKENS" \
    --max-prompt-len "$MAX_PROMPT_LEN" --max-len "$MAX_LEN" --seed "$SEED"
}

score() {  # $1=path $2=tag $3=tp
  echo ">>> [score] $2"
  "$PY" "$SCRIPT" --mode score --model "$1" --model-tag "$2" --tp "$3" \
    --gpu-mem "$GPU_MEM" \
    --out-dir "$OUT_DIR" --student-tag student \
    --topk "$TOPK" --max-score-tokens "$MAX_SCORE_TOKENS" --max-len "$MAX_LEN" --seed "$SEED"
}

if [ "$STAGE" = "all" ] || [ "$STAGE" = "generate" ]; then
  gen "$STUDENT_PATH" student       "$STUDENT_TP"
  gen "$TEACHER_PATH" "$TEACHER_TAG" "$TEACHER_TP"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "score" ]; then
  score "$STUDENT_PATH" student       "$STUDENT_TP"
  score "$TEACHER_PATH" "$TEACHER_TAG" "$TEACHER_TP"
fi

if [ "$STAGE" = "all" ] || [ "$STAGE" = "aggregate" ]; then
  echo ">>> [aggregate]"
  "$PY" "$SCRIPT" --mode aggregate --out-dir "$OUT_DIR" \
    --student-tag student --teacher-tags "$TEACHER_TAG" --seed "$SEED"
fi

if [ "$STAGE" = "coverage" ]; then
  echo ">>> [coverage] $TEACHER_TAG top-k 覆盖率(优先复用 $OUT_DIR/rollouts_student.jsonl)"
  "$PY" "$SCRIPT" --mode coverage \
    --model "$TEACHER_PATH" --model-tag "$TEACHER_TAG" --tp "$TEACHER_TP" \
    --gpu-mem "$GPU_MEM" \
    --topk "$COVERAGE_TOPK" --out-dir "$OUT_DIR" \
    --max-score-tokens "$MAX_SCORE_TOKENS" \
    --parquet "$PARQUET" --prompt-tokenizer "$STUDENT_PATH" \
    --num-prompts "$NUM_PROMPTS" --max-len "$MAX_LEN" --seed "$SEED"
fi

echo ""
if [ "$STAGE" = "coverage" ]; then
  echo "✅ 完成. 覆盖率: $OUT_DIR/coverage_${TEACHER_TAG}.json"
else
  echo "✅ 完成. 汇总: $OUT_DIR/divergence_summary.md (+ .json, plots/)"
fi

#!/bin/bash
# 实验② Policy Gap Curve 扫描 driver
# 复用 3 个 baseline 脚本(已改为 env 可覆盖 TEACHER_MODEL_PATH/EXP_NAME/MODEL_PATH)，
# 在 teacher 阶梯 × {OPD(rkl), KDRL, TGPO(tgpo_reg)} 上逐一起 run。
# 单节点顺序执行；要并行就把内层循环拆到多节点各跑一行。
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT_DIR=$REPO/exp_scripts_qwen2d5_math_7b

# 固定 student（整条阶梯共用同一份、已 pad 到全局 max vocab 的 aligned student）
BASE=/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/liuxinyu67/models
export MODEL_PATH=${MODEL_PATH:-$BASE/Qwen2.5-Math-7B-aligned}

# ===== Teacher 阶梯：tag|aligned路径|该 run 的训练步数 =====
# 步数：OPD/KDRL 早崩可 150 早停；这里给统一上限，按需在下方按方法覆盖。
# TODO: 把下面每个 teacher 换成你 align_tokenizer.py 产出的 *-aligned 真实路径。
TEACHERS=(
  "mathins7b|$BASE/Qwen2.5-Math-7B-Instruct-aligned"
  "qwen7bins|$BASE/Qwen2.5-7B-Instruct-aligned"
  "r1qwen7b|$BASE/DeepSeek-R1-Distill-Qwen-7B-aligned"
  "r1qwen32b|$BASE/DeepSeek-R1-Distill-Qwen-32B-aligned"
  "a3b|$BASE/Qwen3-30B-A3B-Thinking-2507-aligned"
)

# 方法 → 脚本 → 训练步数（TGPO 跑满，OPD/KDRL 早停省算力；想统一就都设 300）
declare -A METHOD_SCRIPT=(
  [rkl]="$SCRIPT_DIR/train_rkl_7b.sh"
  [kdrl]="$SCRIPT_DIR/train_kdrl_7b.sh"
  [tgpo_reg]="$SCRIPT_DIR/train_tgpo_reg_7b.sh"
)
declare -A METHOD_STEPS=(
  [rkl]=150
  [kdrl]=150
  [tgpo_reg]=300
)

for entry in "${TEACHERS[@]}"; do
  ttag="${entry%%|*}"
  tpath="${entry#*|}"
  if [ ! -e "$tpath" ]; then
    echo "[WARN] teacher 路径不存在, 跳过: $tpath" >&2
    continue
  fi
  for method in rkl kdrl tgpo_reg; do
    script="${METHOD_SCRIPT[$method]}"
    steps="${METHOD_STEPS[$method]}"
    exp="exp2_${method}_${ttag}_7b"
    echo "==================================================================="
    echo "[EXP2] method=$method teacher=$ttag steps=$steps exp=$exp"
    echo "       teacher_path=$tpath"
    echo "==================================================================="
    EXP_NAME="$exp" \
    TEACHER_MODEL_PATH="$tpath" \
    MODEL_PATH="$MODEL_PATH" \
    TRAIN_STEPS="$steps" \
      bash "$script"
  done
done

echo "[EXP2] 全部扫描完成。x 轴: grep '[EXP2-GAP] step=1' 各 run 日志, 或读 wandb gap/reverse_kl_k1。"

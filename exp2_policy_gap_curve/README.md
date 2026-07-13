# 师生策略分歧量化（Teacher–Student Policy Divergence）

回应 reviewer：*"quantify teacher-student policy divergence directly — teacher perplexity on
student rollouts, token-level KL, top-k overlap, density-ratio statistics, EOS/length distribution."*

一条离线流水线：**student 采样 rollout → student 与某个 teacher 对同一串 token 做 teacher-forcing
打分 → 汇总全部分歧指标（表 + 图 + json）**。脚本按「一个 student × 一个 teacher」组织；
一个 student 对多个 teacher 只需**换 `TEACHER_PATH/TEACHER_TAG` 跑多次**（`SEED` 固定 =>
student rollouts 每次一致，跨 teacher 可比）。

---

## 实验清单（本仓库计划的三个实验）

| # | student | teacher | 内容 | 预期 gap |
|---|---|---|---|---|
| ① | `qwen2.5_1.5b_math_aligned` | `qwen2.5_7b_math_aligned` | divergence 全指标 | 小 |
| ② | `qwen2.5_1.5b_math_aligned` | `qwen3_30b_a3b_aligned` | divergence 全指标 | 大 |
| ③ | —（在 ② 的 student rollouts 上） | `qwen3_30b_a3b_aligned` | top-{5,10,100,1000} mass 覆盖率 | — |

论点抓手：**30B（②）在每个分歧指标上都应显著大于 7B（①）**，正面支撑论文
「large policy divergence」主张；③ 说明 forward-KL 的 top-k 截断在 30B teacher 上丢了多少尾部质量。

---

## 文件

| 文件 | 作用 |
|---|---|
| `run_divergence_2models.sh` | 一键驱动（一 student × 一 teacher）：generate → score → aggregate；`STAGE=coverage` 跑覆盖率 |
| `measure_divergence.py` | 工作脚本，四模式：`generate` / `score` / `aggregate` / `coverage` |
| `divergence_out_<TEACHER_TAG>/` | 各 teacher 的输出目录（按 tag 自动分开） |

> 训练时的 `gap/reverse_kl_k1`、`gap/mismatch_ratio`（wandb，见 `mix_trainer.py` 的 `[EXP2-GAP]`
> 探针）是**在线**版本，随任何开了 teacher 的训练 run 自动记录；本目录是**离线**版本，
> 覆盖 reviewer 要的全部 5 类指标，且可对任意 checkpoint / 模型对做。两者互补。

---

## 前置条件（硬约束）

1. **共享词表**：token 级 KL / top-k overlap 只在同一套 BPE 词表下成立。student 与 teacher 必须是
   `data/align_tokenizer.py` 产出的 `*-aligned` 版本（pad 到统一 vocab + 对齐特殊 token）。
   脚本在 `score` 阶段会自检：若序列里最大 token id ≥ 模型 `vocab_size` 直接报错。
2. **vLLM 环境（统一 vllm084 / vLLM 0.8.4）**：实验 ②③ 的 `qwen3_30b_a3b` 是 **Qwen3-MoE
   （`Qwen3MoeForCausalLM`）**，旧的 `tgpo`（vLLM 0.6.3）**加载不了**；而 Qwen2.5 系
   （`Qwen2ForCausalLM`）在 0.8.4 也原生支持，故**三个实验统一用 `vllm084`**。
   显式传：`PY=/mnt/lxy/miniconda3/envs/vllm084/bin/python`。
   > 注：`measure_divergence.py` 原按 0.6.3 API 写（`prompt_logprobs` / `TokensPrompt`），
   > 0.8.4 大体兼容但首次跑建议先小规模确认 `out.prompt_logprobs` 结构无变化。
3. **显存**：`qwen3_30b_a3b`（MoE）bf16 约 60GB，24G 卡需 `TEACHER_TP>=4`；1.5B/7B 单卡即可。
   `score`/`coverage` 的 `prompt_logprobs` 会 materialize 大张量，`GPU_MEM` 默认 0.5 留头寸，
   覆盖率取 top-1000 更吃显存，OOM 就降到 0.4。

---

## 快速开始（三个实验）

```bash
cd /mnt/lxy/TGPO/exp2_policy_gap_curve

# 公共环境：统一 vllm084；BASE 指向你 align 后权重所在目录
COMMON="PY=/mnt/lxy/miniconda3/envs/vllm084/bin/python \
  CONDA_ENV_NAME=vllm084 CONDA_SH_PATH=/mnt/lxy/miniconda3/etc/profile.d/conda.sh \
  BASE=/path/to/aligned_models STUDENT_PATH=/path/to/aligned_models/qwen2.5_1.5b_math_aligned"

# ---------- 实验①：teacher = 7B-Math（小 gap，单卡）----------
eval $COMMON CUDA_VISIBLE_DEVICES=0 \
  TEACHER_PATH=/path/to/aligned_models/qwen2.5_7b_math_aligned TEACHER_TAG=teacher7b \
  bash run_divergence_2models.sh

# ---------- 实验②：teacher = 30B-A3B（大 gap，MoE 需 TP>=4）----------
eval $COMMON CUDA_VISIBLE_DEVICES=0,1,2,3 \
  TEACHER_PATH=/path/to/aligned_models/qwen3_30b_a3b_aligned TEACHER_TAG=teacher30b TEACHER_TP=4 \
  bash run_divergence_2models.sh

# ---------- 实验③：30B 的 top-{5,10,100,1000} 覆盖率（复用②的 student rollouts）----------
# 与②同一 TEACHER_TAG => 同一 OUT_DIR，自动复用 rollouts_student.jsonl（训练同口径）
eval $COMMON CUDA_VISIBLE_DEVICES=0,1,2,3 \
  STAGE=coverage COVERAGE_TOPK=1000 GPU_MEM=0.4 \
  TEACHER_PATH=/path/to/aligned_models/qwen3_30b_a3b_aligned TEACHER_TAG=teacher30b TEACHER_TP=4 \
  bash run_divergence_2models.sh
```

分阶段跑（调试或复算）：

```bash
STAGE=generate  ...  bash run_divergence_2models.sh   # 只采样 + 独立生成
STAGE=score     ...  bash run_divergence_2models.sh   # 只打分（需已有 rollouts）
STAGE=aggregate ...  bash run_divergence_2models.sh   # 只汇总（不加载模型，秒级）
STAGE=coverage  ...  bash run_divergence_2models.sh   # 只测覆盖率
```

---

## 可配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `PY` | `python` | **建议显式**指向 `.../envs/vllm084/bin/python` |
| `BASE` | `/mnt/lxy/hf_models` | aligned 权重所在目录（本机若无, 指向 dolphinfs） |
| `STUDENT_PATH` | `$BASE/qwen2.5_1.5b_math_aligned` | student（**aligned**）|
| `TEACHER_PATH` / `TEACHER_TAG` | `$BASE/qwen2.5_7b_math_aligned` / `teacher7b` | teacher（**aligned**）+ 短标签（决定 OUT_DIR）|
| `STUDENT_TP` / `TEACHER_TP` | 1 / 1 | 张量并行；30B-A3B 需 `TEACHER_TP>=4` |
| `GPU_MEM` | 0.5 | `gpu_memory_utilization`；覆盖率 top-1000 OOM 时降到 0.4 |
| `PARQUET` | `data/openr1.parquet` | prompt 来源；**建议 teacher-中立集**，勿用 `openr1.a3b_correct_35k`（偏 teacher，低估 gap）|
| `NUM_PROMPTS` | 256 | 抽样 prompt 数 |
| `N_SAMPLES` | 8 | 每 prompt rollout 数（对齐训练 n=8）|
| `TEMPERATURE` | 1.0 | rollout 温度（对齐训练 rollout 温度，量的是被正则的那个分布）|
| `MAX_TOKENS` | 8192 | 生成最大长度 |
| `MAX_SCORE_TOKENS` | 4096 | 每条 response 打分的最大 token 数（控显存/时长）|
| `TOPK` | 20 | `score` 阶段 `prompt_logprobs` 的 k（top-k overlap / 截断 KL）|
| `COVERAGE_TOPK` | 1000 | `coverage` 阶段 k；=1000 => 输出 top-{1,5,10,20,50,100,1000} |
| `MAX_LEN` | 10240 | vLLM `max_model_len`，需 ≥ prompt+response |
| `SEED` | 1234 | 抽样与采样种子（可复现）|
| `OUT_DIR` | `divergence_out_<TEACHER_TAG>` | 输出目录（按 teacher 自动分开）|
| `STAGE` | all | `all` / `generate` / `score` / `aggregate` / `coverage` |

---

## 流程（run_divergence_2models.sh 内部，以 TEACHER_TAG=teacher30b 为例）

```
[STAGE=all]
generate student    → prompts.jsonl（用 student tokenizer 统一构造 prompt ids）
                      + rollouts_student.jsonl（student 采样序列）
generate teacher30b → gensummary_teacher30b.json（独立生成，测长度/EOS）
score    student    → scores_student.jsonl（student 自打分：logp + top-k）
score    teacher30b → scores_teacher30b.jsonl（teacher-forcing 在 student 序列上）
aggregate           → divergence_summary.{json,md} + plots/

[STAGE=coverage]
coverage teacher30b → coverage_teacher30b.json（在 rollouts_student.jsonl 上测 top-k 覆盖率）
```

关键：**所有打分都在「student 的同一串 prompt+response token ids」上做**（与训练里 teacher
worker 的 teacher-forcing 完全同口径），因此指标可比、可对齐、可算逐 token KL。

---

## 产物说明（`divergence_out_<TEACHER_TAG>/`）

| 文件 | 内容 |
|---|---|
| `prompts.jsonl` | 抽样 prompt（pid、data_source、prompt_token_ids）；本目录内共享 |
| `rollouts_<tag>.jsonl` | 各模型生成序列（token ids、长度、finish_reason）|
| `gensummary_<tag>.json` | 各模型长度/EOS 汇总 + 长度直方图 |
| `scores_<tag>.jsonl` | 各模型逐 token：实际 token logp + top-k ids/logp |
| `divergence_summary.md` | **人看的主表**（下面各指标 + 95% CI）|
| `divergence_summary.json` | 机器可读的完整结果 |
| `coverage_<tag>.json` | `coverage` 模式产物：top-{1,5,10,...} 累积质量 |
| `plots/divergence_bars.png` | reverse_kl_k1 / mismatch_ratio 条形图 |
| `plots/length_dist.png` | student/teacher 生成长度分布 |

> 横比两个 teacher：分别跑出 `divergence_out_teacher7b/` 与 `divergence_out_teacher30b/`，
> 两份 `divergence_summary.md` 因 student rollouts 同源（同 SEED）而可直接对照。

---

## 指标定义（都在 student rollouts 上，按 prompt bootstrap 95% CI）

| 指标 | 定义 | 读法 |
|---|---|---|
| `teacher_ppl` | `exp(−mean_t logπ_T(a_t))` | teacher 对学生文本的困惑度；越高=分歧越大 |
| `reverse_kl_k1` | `mean_t [logπ_θ(a_t) − logπ_T(a_t)]` | 反向 KL 的单样本无偏估计（=训练 `gap/reverse_kl_k1` 口径）|
| `reverse_kl_topk` / `forward_kl_topk` | 两侧 top-k 并集支撑上的截断 KL，`D(π_θ‖π_T)` / `D(π_T‖π_θ)` | 低方差近似，双向都给 |
| `mismatch_ratio` | `mean_t 1[teacher_top1 ≠ a_t]` | top-1 不一致率（=训练 `gap/mismatch_ratio` 口径）|
| `top1/top5/top10_overlap` | `mean_t \|topk_T ∩ topk_θ\| / k` | 支撑集重合度 |
| density-ratio | `logρ=logπ_θ−logπ_T` 的 mean/std/p5/p50/p95、`frac_rho_gt1` | 分布形态；`frac_rho_gt1`=学生比老师更自信的 token 占比 |
| 长度/EOS | 各模型 `len_mean/median/p95`、`trunc_rate`、`stop_rate` | OPD/KDRL 的 length explosion 靠这个坐实 |

**方法学注意**：CI 按 prompt 而非 token 做 bootstrap（同一条回答内 token 相关，按 token 会虚低）；
`reverse_kl_k1` 是精确的 on-policy 估计，`*_kl_topk` 是 top-k 截断近似（标注清楚即可）。

---

## 实验③：top-k 覆盖率（`--mode coverage`，纯推理，不用训练）

回答「forward-KL top-k（训练默认 `algorithm.topk_k=100`）在 teacher top-{5,10,100,1000} 上覆盖了
多少概率质量、尾部丢了多少」。原理：vLLM `prompt_logprobs` 是全词表 log-softmax，`exp(logp)` 即真概率，
逐位置累加 top-k 概率即 coverage@k。

用 `run_divergence_2models.sh STAGE=coverage COVERAGE_TOPK=1000` 即可（见上「快速开始」实验③）。
底层等价命令：

```bash
# 有 student rollouts 就在其上测(同训练口径); 没有则自动用 parquet 的 target(参考解), 免生成
PY=/mnt/lxy/miniconda3/envs/vllm084/bin/python
$PY measure_divergence.py --mode coverage \
  --model /path/qwen3_30b_a3b_aligned --model-tag teacher30b \
  --tp 4 --topk 1000 --gpu-mem 0.4 --out-dir divergence_out_teacher30b \
  --parquet ../data/openr1.parquet --prompt-tokenizer /path/qwen2.5_1.5b_math_aligned \
  --num-prompts 256
```

- **`--topk` 决定输出哪些 k**：脚本取 `ks = {1,5,10,20,50,100} ∪ {K}` 中 ≤K 的项，故
  `--topk 1000` 会给出 **top-{1,5,10,20,50,100,1000}**（含你要的 5/10/100/1000）。
- **`max_logprobs` 自动放开**：vLLM 默认 `max_logprobs=20`，脚本按 `--topk` 放开到 `K+1`
  （`LLM(max_logprobs=K+1)`）。top-1000 => `max_logprobs=1001`，更吃显存，配合 `GPU_MEM=0.4` 与 `TP>=4`。
- 产物 `coverage_<tag>.json`：`coverage.top{1,5,10,20,50,100,1000}` 的 mean/median/p5/p95、
  `frac_top100_below_0.9`（尾部重的位置占比）、`trunc_entropy_topk`。
- 读法：若 top100 覆盖 ≈0.95、`frac_top100_below_0.9` 很小 → 训练 top-100 截断几乎无损；
  若覆盖偏低（尾部到 top-1000 才补齐）→ forward-KL 丢了可观尾部质量，可加大 `K` 或改看 reverse（student top-k 支撑）。

---
# 师生策略分歧量化（Teacher–Student Policy Divergence）

回应 reviewer：*"quantify teacher-student policy divergence directly — teacher perplexity on
student rollouts, token-level KL, top-k overlap, density-ratio statistics, EOS/length distribution."*

一条离线流水线：**student 采样 rollout → 三个模型对同一串 token 做 teacher-forcing 打分 →
汇总全部分歧指标（表 + 图 + json）**。默认三模型：

| 角色 | 模型 | 预期 gap |
|---|---|---|
| student | Qwen2.5-Math-1.5B | — |
| teacher（小 gap） | Qwen2.5-Math-7B | 小 |
| teacher（大 gap） | Qwen3-30B-A3B-Thinking | 大 |

论点抓手：**30B 在每个分歧指标上都应显著大于 7B**，正面支撑论文「large policy divergence」主张。

---

## 文件

| 文件 | 作用 |
|---|---|
| `run_divergence.sh` | 一键驱动：generate → score → aggregate |
| `measure_divergence.py` | 工作脚本，三模式：`generate` / `score` / `aggregate` |
| `divergence_out/`（默认输出目录） | 所有中间产物与最终汇总 |

> 训练时的 `gap/reverse_kl_k1`、`gap/mismatch_ratio`（wandb，见 `mix_trainer.py` 的 `[EXP2-GAP]`
> 探针）是**在线**版本，随任何开了 teacher 的训练 run 自动记录；本目录是**离线**版本，
> 覆盖 reviewer 要的全部 5 类指标，且可对任意 checkpoint / 模型对做。两者互补。

---

## 前置条件（硬约束）

1. **共享词表**：token 级 KL / top-k overlap 只在同一套 BPE 词表下成立。三个模型必须是
   `data/align_tokenizer.py` 产出的 `*-aligned` 版本（pad 到统一 vocab + 对齐特殊 token）。
   脚本在 `score` 阶段会自检：若序列里最大 token id ≥ 模型 `vocab_size` 直接报错。
2. **vLLM 环境**：在训练用的 conda 环境（本仓库 `config/env.sh` 默认 `tgpo`；集群上是 `luffy`）里跑。
3. **显存**：Qwen3-30B-A3B（MoE）bf16 约 60GB，24G 卡需 `T30B_TP>=4`。1.5B/7B 单卡即可。

---

## 快速开始

```bash
cd /mnt/lxy/TGPO/exp2_policy_gap_curve

# 改成你 align 后的真实路径；其余用默认
STUDENT_PATH=/path/Qwen2.5-Math-1.5B-aligned \
T7B_PATH=/path/Qwen2.5-Math-7B-aligned \
T30B_PATH=/path/Qwen3-30B-A3B-Thinking-2507-aligned \
T30B_TP=4 \
bash run_divergence.sh
```

分阶段跑（调试或复算）：

```bash
STAGE=generate  bash run_divergence.sh   # 只采样 + 独立生成
STAGE=score     bash run_divergence.sh   # 只打分（需已有 rollouts）
STAGE=aggregate bash run_divergence.sh   # 只汇总（不加载模型，秒级）
```

---

## 可配置项（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `STUDENT_PATH` / `T7B_PATH` / `T30B_PATH` | dolphinfs 上的 aligned 路径 | 三个模型（**必改**成你的真实路径） |
| `STUDENT_TP` / `T7B_TP` / `T30B_TP` | 1 / 1 / 4 | 张量并行 |
| `PARQUET` | `data/openr1.parquet` | prompt 来源；**建议 teacher-中立集**，勿用 `openr1.a3b_correct_35k`（偏 teacher，低估 gap） |
| `NUM_PROMPTS` | 256 | 抽样 prompt 数 |
| `N_SAMPLES` | 8 | 每 prompt rollout 数（对齐训练 n=8） |
| `TEMPERATURE` | 1.0 | rollout 温度（对齐训练 rollout 温度，量的是被正则的那个分布） |
| `MAX_TOKENS` | 8192 | 生成最大长度 |
| `MAX_SCORE_TOKENS` | 4096 | 每条 response 打分的最大 token 数（控显存/时长） |
| `TOPK` | 20 | `prompt_logprobs` 的 k（top-k overlap / 截断 KL 用） |
| `MAX_LEN` | 10240 | vLLM `max_model_len`，需 ≥ prompt+response |
| `SEED` | 1234 | 抽样与采样种子（可复现） |
| `OUT_DIR` | `exp2_policy_gap_curve/divergence_out` | 输出目录 |
| `STAGE` | all | `all` / `generate` / `score` / `aggregate` |

---

## 流程（run_divergence.sh 内部）

```
generate student   → prompts.jsonl（用 student tokenizer 统一构造 prompt ids）
                     + rollouts_student.jsonl（student 采样序列）
generate teacher7b → gensummary_teacher7b.json（独立生成，测长度/EOS）
generate teacher30b→ gensummary_teacher30b.json
score    student   → scores_student.jsonl（student 自打分：logp + top-k）
score    teacher7b → scores_teacher7b.jsonl（teacher-forcing 在 student 序列上）
score    teacher30b→ scores_teacher30b.jsonl
aggregate          → divergence_summary.{json,md} + plots/
```

关键：**所有打分都在「student 的同一串 prompt+response token ids」上做**（与训练里 teacher
worker 的 teacher-forcing 完全同口径），因此指标可比、可对齐、可算逐 token KL。

---

## 产物说明（`divergence_out/`）

| 文件 | 内容 |
|---|---|
| `prompts.jsonl` | 抽样 prompt（pid、data_source、prompt_token_ids）；全流程共享 |
| `rollouts_<tag>.jsonl` | 各模型生成序列（token ids、长度、finish_reason） |
| `gensummary_<tag>.json` | 各模型长度/EOS 汇总 + 长度直方图 |
| `scores_<tag>.jsonl` | 各模型逐 token：实际 token logp + top-k ids/logp |
| **`divergence_summary.md`** | **人看的主表**（下面各指标 + 95% CI） |
| `divergence_summary.json` | 机器可读的完整结果 |
| `plots/divergence_bars.png` | reverse_kl_k1 / mismatch_ratio 条形图 |
| `plots/length_dist.png` | 三模型生成长度分布 |

---

## 指标定义（都在 student rollouts 上，按 prompt bootstrap 95% CI）

| 指标 | 定义 | 读法 |
|---|---|---|
| `teacher_ppl` | `exp(−mean_t logπ_T(a_t))` | teacher 对学生文本的困惑度；越高=分歧越大 |
| `reverse_kl_k1` | `mean_t [logπ_θ(a_t) − logπ_T(a_t)]` | 反向 KL 的单样本无偏估计（=训练 `gap/reverse_kl_k1` 口径）|
| `reverse_kl_topk` / `forward_kl_topk` | 在两侧 top-k 并集支撑上的截断 KL，`D(π_θ‖π_T)` / `D(π_T‖π_θ)` | 低方差近似，双向都给 |
| `mismatch_ratio` | `mean_t 1[teacher_top1 ≠ a_t]` | top-1 不一致率（=训练 `gap/mismatch_ratio` 口径）|
| `top1/top5/top10_overlap` | `mean_t |topk_T ∩ topk_θ| / k` | 支撑集重合度 |
| density-ratio | `logρ=logπ_θ−logπ_T` 的 mean/std/p5/p50/p95、`frac_rho_gt1` | 分布形态；`frac_rho_gt1`=学生比老师更自信的 token 占比 |
| 长度/EOS | 各模型 `len_mean/median/p95`、`trunc_rate`、`stop_rate` | OPD/KDRL 的 length explosion 靠这个坐实 |

**方法学注意**：CI 按 prompt 而非 token 做 bootstrap（同一条回答内 token 相关，按 token 会虚低）；
`reverse_kl_k1` 是精确的 on-policy 估计，`*_kl_topk` 是 top-k 截断近似（标注清楚即可）。

---

## 附:top-k 覆盖率测试（`--mode coverage`，纯推理，不用训练）

回答「forward-KL top-k（训练默认 `algorithm.topk_k=100`）在 teacher top-5/10/100 上覆盖了多少概率质量、
尾部丢了多少」。原理：vLLM `prompt_logprobs` 是全词表 log-softmax，`exp(logp)` 即真概率，
逐位置累加 top-k 概率即 coverage@k。

```bash
# 有 student rollouts 就在其上测(同训练口径); 没有则自动用 parquet 的 target(参考解), 免生成
python measure_divergence.py --mode coverage \
  --model /path/Qwen3-30B-A3B-Thinking-2507-aligned --model-tag teacher30b \
  --tp 4 --topk 100 --out-dir divergence_out \
  --parquet ../data/openr1.parquet --prompt-tokenizer /path/Qwen2.5-Math-1.5B-aligned \
  --num-prompts 128
```

- **注意**：vLLM 默认 `max_logprobs=20`，脚本会自动按 `--topk` 放开到 100+（`LLM(max_logprobs=K+1)`）。
- 产物 `coverage_<tag>.json`：`coverage.top{1,5,10,20,50,100}` 的 mean/median/p5/p95、
  `frac_top100_below_0.9`（尾部重的位置占比）、`trunc_entropy_topk`。
- 读法：若 top100 覆盖 ≈0.95、`frac_top100_below_0.9` 很小 → 训练 top-100 截断几乎无损；
  若覆盖偏低 → forward-KL 丢了可观尾部质量，可加大 `K` 或改看 reverse（student top-k 支撑）。

## 常见延伸

- **沿训练演化**：把 `STUDENT_PATH` 指向某个 `global_step_N/actor_hf`（用 `run_merge_eval.sh` 合出来的
  HF 权重），重跑即可得到「分歧随训练收窄」的曲线。
- **跨方法对比**：对 tgpo / rkl / kdrl 各自的 checkpoint 分别跑，横比同一分歧指标。
- **鲁棒性**：主结果用训练 prompt，附录再用 teacher-中立集（MATH/OpenR1 未筛子集）复现，证明结论非筛选所致。

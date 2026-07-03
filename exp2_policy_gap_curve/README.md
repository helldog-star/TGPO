# 实验② Policy Gap Curve（teacher–student 分歧 vs 方法鲁棒性）

## 目标与论点
横轴 = teacher 与 student 的初始 policy gap，纵轴 = 蒸馏后 student 的下游表现。
画三条曲线 **TGPO / KDRL / OP Distill(RKL)**，论证：

> gap 增大时 KDRL、OP Distill 崩（性能掉到 base 以下、length explosion），
> 而 TGPO 在同样的 teacher、同样的 gap 下仍稳健提升。

这条曲线是论文「large policy divergence」主张的核心实证图。

---

## 横轴：用「测量值」而不是「teacher 标签」

x 轴用 **step-1 实测的逐 token 反向 KL（k1 估计量）**：
`gap/reverse_kl_k1 = E_{y~π_θ}[log π_θ(y) − log π_T(y)]`
辅以 `gap/mismatch_ratio`（teacher argmax ≠ student 采样 token 的逐 token 占比，ρ>1 的代理）。

- 这两个指标已在 `mix_trainer.py`（fit loop，union old_log_prob 之后）以**方法无关**的方式埋点：
  teacher worker 两条分支恒返回 `teacher_log_prob` / `teacher_predict_ids`，故 KDRL/RKL/TGPO 同口径。
- 每步都写 wandb（`gap/reverse_kl_k1`、`gap/mismatch_ratio`），并在 step≤1 打印 `[EXP2-GAP]` 便于 grep。
- **取 step-1 的值作为该 (student, teacher) 点的 x 坐标**（此时 student 尚未被任一方法更新，三条 run 的 x 应一致，可互相交叉验证）。

> 为什么用 reverse KL 而非 teacher 大小/家族标签：把「哪个 teacher」这一类别变量，换成连续、可比、且正是 OPD 所优化的那个量。x 轴自然单调，无需预设 teacher 顺序。

---

## Teacher 阶梯（student = Qwen2.5-Math-7B）

**硬约束：teacher 必须与 student 共享同一套 BPE 词表**（`data/align_tokenizer.py` 只做
vocab padding 到 `max(vocab)` + 统一特殊 token，不是跨 tokenizer 重映射）。
→ 阶梯锁在 **Qwen 系 tokenizer** 内；Llama/Mistral/Gemma 出局。

按预期 gap 从小到大（实际顺序以实测 x 为准）：

| # | Teacher | 与 student 关系 | 预期 gap | 备注 |
|---|---------|----------------|----------|------|
| 1 | Qwen2.5-Math-7B-Instruct | 同 base 同 size，短 CoT | 最小 | 近同分布 |
| 2 | Qwen2.5-7B-Instruct（或 Qwen2.5-Math-72B-Instruct） | 同族，通用/更大 | 小–中 | 风格/规模差 |
| 3 | DeepSeek-R1-Distill-Qwen-7B | Qwen2.5 底座，长 CoT 蒸馏 | 中–大 | 进入长思考分布 |
| 4 | DeepSeek-R1-Distill-Qwen-32B（或 QwQ-32B） | 更大长 CoT | 大 | 32B dense，param_offload |
| 5 | Qwen3-30B-A3B-Thinking-2507 | 新一代 + MoE + 重思考 | 最大 | 现有主 teacher |

- **最小可行 = 3 点**（#1, #3, #5）；推荐 4–5 点。
- 每个 teacher 需先用 `align_tokenizer.py` 对齐到 student 词表/特殊 token。
  **为保证 student 跨点逐字节一致**：把 student 一次性 pad 到整条阶梯的全局 max vocab，
  各 teacher 复用同一份 `student-aligned`；用脚本末尾打印的对齐诊断表核对 vocab/eos 一致。
- 32B/72B teacher 用 `teacher_ref.fsdp_config.param_offload=True` 放 CPU；注意单节点内存。

---

## 固定量 / 混杂控制（写进论文 setup）

跨所有点、所有方法严格固定：
- **同一份 prompt 集**（不要按 teacher 重新筛）。当前 `openr1.a3b_correct_35k` 是按 A3B 答对筛的，
  略偏向 A3B；可在附录用 teacher-中立 prompt 集（全量 OpenR1/MATH 子集）做鲁棒性复现。
- 同一 student 初始化、同样 batch/lr/steps/采样温度/熵系数。
- 三方法各用**自己标准的 recipe**（TGPO: coef=2e-3 + decay=1e-5 退火；KDRL/RKL 各自标准），
  且该 recipe 跨 gap 不变 —— 即「每个方法以其最佳设定」公平比。

**混杂杀手**：TGPO 用的是和 KDRL/OPD **完全相同的 teacher、相同的实测 gap**。
若 TGPO 在 OPD/KDRL 崩掉的高 gap 点仍提升，则说明 teacher 本身不是「差」，
差的是各方法对 gap 的处理 —— 这正是要论证的点。

---

## 纵轴与作图

- **主图**：best/final 平均准确率 vs gap，三条曲线（用论文 eval 套件 AIME24/25, AMC, MATH500, Minerva, Olympiad）。
- **凸显崩溃**：Δacc = final − base(student 起点)；OPD/KDRL 在大 gap 端转负，TGPO 近水平为正。
- **诊断小图（很有说服力）**：训练末期 reward vs gap、response length vs gap（OPD 的 length explosion）。
  reward/length 直接取各 run 的 wandb 末段；x 仍用 `gap/reverse_kl_k1`。

---

## Run 矩阵与算力

3 方法 × 5 teacher = **15 run**（7B headline）。
- OPD/KDRL 在大 gap 端早崩 → 可 `trainer.total_training_steps=150` 早停省算力；TGPO 跑满 300。
- 1.5B 作附录鲁棒性（再 15 run，预算够再加；本机不足以跑，需上集群）。

---

## 操作步骤

1. **对齐 teacher**：对阶梯里每个 teacher 跑 `data/align_tokenizer.py --student <Qwen2.5-Math-7B> --teacher <T>`，
   得到 `<T>-aligned`；核对诊断表 vocab/eos 与 student 一致。
2. **填 teacher 路径**：编辑 `run_gap_curve.sh` 顶部 `TEACHERS` 数组（tag→aligned 路径）。
3. **跑扫描**：`bash exp2_policy_gap_curve/run_gap_curve.sh`（单节点顺序跑；或自行拆分到多节点）。
4. **取 x 轴**：每个 run 日志 grep `[EXP2-GAP] step=1`，或读 wandb `gap/reverse_kl_k1`（step 1）。
   同一 teacher 三方法的 x 应一致 → 取均值/任一。
5. **取 y 轴**：各 run 的 eval 准确率（best/final）；诊断图取末段 reward/length。
6. **画图**：x=`gap/reverse_kl_k1`，三条曲线 + Δacc + 诊断小图。

---

## Caveats
- top-k 截断不影响本实验 x 轴（k1 用的是采样 token 的 full logp，非 top-k）。
- mismatch_ratio 与 reverse_kl_k1 通常同序但不完全单调；正式 x 轴用 reverse_kl_k1。
- 若某 teacher 的 reverse_kl_k1 与预期阶梯顺序不符，以实测为准重排 x（曲线本就按实测 x 画）。
- OPD/KDRL 早停点要在论文里说明（避免「跑得短所以差」的质疑）：报告 best-over-training，而非固定步。

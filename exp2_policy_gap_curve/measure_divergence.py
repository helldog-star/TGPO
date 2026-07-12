#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线量化师生「策略分歧」(回应 reviewer: teacher perplexity / token-KL / top-k overlap /
density-ratio / EOS-length distribution)。

设计要点(与 TGPO 训练里的 teacher-forcing 严格同口径):
  1. 用 STUDENT 的 tokenizer 统一构造 prompt 的 token-ids;三个模型都在「同一串 ids」上打分,
     因此要求它们共享同一套(aligned)词表 —— 与 mix_fsdp_worker 里 teacher 打分完全一致。
  2. student 先 rollout(温度=训练 rollout 温度, 默认 1.0), 得到回答 token 序列;
  3. 每个模型(含 student 自己)用 vLLM prompt_logprobs 对「同一串 prompt+response ids」做
     teacher-forcing 打分, 拿到逐 token 的 raw log-softmax(温度=1)logp + top-k;
  4. aggregate 汇总所有指标 + 按 prompt bootstrap 置信区间 + 出图。

三种模式(一个进程只加载一个模型, 由 run_divergence.sh 顺序调用):
  --mode generate  : 采样 rollout(student), 或独立生成测长度/EOS(任意模型)
  --mode score     : teacher-forcing 打分, 产出逐 token logp + top-k
  --mode aggregate : 读打分结果, 计算并汇总全部指标(不加载模型)

用法见 run_divergence.sh。
"""
import argparse
import json
import math
import os
import sys

import numpy as np


# ----------------------------- 公共工具 -----------------------------
def log(msg):
    print(f"[divergence] {msg}", flush=True)


def load_prompts_from_parquet(parquet_path, tokenizer, num_prompts, seed, max_prompt_len):
    """从训练/评测 parquet 读取 prompt(chat messages), 用 student tokenizer 转 ids。"""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "prompt" not in df.columns:
        raise ValueError(f"{parquet_path} 无 'prompt' 列; 现有列: {list(df.columns)}")
    # 固定随机种子抽样, 保证可复现
    rng = np.random.default_rng(seed)
    n = min(num_prompts, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    idx.sort()

    prompts = []
    for i in idx:
        row = df.iloc[int(i)]
        msgs = row["prompt"]
        if isinstance(msgs, np.ndarray):
            msgs = msgs.tolist()
        if isinstance(msgs, list) and len(msgs) > 0 and isinstance(msgs[0], dict):
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        else:
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": str(msgs)}], add_generation_prompt=True, tokenize=True
            )
        if len(ids) > max_prompt_len:
            # 过长 prompt 直接跳过(与训练 max_prompt_length 语义一致)
            continue
        data_source = row["data_source"] if "data_source" in df.columns else ""
        prompts.append({"pid": int(i), "data_source": str(data_source), "prompt_token_ids": list(map(int, ids))})
    log(f"从 {parquet_path} 抽取 {len(prompts)} 条 prompt(请求 {num_prompts}, 种子 {seed})")
    return prompts


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_llm(model_path, tp, gpu_mem, max_len, seed, dtype):
    from vllm import LLM

    log(f"加载模型: {model_path} (tp={tp}, gpu_mem={gpu_mem}, max_len={max_len}, dtype={dtype})")
    return LLM(
        model=model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_mem,
        dtype=dtype,
        max_model_len=max_len,
        trust_remote_code=True,
        seed=seed,
        enforce_eager=False,
    )


def as_token_prompts(id_lists):
    """把 token-id 列表包装成 vLLM 可接受的输入(兼容多版本)。"""
    try:
        from vllm.inputs import TokensPrompt

        return [TokensPrompt(prompt_token_ids=ids) for ids in id_lists]
    except Exception:
        return [{"prompt_token_ids": ids} for ids in id_lists]


# ----------------------------- generate -----------------------------
def mode_generate(args):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    prompt_tok = AutoTokenizer.from_pretrained(args.prompt_tokenizer or args.model, trust_remote_code=True)

    # prompts.jsonl: 全流程共享同一份 prompt(用 student tokenizer 构造)。第一个(student)run 负责生成它。
    prompts_file = os.path.join(args.out_dir, "prompts.jsonl")
    if os.path.exists(prompts_file):
        prompts = list(read_jsonl(prompts_file))
        log(f"复用已存在的 prompts: {prompts_file} ({len(prompts)} 条)")
    else:
        if not args.parquet:
            raise ValueError("首次运行需 --parquet 来构造 prompts.jsonl")
        prompts = load_prompts_from_parquet(
            args.parquet, prompt_tok, args.num_prompts, args.seed, args.max_prompt_len
        )
        os.makedirs(args.out_dir, exist_ok=True)
        with open(prompts_file, "w") as f:
            for p in prompts:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        log(f"已写 prompts: {prompts_file}")

    llm = build_llm(args.model, args.tp, args.gpu_mem, args.max_len, args.seed, args.dtype)
    sp = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        seed=args.seed,
    )
    inputs = as_token_prompts([p["prompt_token_ids"] for p in prompts])
    outs = llm.generate(inputs, sp)

    roll_path = os.path.join(args.out_dir, f"rollouts_{args.model_tag}.jsonl")
    lengths, finish = [], []
    with open(roll_path, "w") as f:
        for p, out in zip(prompts, outs):
            for s_i, comp in enumerate(out.outputs):
                tids = list(map(int, comp.token_ids))
                lengths.append(len(tids))
                finish.append(comp.finish_reason)
                f.write(json.dumps({
                    "pid": p["pid"], "sample": s_i,
                    "data_source": p.get("data_source", ""),
                    "prompt_token_ids": p["prompt_token_ids"],
                    "resp_token_ids": tids,
                    "resp_len": len(tids),
                    "finish_reason": comp.finish_reason,
                }, ensure_ascii=False) + "\n")
    log(f"已写 rollouts: {roll_path} ({len(lengths)} 条)")

    # 长度 / EOS 汇总(独立生成指标)
    lengths = np.array(lengths, dtype=float)
    n_trunc = sum(1 for r in finish if r == "length")
    summ = {
        "model_tag": args.model_tag, "model": args.model,
        "n_seq": int(len(lengths)),
        "len_mean": float(lengths.mean()) if len(lengths) else 0.0,
        "len_median": float(np.median(lengths)) if len(lengths) else 0.0,
        "len_p95": float(np.percentile(lengths, 95)) if len(lengths) else 0.0,
        "len_max": float(lengths.max()) if len(lengths) else 0.0,
        "trunc_rate": float(n_trunc / max(len(finish), 1)),   # finish_reason==length 的比例
        "stop_rate": float(sum(1 for r in finish if r == "stop") / max(len(finish), 1)),
        "temperature": args.temperature, "max_tokens": args.max_tokens,
        "length_hist": np.histogram(lengths, bins=40)[0].tolist() if len(lengths) else [],
        "length_hist_edges": np.histogram(lengths, bins=40)[1].tolist() if len(lengths) else [],
    }
    with open(os.path.join(args.out_dir, f"gensummary_{args.model_tag}.json"), "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    log(f"长度/EOS: mean={summ['len_mean']:.1f} median={summ['len_median']:.1f} "
        f"trunc={summ['trunc_rate']:.3f} stop={summ['stop_rate']:.3f}")


# ----------------------------- score -----------------------------
def mode_score(args):
    from vllm import SamplingParams

    # 打分对象 = student 的 rollouts(同一串 ids, 所有模型都打这份)
    roll_path = args.rollouts or os.path.join(args.out_dir, f"rollouts_{args.student_tag}.jsonl")
    rolls = list(read_jsonl(roll_path))
    log(f"待打分序列: {len(rolls)} 条 (来自 {roll_path})")

    # 词表对齐自检: 最大 token id 必须 < 打分模型 vocab_size, 否则 ids 语义错位
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab = getattr(cfg, "vocab_size", None)
    max_id = 0
    for r in rolls:
        if r["prompt_token_ids"]:
            max_id = max(max_id, max(r["prompt_token_ids"]))
        if r["resp_token_ids"]:
            max_id = max(max_id, max(r["resp_token_ids"]))
    if vocab is not None and max_id >= vocab:
        raise ValueError(
            f"词表不对齐! 序列里最大 token id={max_id} >= 模型 vocab_size={vocab}。"
            f" 请用 align_tokenizer.py 产出的 *-aligned 模型(共享 padded 词表)再打分。")
    log(f"词表自检 OK: max_token_id={max_id} < vocab_size={vocab}")

    llm = build_llm(args.model, args.tp, args.gpu_mem, args.max_len, args.seed, args.dtype)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)

    # 构造 full ids = prompt + response(可截断 response 以控显存/时长)
    full_ids, meta = [], []
    for r in rolls:
        resp = r["resp_token_ids"][: args.max_score_tokens]
        seq = r["prompt_token_ids"] + resp
        seq = seq[: args.max_len]                       # 不超模型窗口
        plen = min(len(r["prompt_token_ids"]), len(seq))
        full_ids.append(seq)
        meta.append({"pid": r["pid"], "sample": r["sample"],
                     "data_source": r.get("data_source", ""),
                     "prompt_len": plen, "resp_len": max(len(seq) - plen, 0)})

    outs = llm.generate(as_token_prompts(full_ids), sp)

    out_path = os.path.join(args.out_dir, f"scores_{args.model_tag}.jsonl")
    with open(out_path, "w") as f:
        for m, seq, out in zip(meta, full_ids, outs):
            plen, rlen = m["prompt_len"], m["resp_len"]
            pl = out.prompt_logprobs  # list[pos] -> {tid: Logprob} or None
            tok_logp, topk_ids, topk_lp = [], [], []
            for j in range(rlen):
                pos = plen + j            # 预测该位置 token 的分布存在 prompt_logprobs[pos]
                tid = seq[pos]
                d = pl[pos] if (pl is not None and pos < len(pl) and pl[pos] is not None) else None
                if d is None:
                    tok_logp.append(float("nan")); topk_ids.append([]); topk_lp.append([])
                    continue
                # 实际 token 的 logp
                lp_tid = d.get(tid, None)
                tok_logp.append(float(lp_tid.logprob) if lp_tid is not None else float("nan"))
                # top-k(按 logp 降序取前 topk)
                items = sorted(d.items(), key=lambda kv: kv[1].logprob, reverse=True)[: args.topk]
                topk_ids.append([int(k) for k, _ in items])
                topk_lp.append([float(v.logprob) for _, v in items])
            f.write(json.dumps({
                "pid": m["pid"], "sample": m["sample"], "data_source": m["data_source"],
                "resp_token_ids": seq[plen: plen + rlen],
                "tok_logp": tok_logp, "topk_ids": topk_ids, "topk_logp": topk_lp,
            }, ensure_ascii=False) + "\n")
    log(f"已写打分: {out_path}")


# ----------------------------- aggregate -----------------------------
def _softmax_from_logp(logps):
    a = np.array(logps, dtype=float)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()


def _trunc_kl(ids_p, lp_p, ids_q, lp_q, eps=1e-8):
    """top-k 截断 KL: 在两侧 top-k 的并集支撑上, 缺失质量用 eps 兜底。
    返回 (KL(P||Q), KL(Q||P)); 这里 P=teacher, Q=student(见调用处)。"""
    if not ids_p or not ids_q:
        return float("nan"), float("nan")
    pp = _softmax_from_logp(lp_p); qq = _softmax_from_logp(lp_q)
    P = dict(zip(ids_p, pp)); Q = dict(zip(ids_q, qq))
    union = set(ids_p) | set(ids_q)
    kl_pq = 0.0; kl_qp = 0.0
    for t in union:
        p = P.get(t, eps); q = Q.get(t, eps)
        kl_pq += p * math.log(p / q)
        kl_qp += q * math.log(q / p)
    return kl_pq, kl_qp


def _bootstrap_ci(per_prompt_vals, n_boot=1000, seed=0):
    """按 prompt bootstrap(token 在 prompt 内相关, 故聚合到 prompt 再重采样)。"""
    vals = np.array([v for v in per_prompt_vals if v is not None and not (isinstance(v, float) and math.isnan(v))])
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)]
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mode_aggregate(args):
    student_tag = args.student_tag
    teacher_tags = args.teacher_tags

    def scores_path(tag):
        return os.path.join(args.out_dir, f"scores_{tag}.jsonl")

    # 读 student 自打分(拿 student logp + top-k)
    stu = {}
    for r in read_jsonl(scores_path(student_tag)):
        stu[(r["pid"], r["sample"])] = r

    report = {"student": student_tag, "teachers": {}, "length_eos": {}}

    # 长度/EOS: 直接汇总各 gensummary
    for tag in [student_tag] + teacher_tags:
        gp = os.path.join(args.out_dir, f"gensummary_{tag}.json")
        if os.path.exists(gp):
            with open(gp) as f:
                report["length_eos"][tag] = json.load(f)

    for tt in teacher_tags:
        # 逐 token 累加, 按 prompt 聚合以便 bootstrap
        per_prompt = {}  # pid -> dict of lists
        n_tok = 0
        for r in read_jsonl(scores_path(tt)):
            key = (r["pid"], r["sample"])
            s = stu.get(key)
            if s is None:
                continue
            pid = r["pid"]
            pp = per_prompt.setdefault(pid, {"logrho": [], "tea_logp": [], "mismatch": [],
                                             "ov1": [], "ov5": [], "ov10": [], "fkl": [], "rkl": []})
            resp = r["resp_token_ids"]
            for j in range(len(resp)):
                lt = r["tok_logp"][j]        # teacher logp(a_t)
                ls = s["tok_logp"][j]        # student logp(a_t)
                if lt is None or ls is None or math.isnan(lt) or math.isnan(ls):
                    continue
                n_tok += 1
                logrho = ls - lt             # = reverse_kl_k1 被积项
                pp["logrho"].append(logrho)
                pp["tea_logp"].append(lt)
                # top-k(teacher vs student)
                t_ids, t_lp = r["topk_ids"][j], r["topk_logp"][j]
                s_ids, s_lp = s["topk_ids"][j], s["topk_logp"][j]
                a_t = resp[j]
                # mismatch(与训练同口径): teacher top-1 != student 采样 token
                pp["mismatch"].append(1.0 if (not t_ids or t_ids[0] != a_t) else 0.0)
                # top-k overlap
                for k, name in [(1, "ov1"), (5, "ov5"), (10, "ov10")]:
                    st = set(t_ids[:k]); ss = set(s_ids[:k])
                    pp[name].append(len(st & ss) / float(k) if st and ss else 0.0)
                # top-k 截断双向 KL(P=teacher, Q=student)
                kl_ts, kl_st = _trunc_kl(t_ids, t_lp, s_ids, s_lp)
                if not math.isnan(kl_ts):
                    pp["fkl"].append(kl_ts)   # forward KL D(teacher||student)
                    pp["rkl"].append(kl_st)   # reverse KL D(student||teacher)

        # prompt 级均值 -> bootstrap
        def per_prompt_metric(field, fn=np.mean):
            out = []
            for pid, d in per_prompt.items():
                if d[field]:
                    out.append(fn(d[field]))
            return out

        rows = {}
        rows["teacher_ppl"] = math.exp(-np.mean(sum([d["tea_logp"] for d in per_prompt.values()], [])))
        rev = per_prompt_metric("logrho")
        m, lo, hi = _bootstrap_ci(rev, seed=args.seed)
        rows["reverse_kl_k1"] = {"mean": m, "ci95": [lo, hi]}
        # density-ratio 统计
        allrho = np.array(sum([d["logrho"] for d in per_prompt.values()], []), dtype=float)
        rows["density_ratio"] = {
            "logrho_mean": float(allrho.mean()) if allrho.size else float("nan"),
            "logrho_std": float(allrho.std()) if allrho.size else float("nan"),
            "logrho_p5": float(np.percentile(allrho, 5)) if allrho.size else float("nan"),
            "logrho_p50": float(np.percentile(allrho, 50)) if allrho.size else float("nan"),
            "logrho_p95": float(np.percentile(allrho, 95)) if allrho.size else float("nan"),
            "frac_rho_gt1": float((allrho > 0).mean()) if allrho.size else float("nan"),  # student 比 teacher 更自信
        }
        for field, label in [("mismatch", "mismatch_ratio"), ("ov1", "top1_overlap"),
                             ("ov5", "top5_overlap"), ("ov10", "top10_overlap"),
                             ("fkl", "forward_kl_topk"), ("rkl", "reverse_kl_topk")]:
            mm, llo, hhi = _bootstrap_ci(per_prompt_metric(field), seed=args.seed)
            rows[label] = {"mean": mm, "ci95": [llo, hhi]}
        rows["_n_tokens"] = int(n_tok)
        rows["_n_prompts"] = int(len(per_prompt))
        report["teachers"][tt] = rows
        log(f"[{tt}] ppl={rows['teacher_ppl']:.3f} revKL_k1={rows['reverse_kl_k1']['mean']:.4f} "
            f"mismatch={rows['mismatch_ratio']['mean']:.3f} top1ov={rows['top1_overlap']['mean']:.3f} "
            f"(tokens={n_tok})")

    out_json = os.path.join(args.out_dir, "divergence_summary.json")
    with open(out_json, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"已写汇总: {out_json}")

    _write_markdown(report, os.path.join(args.out_dir, "divergence_summary.md"))
    if not args.no_plots:
        try:
            _make_plots(args.out_dir, report)
        except Exception as e:
            log(f"[warn] 出图失败(不影响数值): {e}")


def _fmt(x):
    if isinstance(x, dict) and "mean" in x:
        lo, hi = x["ci95"]
        return f"{x['mean']:.4f} [{lo:.4f},{hi:.4f}]"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _write_markdown(report, path):
    lines = [f"# 师生策略分歧 (student = {report['student']})", ""]
    tts = list(report["teachers"].keys())
    metrics = ["teacher_ppl", "reverse_kl_k1", "reverse_kl_topk", "forward_kl_topk",
               "mismatch_ratio", "top1_overlap", "top5_overlap", "top10_overlap"]
    lines.append("## Teacher-forcing 指标(在 student rollouts 上)")
    lines.append("| metric | " + " | ".join(tts) + " |")
    lines.append("|" + "---|" * (len(tts) + 1))
    for mtr in metrics:
        row = [mtr]
        for tt in tts:
            row.append(_fmt(report["teachers"][tt].get(mtr, "-")))
        lines.append("| " + " | ".join(row) + " |")
    # density-ratio
    lines += ["", "## Density-ratio  logρ = logπ_θ − logπ_T", "| stat | " + " | ".join(tts) + " |",
              "|" + "---|" * (len(tts) + 1)]
    for k in ["logrho_mean", "logrho_std", "logrho_p5", "logrho_p50", "logrho_p95", "frac_rho_gt1"]:
        row = [k] + [f"{report['teachers'][tt]['density_ratio'][k]:.4f}" for tt in tts]
        lines.append("| " + " | ".join(row) + " |")
    # length/eos
    if report["length_eos"]:
        lines += ["", "## 长度 / EOS(独立生成)", "| model | len_mean | len_median | len_p95 | trunc_rate | stop_rate |",
                  "|---|---|---|---|---|---|"]
        for tag, s in report["length_eos"].items():
            lines.append(f"| {tag} | {s['len_mean']:.1f} | {s['len_median']:.1f} | {s['len_p95']:.1f} | "
                         f"{s['trunc_rate']:.3f} | {s['stop_rate']:.3f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"已写 Markdown 表: {path}")


def _make_plots(out_dir, report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    tts = list(report["teachers"].keys())

    # 1) 关键指标条形图
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(tts, [report["teachers"][t]["reverse_kl_k1"]["mean"] for t in tts])
    ax[0].set_title("reverse_kl_k1 (student→teacher)"); ax[0].set_ylabel("nats/token")
    ax[1].bar(tts, [report["teachers"][t]["mismatch_ratio"]["mean"] for t in tts], color="#d97706")
    ax[1].set_title("mismatch_ratio (top-1 disagreement)")
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "divergence_bars.png"), dpi=140); plt.close(fig)

    # 2) 长度分布
    if report["length_eos"]:
        fig, a = plt.subplots(figsize=(7, 4))
        for tag, s in report["length_eos"].items():
            if s.get("length_hist"):
                edges = np.array(s["length_hist_edges"]); centers = (edges[:-1] + edges[1:]) / 2
                h = np.array(s["length_hist"], dtype=float); h = h / max(h.sum(), 1)
                a.plot(centers, h, label=tag)
        a.set_xlabel("response length (tokens)"); a.set_ylabel("density"); a.legend(); a.set_title("生成长度分布")
        fig.tight_layout(); fig.savefig(os.path.join(plot_dir, "length_dist.png"), dpi=140); plt.close(fig)
    log(f"已出图: {plot_dir}")


# ----------------------------- CLI -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="离线量化师生策略分歧")
    p.add_argument("--mode", required=True, choices=["generate", "score", "aggregate"])
    p.add_argument("--model", default="", help="模型路径(generate/score)")
    p.add_argument("--model-tag", default="", help="模型短标签(用于文件名)")
    p.add_argument("--out-dir", required=True)
    # 数据 / 采样
    p.add_argument("--parquet", default="", help="prompt 来源 parquet(首次 generate 必填)")
    p.add_argument("--prompt-tokenizer", default="", help="构造 prompt 用的 tokenizer(默认=student 模型); 保证三模型同一串 ids")
    p.add_argument("--num-prompts", type=int, default=256)
    p.add_argument("--n-samples", type=int, default=8, help="每 prompt rollout 数")
    p.add_argument("--temperature", type=float, default=1.0, help="rollout 温度(默认=训练 rollout 温度 1.0)")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    # 打分
    p.add_argument("--topk", type=int, default=20, help="prompt_logprobs 的 k(top-k overlap/截断 KL 用)")
    p.add_argument("--rollouts", default="", help="score: 待打分 rollouts.jsonl(默认=student 的)")
    p.add_argument("--max-score-tokens", type=int, default=4096, help="每条 response 打分的最大 token 数(控显存)")
    # 引擎
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-len", type=int, default=10240, help="vLLM max_model_len(需 >= prompt+response)")
    p.add_argument("--dtype", default="bfloat16")
    # aggregate
    p.add_argument("--student-tag", default="student")
    p.add_argument("--teacher-tags", nargs="*", default=[], help="aggregate: teacher 标签列表")
    p.add_argument("--no-plots", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "generate":
        if not args.model or not args.model_tag:
            sys.exit("generate 需 --model 与 --model-tag")
        mode_generate(args)
    elif args.mode == "score":
        if not args.model or not args.model_tag:
            sys.exit("score 需 --model 与 --model-tag")
        mode_score(args)
    else:
        if not args.teacher_tags:
            sys.exit("aggregate 需 --teacher-tags")
        mode_aggregate(args)


if __name__ == "__main__":
    main()

import numpy as np
import torch
from collections import defaultdict
import time

import verl.utils.torch_functional as verl_F

def compute_sft_pure_loss(log_prob, eos_mask):
    sft_losses = -log_prob
    sft_loss = verl_F.masked_mean(sft_losses, eos_mask)
    return sft_loss

def compute_grpo_outcome_advantage_split(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   on_policy_mask: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            # only include on-policy samples for mean and std calculation
            if on_policy_mask[i].item() is True:
                id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        # process std
        for idx in id2std:
            if id2std[idx].item() == 0:
                id2std[idx] = torch.tensor(1.0)
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores

def compute_token_on_off_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    cliprange, 
    clip_upper_bound,
    prefix_mask, 
    off_cliprange, 
    off_normalize=False, 
    off_abs_cliprange=None, 
    off_max_clip=None, 
    off_min_clip=None,
    all_max_clip=None, 
    off_policy_reshape="no_reshape", 
    off_policy_reshape_weight=1.0, 
    off_policy_reshape_pow_exp=0.5,
    on_policy_reshape="no_reshape", 
    on_policy_reshape_weight=1.0,
    on_policy_reshape_pow_exp=0.5,
    target_probs=None,
    loss_remove_token_mean=False,
    loss_remove_clip=False,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347
        prefix_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    # off-policy loss
    # compute off-policy probability
    
    negative_approx_kl = log_prob - old_log_prob
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    if on_policy_reshape == "no_reshape":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
    elif on_policy_reshape == "logp":
        ratio = log_prob - old_log_prob
    elif on_policy_reshape == "p_logp":
        ratio = torch.exp(negative_approx_kl) + on_policy_reshape_weight * negative_approx_kl
    elif on_policy_reshape == "square_root":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.sqrt(ratio)
    elif on_policy_reshape == "pow":
        ratio = torch.exp(negative_approx_kl) # [bsz, l]
        ratio = torch.pow(ratio, on_policy_reshape_pow_exp)
    elif on_policy_reshape == "p_div_p_0.1":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.1)
        f_old_prob = old_prob / (old_prob + 0.1)
        ratio = f_prob / f_old_prob
    elif on_policy_reshape == "p_div_p_0.5":
        prob = torch.exp(log_prob)
        old_prob = torch.exp(old_log_prob)
        f_prob = prob / (prob + 0.5)
        f_old_prob = old_prob / (old_prob + 0.5)
        ratio = f_prob / f_old_prob
    else:
        raise ValueError(f"Invalid on_policy_reshape: {on_policy_reshape}")

    on_pg_losses = -advantages * ratio
    upper_bound = max(clip_upper_bound, 1.0 + cliprange)
    if upper_bound == clip_upper_bound:
        print('clip upper bound is used: ', clip_upper_bound)

    if loss_remove_clip is False:
        on_pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, upper_bound)
        on_pg_clipfrac = verl_F.masked_mean(torch.gt(on_pg_losses2, on_pg_losses).float(), eos_mask)
        on_pg_losses = torch.max(on_pg_losses, on_pg_losses2)
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
    else:
        on_pg_loss = verl_F.masked_mean(on_pg_losses, (~prefix_mask) * eos_mask)
        on_pg_clipfrac = torch.tensor(0.0)
    
    # compute off-policy loss
    if target_probs is None:
        off_ratio = torch.exp(log_prob) # [bsz, l]
        if off_policy_reshape == "no_reshape":
            pass
        elif off_policy_reshape == "logp":
            off_ratio = log_prob * off_policy_reshape_weight
        elif off_policy_reshape == "p_logp":
            off_ratio = log_prob * off_policy_reshape_weight + off_ratio
        elif off_policy_reshape == "square_root":
            off_ratio = torch.sqrt(off_ratio)
        elif off_policy_reshape == "p_div_p_0.1":
            off_ratio = off_ratio / (off_ratio + 0.1)
        elif off_policy_reshape == "p_div_p_0.5":
            off_ratio = off_ratio / (off_ratio + 0.5)
        elif off_policy_reshape == "p_div_p_0.3":
            off_ratio = off_ratio / (off_ratio + 0.3)
        elif off_policy_reshape == "pow":
            off_ratio = torch.pow(off_ratio, off_policy_reshape_pow_exp)
        else:
            raise ValueError(f"Invalid off_policy_reshape: {off_policy_reshape}")
    else:
        assert target_probs.shape == log_prob.shape
        off_ratio = torch.exp(log_prob) / (target_probs+1e-6)
        # off_ratio[log_prob == 0] = 0
        off_ratio = off_ratio * prefix_mask
        # assert ((target_probs > 0) == prefix_mask).all()
        
    # clip off-policy ratio
    if off_max_clip is not None:
        off_ratio = torch.clamp(off_ratio, max=off_max_clip)
        off_ratio_max_clip_frac = verl_F.masked_mean((off_ratio == off_max_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_max_clip_frac = torch.tensor(0.0)
        
    if off_min_clip is not None:
        off_ratio = torch.clamp(off_ratio, min=off_min_clip)
        off_ratio_min_clip_frac = verl_F.masked_mean((off_ratio == off_min_clip).float(), prefix_mask * eos_mask)
    else:
        off_ratio_min_clip_frac = torch.tensor(0.0)

    off_ratio_mean = verl_F.masked_mean(off_ratio, prefix_mask * eos_mask)
    if off_ratio_mean.isnan().any().item():
        off_ratio_mean = torch.tensor(0.0)

    off_pg_losses = -advantages * off_ratio
    off_pg_loss = verl_F.masked_mean(off_pg_losses, prefix_mask * eos_mask)
    if off_pg_loss.isnan().item() is True:
        off_pg_loss = torch.tensor(0.0)
    off_pg_clipfrac = torch.tensor(0.0)
    
    prefix_mask = prefix_mask.float()
    pg_losses = off_pg_losses * prefix_mask + on_pg_losses * (1 - prefix_mask)
    
    # log on/off probs
    off_policy_probs = torch.exp(log_prob)
    off_policy_prob = verl_F.masked_mean(off_policy_probs, prefix_mask * eos_mask)
    if off_policy_prob.isnan().item() is True:
        off_policy_prob = torch.tensor(0.0)
    on_policy_probs = torch.exp(old_log_prob)
    on_policy_prob = verl_F.masked_mean(on_policy_probs, (1.0-prefix_mask) * eos_mask)
    if on_policy_prob.isnan().item() is True:
        on_policy_prob = torch.tensor(0.0)
            
    if all_max_clip is not None:
        p_on = torch.exp(log_prob)
        p_on_mask = (p_on <= all_max_clip).float()
        eos_mask = eos_mask * p_on_mask
        pg_losses = pg_losses * p_on_mask
        
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    return {
        "pg_loss": pg_loss,
        "off_pg_loss": off_pg_loss,
        "on_pg_loss": on_pg_loss,
        "off_pg_clipfrac": off_pg_clipfrac,
        "on_pg_clipfrac": on_pg_clipfrac,
        "ppo_kl": ppo_kl,
        "off_policy_prob": off_policy_prob,
        "on_policy_prob": on_policy_prob,
        "off_ratio_mean": off_ratio_mean,
        "off_ratio_max_clip_frac": off_ratio_max_clip_frac,
        "off_ratio_min_clip_frac": off_ratio_min_clip_frac,
    }

# tgpo
def compute_token_on_tipo_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    eos_mask: torch.Tensor,
    teacher_ids_log_probs: torch.Tensor,
    cliprange: float,
    teacher_coef: float = 0.1,
    teacher_topk_logits: torch.Tensor | None = None,
    student_teacher_topk_log_probs: torch.Tensor | None = None,
    loss_remove_clip: bool = False,
    loss_remove_token_mean: bool = False,
    kl_direction: str = "forward",
    reg_only: bool = False,
):
    """
    PPO Loss with Teacher Regularization (差分正则)

    Args:
        old_log_prob: (bs, response_length) - 旧策略的log概率
        log_prob: (bs, response_length) - 当前策略的log概率
        advantages: (bs, response_length) - 优势函数
        eos_mask: (bs, response_length) - 有效token的mask
        teacher_ids_log_probs: (bs, response_length) - student对teacher预测token的log概率
        cliprange: PPO裁剪范围
        teacher_coef: 正则化系数 λ / w
        kl_direction: "forward" = TGPO 的 forward KL / CE 引导 (默认);
                      "reverse" = 分布级 reverse KL D_KL(π_θ||π_T) (在 teacher top-k 上,
                      student 在该支撑上重归一化), 用作论文 §2 的 "RKL 进 regularization" 对照。
                      reverse 要求 teacher_topk_logits 与 student_teacher_topk_log_probs 非空。
        reg_only: True 时目标只保留 teacher 正则项, 丢掉 GRPO 的 pg_loss
                  (用于 §2 纯 RKL: 不掺 RLVR 结果奖励)。

    Returns:
        total_loss, pg_loss, teacher_reg_loss, pg_clipfrac, ppo_kl
    """

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    if loss_remove_clip is False:
        pg_losses = torch.max(pg_losses, pg_losses2)

    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)

    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    if kl_direction == "reverse":
        # 分布级 reverse KL: D_KL(π_θ || π_T) = Σ_v π_θ(v)[logπ_θ(v) - logπ_T(v)], 在 teacher top-k 上。
        # 解析可微 (梯度穿过 student 的 softmax 权重 π_θ(v)), 不走 REINFORCE, 无采样方差, 无 “+1”。
        assert teacher_topk_logits is not None and student_teacher_topk_log_probs is not None, \
            "kl_direction='reverse' 需要 teacher_topk_logits 与 student_teacher_topk_log_probs (请开 use_tipo_topk_kl)"
        teacher_log_probs_topk = torch.log_softmax(teacher_topk_logits.float(), dim=-1)  # teacher 在其 top-k 上的分布 (detach)
        # student_teacher_topk_log_probs 是 student 全词表 log-softmax 在 teacher top-k id 上的取值,
        # reverse KL 需 student 在该 top-k 支撑上的(可微)归一化分布:
        student_logp_topk = student_teacher_topk_log_probs.float()
        student_logp_norm = student_logp_topk - torch.logsumexp(student_logp_topk, dim=-1, keepdim=True)
        student_probs_norm = torch.exp(student_logp_norm)
        rev_kl_token = torch.sum(
            student_probs_norm * (student_logp_norm - teacher_log_probs_topk),
            dim=-1,
        )
        if loss_remove_token_mean is True:
            teacher_reg_loss = (rev_kl_token * eos_mask).sum() / eos_mask.shape[-1]
        else:
            teacher_reg_loss = verl_F.masked_mean(rev_kl_token, eos_mask)
    else:
        # 默认 teacher 正则：student forcing teacher token 的 CE（硬标签, forward KL / 引导）。
        teacher_reg_loss = -verl_F.masked_mean(teacher_ids_log_probs, eos_mask)
        # 可选：在 teacher top-k 分布上做 forward KL: KL(teacher || student)。
        if teacher_topk_logits is not None and student_teacher_topk_log_probs is not None:
            teacher_log_probs_topk = torch.log_softmax(teacher_topk_logits.float(), dim=-1)
            teacher_probs_topk = torch.exp(teacher_log_probs_topk)
            fwd_kl_token = torch.sum(
                teacher_probs_topk * (teacher_log_probs_topk - student_teacher_topk_log_probs.float()),
                dim=-1,
            )
            if loss_remove_token_mean is True:
                teacher_reg_loss = (fwd_kl_token * eos_mask).sum() / eos_mask.shape[-1]
            else:
                teacher_reg_loss = verl_F.masked_mean(fwd_kl_token, eos_mask)

    if reg_only:
        # §2 纯 RKL/纯正则: 目标不含 GRPO pg_loss (pg_loss 仍返回, 仅作日志诊断)。
        total_loss = teacher_coef * teacher_reg_loss
    else:
        total_loss = pg_loss + teacher_coef * teacher_reg_loss

    return total_loss, pg_loss, teacher_reg_loss, pg_clipfrac, ppo_kl


# kdrl
def compute_token_on_kdrl_loss(
    old_log_prob: torch.Tensor,      # π_θ_old 的 log prob
    log_prob: torch.Tensor,          # π_θ 的 log prob (当前策略)
    advantages: torch.Tensor,        # 优势函数
    eos_mask: torch.Tensor,          # token mask
    teacher_log_prob: torch.Tensor,  # π_T 的 log prob (teacher)
    cliprange: float,                # PPO clip 范围
    teacher_coef: float = 0.002,     # β 系数
    loss_remove_clip: bool = False,
    loss_remove_token_mean: bool = False
):
    """
    KDRL Loss: J_KDRL(θ) = J_GRPO(θ) - β * D_KL^k2(π_θ || π_T)
    """
    
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    if loss_remove_clip is False:
        pg_losses = torch.max(pg_losses, pg_losses2)
    
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
        print(f'no token mean: mean normalization {eos_mask.shape[-1]}')
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    
    # 计算 R_i,t(θ) = log[π_T / π_θ]
    R_theta = teacher_log_prob - log_prob  # shape: [batch, seq_len]
    
    # 计算 D_KL^k2(π_θ || π_T)
    R_theta_squared = R_theta ** 2  # R_i,t(θ)^2
    kd_token_loss = 0.5 * R_theta_squared # 1/2 * R_i,t(θ)^2
    if loss_remove_token_mean is True:
        teacher_reg_loss = (kd_token_loss * eos_mask).sum() / eos_mask.shape[-1]
    else:
        teacher_reg_loss = verl_F.masked_mean(kd_token_loss, eos_mask)
    
    # J_KDRL(θ) = J_GRPO(θ) - β * D_KL^k2(π_θ || π_T)
    total_loss = pg_loss + teacher_coef * teacher_reg_loss

    return total_loss, pg_loss, teacher_reg_loss, pg_clipfrac, ppo_kl


# rkl_reg (k1 / score-function): 纯 reverse KL 作为可微 loss, 不含 GRPO 结果奖励 (pg_loss)。
# 用途 = 论文 §2 对照(分布级 RKL-Reg 的姊妹版): 标准 RKL(OP Distill) 把 reverse KL 放进
# advantage(REINFORCE), 这里把同一个 score-function 估计写成可微 loss 项。无 “+1” 直接项(其期望恒为0),
# 权重 stop_grad(logρ) detach。二者都不掺 RLVR, 论证 reverse KL 无论放哪在 cross-family 都崩。
def compute_token_on_rkl_reg_loss(
    old_log_prob: torch.Tensor,      # π_θ_old 的 log prob (仅用于日志 ppo_kl / pg_loss 诊断)
    log_prob: torch.Tensor,          # π_θ 的 log prob (当前策略, 带梯度)
    advantages: torch.Tensor,        # 优势函数 (仅用于诊断性 pg_loss, 不进入目标)
    eos_mask: torch.Tensor,          # token mask
    teacher_log_prob: torch.Tensor,  # π_T 的 log prob (teacher 对 student 采样 token 的 logp, 无梯度)
    cliprange: float,                # PPO clip 范围 (仅诊断)
    teacher_coef: float = 1.0,       # w 系数; 纯 RKL 默认 1.0 (与标准 RKL 的 advantage 同尺度)
    loss_remove_clip: bool = False,
    loss_remove_token_mean: bool = False
):
    """
    纯 RKL-Reg Loss (k1 / score-function surrogate): 目标 J(θ) = w * D_KL(π_θ || π_T), 无 GRPO 项。

    on-policy reverse KL 的梯度 = E[∇logπ_θ · logρ] (score-function; logρ = logp - teacher_logp)。
    其可微 surrogate(梯度相等) = mean( stop_grad(logρ) · logπ_θ ):
      - 权重 stop_grad(logρ) 必须 detach, 否则会多出 logp·∇logp 杂项。
      - 不含 “+1” 直接项: E_πθ[∇logπ_θ] = ∇∫π_θ = 0, 常数 baseline 不改期望梯度, 故丢弃。
    注意: teacher_reg_loss 的“数值”无 KL 含义(只是 surrogate), 仅其“梯度”= ∇(reverse KL)。
    与标准 RKL(OP Distill) 几乎同一估计量(后者把 logρ 当 advantage 走 -A·ratio); 区别仅在
    logρ 用 current_logp(此处)还是 old_logp(OP Distill) + 无 GRPO 组归一化/clip。
    与分布级 RKL-Reg(tipo_loss kl_direction=reverse) 互为姊妹: 那个是全分布解析可微、无采样方差。
    """

    # 以下 pg_loss / ppo_kl / pg_clipfrac 仅用于 wandb 诊断, 不进入梯度目标。
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    if loss_remove_clip is False:
        pg_losses = torch.max(pg_losses, pg_losses2)
    if loss_remove_token_mean is True:
        pg_loss = (pg_losses * eos_mask).sum() / eos_mask.shape[-1]
    else:
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    # score-function surrogate: weight = stop_grad(logρ), 仅 log_prob 带梯度, 无 “+1”。
    R_theta = log_prob - teacher_log_prob          # logρ (含梯度, 但下面只取其 detach 做权重)
    sf_token = R_theta.detach() * log_prob          # ∇(sf_token) = logρ · ∇logπ_θ = ∇(reverse KL)
    if loss_remove_token_mean is True:
        teacher_reg_loss = (sf_token * eos_mask).sum() / eos_mask.shape[-1]
    else:
        teacher_reg_loss = verl_F.masked_mean(sf_token, eos_mask)

    # 纯 RKL: 目标里不含 pg_loss(GRPO)。外层最小化 total_loss ⟺ 梯度下降 reverse KL。
    total_loss = teacher_coef * teacher_reg_loss

    return total_loss, pg_loss, teacher_reg_loss, pg_clipfrac, ppo_kl


# rkl
def compute_rkl_advantage(
    old_log_probs: torch.Tensor,
    teacher_log_prob: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        old_log_probs: `(torch.Tensor)`
            shape is (bs, response_length)
        teacher_log_prob: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(np.ndarray)`

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """


    scores = (teacher_log_prob - old_log_probs) * response_mask

    return scores, scores


def compute_rkl_topk_advantage(
    student_topk_ids: torch.Tensor,
    student_topk_logits: torch.Tensor,
    teacher_topk_ids: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    response_mask: torch.Tensor,
    chunk_size: int = 8192,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute distribution-level top-k reverse KL approximation:
      RKL_topk ~= KL(student || teacher) on intersect(topk_teacher, topk_student)
    and return advantage = -RKL_topk.
    """
    bsz, response_length, _ = student_topk_ids.shape
    t0 = time.perf_counter()
    flat_scores = torch.zeros(
        bsz * response_length,
        dtype=student_topk_logits.dtype,
        device=student_topk_logits.device,
    )
    valid_token_idx = torch.nonzero(response_mask.reshape(-1) > 0, as_tuple=False).squeeze(-1)
    if valid_token_idx.numel() == 0:
        scores = flat_scores.view(bsz, response_length) * response_mask
        return scores, scores

    flat_student_ids = student_topk_ids.reshape(bsz * response_length, -1)
    flat_student_logits = student_topk_logits.reshape(bsz * response_length, -1)
    flat_teacher_ids = teacher_topk_ids.reshape(bsz * response_length, -1)
    flat_teacher_logits = teacher_topk_logits.reshape(bsz * response_length, -1)
    total_pairs = 0
    matched_pairs = 0
    total_rows = 0
    matched_rows = 0

    for start in range(0, valid_token_idx.numel(), chunk_size):
        idx = valid_token_idx[start:start + chunk_size]
        student_ids = flat_student_ids[idx].long()
        student_logits = flat_student_logits[idx].float()
        teacher_ids = flat_teacher_ids[idx].long()
        teacher_logits = flat_teacher_logits[idx].float()
        chunk_scores = torch.zeros(student_ids.size(0), dtype=student_logits.dtype, device=student_logits.device)

        # Faster than materializing [chunk, k, k] boolean match tensor:
        # sort student ids then locate teacher ids by batched binary search.
        k_student = student_ids.size(1)
        student_sorted_ids, student_sort_idx = torch.sort(student_ids, dim=1)
        student_sorted_logits = torch.gather(student_logits, dim=1, index=student_sort_idx)

        search_pos = torch.searchsorted(student_sorted_ids, teacher_ids, right=False)
        in_bound = search_pos < k_student
        safe_pos = search_pos.clamp(max=k_student - 1)
        matched_student_ids = torch.gather(student_sorted_ids, dim=1, index=safe_pos)
        has_match = in_bound & (matched_student_ids == teacher_ids)
        row_has_common = has_match.any(dim=1)
        matched_pairs += int(has_match.sum().item())
        total_pairs += int(has_match.numel())
        matched_rows += int(row_has_common.sum().item())
        total_rows += int(row_has_common.numel())
        if row_has_common.any():
            aligned_student_logits = torch.gather(student_sorted_logits, dim=1, index=safe_pos)

            neg_inf = torch.finfo(student_logits.dtype).min
            teacher_common_logits = teacher_logits.masked_fill(~has_match, neg_inf)
            student_common_logits = aligned_student_logits.masked_fill(~has_match, neg_inf)

            teacher_common_logits = teacher_common_logits[row_has_common]
            student_common_logits = student_common_logits[row_has_common]

            teacher_logp = teacher_common_logits - torch.logsumexp(teacher_common_logits, dim=1, keepdim=True)
            student_logp = student_common_logits - torch.logsumexp(student_common_logits, dim=1, keepdim=True)
            student_prob = torch.exp(student_logp)
            rkl = (student_prob * (student_logp - teacher_logp)).sum(dim=1)
            # advantage = -RKL so that maximizing advantage minimizes RKL.
            chunk_scores[row_has_common] = -rkl

        flat_scores[idx] = chunk_scores.to(flat_scores.dtype)

    scores = flat_scores.view(bsz, response_length) * response_mask
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pair_coverage = matched_pairs / max(total_pairs, 1)
    row_coverage = matched_rows / max(total_rows, 1)
    print(
        f"[rkl_topk] time_ms={elapsed_ms:.1f} "
        f"pair_coverage={pair_coverage:.3f} "
        f"token_coverage={row_coverage:.3f} "
        f"valid_tokens={int(valid_token_idx.numel())} chunk_size={chunk_size}"
    )
    return scores, scores

# 所有tok grpo adv + teacher rkl [👎]
def compute_grpo_merge_rkl_advantage(token_level_rewards: torch.Tensor,
                                    eos_mask: torch.Tensor,
                                    index: torch.Tensor,
                                    old_log_probs: torch.Tensor,
                                    teacher_log_prob: torch.Tensor,
                                    epsilon: float = 1e-6,
                                    use_std: bool = True):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        reverse_kl = (teacher_log_prob - old_log_probs) * eos_mask
        teacher_coef = 0.002
        scores = scores + teacher_coef * reverse_kl

    return scores, scores

# 所有tok grpo adv + stuforce teacher ce
def compute_tipo_advantage(token_level_rewards: torch.Tensor,
                            entropys: torch.Tensor,
                            eos_mask: torch.Tensor,
                            index: torch.Tensor,
                            teacher_predict_ids: torch.Tensor,
                            student_predict_ids: torch.Tensor,
                            teacher_ids_log_probs: torch.Tensor,  # (bs, seq_len)
                            epsilon: float = 1e-6,
                            teacher_coef: float = 0.002,
                            use_std: bool = True):
    
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    # 1. 保存原始分数的统计信息 (Raw Score Stats)
    raw_scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    raw_score_mean = raw_scores.mean().item()
    
    scores = raw_scores.clone() # 复制一份用于归一化计算

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        
        for i in range(bsz):
            if use_std:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = (scores[i] - id2mean[index[i]])
        
        # 扩展到序列长度
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

        # ===== 统计信息计算 =====
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0
        total_valid_tokens = eos_mask.sum() + epsilon
        
        # 1. Mismatch Ratio
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens

        # ===== 计算 Teacher Signal =====
        teacher_signal = torch.exp(teacher_ids_log_probs) 
        weighted_teacher_signal = teacher_coef * teacher_signal

        # 2. 信号强度对比
        # 计算 Outcome Reward 的平均绝对值强度 (只看有效token)
        outcome_magnitude = scores[eos_mask.bool()].abs().mean().item()
        # 计算 Weighted Teacher Signal 的平均绝对值强度
        teacher_magnitude = weighted_teacher_signal[eos_mask.bool()].abs().mean().item()
        
        # 3. Teacher Probability 未加权均值，代表模型认为 Teacher Token 是对的概率有多大
        avg_teacher_prob = teacher_signal[eos_mask.bool()].mean().item()

        # Raw_S: 原始分数均值 (监控做题能力)
        # T_Prob: 模型预测Teacher Token的平均概率 (监控模仿能力)
        # Mag_Out/Mag_T: 两种奖励信号的强度对比 (监控权重平衡)
        print(f"[TIPO] Raw_S: {raw_score_mean:.2f} | Ent: {avg_all_entropy:.3f} | Diff: {mismatch_ratio:.1%} | "
              f"T_Prob: {avg_teacher_prob:.3f} | "
              f"Mag(Out/Tea): {outcome_magnitude:.3f}/{teacher_magnitude:.3f}")

        scores = scores + weighted_teacher_signal

    return scores, scores

# 所有tok均仅采用student forcing teacher ce [👎]
def compute_opsft_advantage(entropys: torch.Tensor,
                        eos_mask: torch.Tensor,
                        teacher_predict_ids: torch.Tensor,
                        student_predict_ids: torch.Tensor,
                        teacher_ids_log_probs: torch.Tensor,  # (bs, seq_len)
                        epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        
        # 计算 CE Loss (Loss通常为正数: -log_probs)
        ce_loss = -teacher_ids_log_probs 

        # ===== 统计信息打印 (Statistics Logging) =====
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0

        total_valid_tokens = eos_mask.sum() + epsilon
        
        # Teacher/Student 不一致比例 (Mismatch Ratio)
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens
        
        # 不一致位置的熵统计 (Diff Ent)
        diff_entropys = entropys[is_diff.bool()]
        if diff_entropys.numel() > 0:
            diff_ent_min = diff_entropys.min().item()
            diff_ent_mean = diff_entropys.mean().item()
            diff_ent_max = diff_entropys.max().item()
        else:
            diff_ent_min = diff_ent_mean = diff_ent_max = 0.0

        # 打印请求的信息
        print(f"[TIPO-Entro] Ent_Mean: {avg_all_entropy:.4f} | Diff_Ratio: {mismatch_ratio:.1%} | Diff_Ent(Min/Avg/Max): {diff_ent_min:.3f}/{diff_ent_mean:.3f}/{diff_ent_max:.3f}")

        # ===== 仅采用teacher监督 =====
        teacher_signal = -ce_loss
        scores = teacher_signal

    return scores, scores

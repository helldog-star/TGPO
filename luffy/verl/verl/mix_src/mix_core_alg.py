import numpy as np
import torch
from collections import defaultdict

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
    loss_remove_clip: bool = False,
    loss_remove_token_mean: bool = False
):
    """
    PPO Loss with Teacher Regularization
    
    Args:
        old_log_prob: (bs, response_length) - 旧策略的log概率
        log_prob: (bs, response_length) - 当前策略的log概率
        advantages: (bs, response_length) - 优势函数
        eos_mask: (bs, response_length) - 有效token的mask
        teacher_ids_log_probs: (bs, response_length) - student对teacher预测token的log概率
        cliprange: PPO裁剪范围
        teacher_coef: 正则化系数 λ
    
    Returns:
        total_loss: 总损失
        pg_loss: PPO策略梯度损失
        teacher_reg_loss: Teacher正则化损失
        pg_clipfrac: PPO裁剪比例
        ppo_kl: PPO KL散度
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
    
    # 最大化student对teacher预测token的log概率，等价于最小化负log概率（负对数似然）
    teacher_reg_loss = -verl_F.masked_mean(teacher_ids_log_probs, eos_mask)
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
    total_loss = pg_loss - teacher_coef * teacher_reg_loss
    
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

# 高熵tok grpo adv + 低熵tok stuforce teacher ce
def compute_tipo_mix_advantage(token_level_rewards: torch.Tensor,
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
    raw_scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    raw_score_mean = raw_scores.mean().item()

    scores = raw_scores.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    entropy_quantile =  0.8

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

        # 提取整个 batch 中所有有效的 entropy 值 
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0
        total_valid_tokens = eos_mask.sum() + epsilon

        if valid_all_entropys.numel() > 0:
            # 计算全局阈值 
            global_threshold = torch.quantile(valid_all_entropys, entropy_quantile)
        else:
            global_threshold = torch.tensor(0.0, device=entropys.device)

        high_entropy_mask = (entropys > global_threshold).float() * eos_mask
        low_entropy_mask = (entropys <= global_threshold).float() * eos_mask

        # ===== 基于相对位置的高熵 Mask 分布分析 =====
        # 1. 计算每个 token 的绝对位置索引 (0, 1, 2, ...)
        seq_len = entropys.shape[1]
        token_indices = torch.arange(seq_len, device=entropys.device).unsqueeze(0).expand(bsz, -1)
        
        # 2. 获取每个样本的真实长度
        # 假设 eos_mask 是 (bs, seq_len)，1为有效，0为padding
        sample_lengths = eos_mask.sum(dim=1, keepdim=True) # (bs, 1)
        
        # 3. 计算相对位置 (0.0 ~ 1.0)
        # 避免除以0 (虽然理论上 mask 为 1 的地方 length 肯定 > 0)
        relative_positions = token_indices.float() / (sample_lengths.float() + 1e-6)
        
        # 4. 只提取有效 token 的数据
        valid_mask_bool = eos_mask.bool()
        flat_relative_pos = relative_positions[valid_mask_bool] # (Total_Valid_Tokens,)
        flat_high_ent_mask = high_entropy_mask[valid_mask_bool] # (Total_Valid_Tokens,)
        
        # 5. 分桶统计 (5 Bins: 0-20%, 20-40%, ...)
        num_bins = 5
        bin_results = []
        
        for i in range(num_bins):
            # 定义桶的范围
            lower = i / num_bins
            upper = (i + 1) / num_bins
            
            # 找到落在该进度区间内的 token
            if i == num_bins - 1:
                # 最后一个桶包含 1.0
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos <= upper)
            else:
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos < upper)
            
            if in_bin.sum() > 0:
                # 计算该进度区间内，高熵 mask 的比例
                avg_ratio = flat_high_ent_mask[in_bin].mean().item()
                bin_results.append(f"{avg_ratio:.1%}")
            else:
                bin_results.append("-")
        
        print(f"[Ent-Dist-Rel] (0% -> 100%): {' -> '.join(bin_results)}")

        teacher_signal = torch.exp(teacher_ids_log_probs)
        
        weighted_teacher_signal = teacher_coef * teacher_signal * low_entropy_mask

        # 统计信息
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens

        # 整个序列的平均 Outcome Reward 强度
        outcome_magnitude = scores[eos_mask.bool()].abs().mean().item()
        
        # Teacher Magnitude: 低熵区Teacher 信号的平均强度
        valid_low_ent_count = low_entropy_mask.sum() + epsilon
        teacher_magnitude = weighted_teacher_signal.sum() / valid_low_ent_count
        
        # Teacher Prob: 模型在低熵区域对 Teacher Token 的平均置信度
        avg_teacher_prob = (teacher_signal * low_entropy_mask).sum() / valid_low_ent_count

        print(f"[TIPO-Mix] Raw_S: {raw_score_mean:.2f} | Ent: {avg_all_entropy:.3f} | Diff: {mismatch_ratio:.1%} | "
              f"T_Prob(LowEnt): {avg_teacher_prob:.3f} | "
              f"Mag(Out/Tea): {outcome_magnitude:.3f}/{teacher_magnitude.item():.3f}")

        scores = scores * high_entropy_mask + weighted_teacher_signal

    return scores, scores

# 所有tok grpo adv + 高熵tok stuforce teacher ce
def compute_tipo_high_advantage(token_level_rewards: torch.Tensor,
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
    raw_scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    raw_score_mean = raw_scores.mean().item()

    scores = raw_scores.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    entropy_quantile =  0.8

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

        # 提取整个 batch 中所有有效的 entropy 值 
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0
        total_valid_tokens = eos_mask.sum() + epsilon

        if valid_all_entropys.numel() > 0:
            # 计算全局阈值 
            global_threshold = torch.quantile(valid_all_entropys, entropy_quantile)
        else:
            global_threshold = torch.tensor(0.0, device=entropys.device)

        high_entropy_mask = (entropys > global_threshold).float() * eos_mask
        # low_entropy_mask = (entropys <= global_threshold).float() * eos_mask

        # ===== 基于相对位置的高熵 Mask 分布分析 =====
        # 1. 计算每个 token 的绝对位置索引 (0, 1, 2, ...)
        seq_len = entropys.shape[1]
        token_indices = torch.arange(seq_len, device=entropys.device).unsqueeze(0).expand(bsz, -1)
        
        # 2. 获取每个样本的真实长度
        # 假设 eos_mask 是 (bs, seq_len)，1为有效，0为padding
        sample_lengths = eos_mask.sum(dim=1, keepdim=True) # (bs, 1)
        
        # 3. 计算相对位置 (0.0 ~ 1.0)
        # 避免除以0 (虽然理论上 mask 为 1 的地方 length 肯定 > 0)
        relative_positions = token_indices.float() / (sample_lengths.float() + 1e-6)
        
        # 4. 只提取有效 token 的数据
        valid_mask_bool = eos_mask.bool()
        flat_relative_pos = relative_positions[valid_mask_bool] # (Total_Valid_Tokens,)
        flat_high_ent_mask = high_entropy_mask[valid_mask_bool] # (Total_Valid_Tokens,)
        
        # 5. 分桶统计 (5 Bins: 0-20%, 20-40%, ...)
        num_bins = 5
        bin_results = []
        
        for i in range(num_bins):
            # 定义桶的范围
            lower = i / num_bins
            upper = (i + 1) / num_bins
            
            # 找到落在该进度区间内的 token
            if i == num_bins - 1:
                # 最后一个桶包含 1.0
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos <= upper)
            else:
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos < upper)
            
            if in_bin.sum() > 0:
                # 计算该进度区间内，高熵 mask 的比例
                avg_ratio = flat_high_ent_mask[in_bin].mean().item()
                bin_results.append(f"{avg_ratio:.1%}")
            else:
                bin_results.append("-")
        
        print(f"[Ent-Dist-Rel] (0% -> 100%): {' -> '.join(bin_results)}")

        teacher_signal = torch.exp(teacher_ids_log_probs)
        
        weighted_teacher_signal = teacher_coef * teacher_signal * high_entropy_mask

        # 统计信息
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens

        # 整个序列的平均 Outcome Reward 强度
        outcome_magnitude = scores[eos_mask.bool()].abs().mean().item()
        
        # Teacher Magnitude: 高熵区Teacher 信号的平均强度
        valid_high_ent_count = high_entropy_mask.sum() + epsilon
        teacher_magnitude = weighted_teacher_signal.sum() / valid_high_ent_count
        
        # Teacher Prob: 模型在高熵区域对 Teacher Token 的平均置信度
        avg_teacher_prob = (teacher_signal * high_entropy_mask).sum() / valid_high_ent_count

        print(f"[TIPO-Mix] Raw_S: {raw_score_mean:.2f} | Ent: {avg_all_entropy:.3f} | Diff: {mismatch_ratio:.1%} | "
              f"T_Prob(HighEnt): {avg_teacher_prob:.3f} | "
              f"Mag(Out/Tea): {outcome_magnitude:.3f}/{teacher_magnitude.item():.3f}")

        scores = scores + weighted_teacher_signal

    return scores, scores


# 高熵tok grpo adv + 高熵tok stuforce teacher ce
def compute_tipo_high_both_advantage(token_level_rewards: torch.Tensor,
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
    raw_scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    raw_score_mean = raw_scores.mean().item()

    scores = raw_scores.clone()
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    entropy_quantile =  0.8

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

        # 提取整个 batch 中所有有效的 entropy 值 
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0
        total_valid_tokens = eos_mask.sum() + epsilon

        if valid_all_entropys.numel() > 0:
            # 计算全局阈值 
            global_threshold = torch.quantile(valid_all_entropys, entropy_quantile)
        else:
            global_threshold = torch.tensor(0.0, device=entropys.device)

        high_entropy_mask = (entropys > global_threshold).float() * eos_mask
        # low_entropy_mask = (entropys <= global_threshold).float() * eos_mask

        # ===== 基于相对位置的高熵 Mask 分布分析 =====
        # 1. 计算每个 token 的绝对位置索引 (0, 1, 2, ...)
        seq_len = entropys.shape[1]
        token_indices = torch.arange(seq_len, device=entropys.device).unsqueeze(0).expand(bsz, -1)
        
        # 2. 获取每个样本的真实长度
        # 假设 eos_mask 是 (bs, seq_len)，1为有效，0为padding
        sample_lengths = eos_mask.sum(dim=1, keepdim=True) # (bs, 1)
        
        # 3. 计算相对位置 (0.0 ~ 1.0)
        # 避免除以0 (虽然理论上 mask 为 1 的地方 length 肯定 > 0)
        relative_positions = token_indices.float() / (sample_lengths.float() + 1e-6)
        
        # 4. 只提取有效 token 的数据
        valid_mask_bool = eos_mask.bool()
        flat_relative_pos = relative_positions[valid_mask_bool] # (Total_Valid_Tokens,)
        flat_high_ent_mask = high_entropy_mask[valid_mask_bool] # (Total_Valid_Tokens,)
        
        # 5. 分桶统计 (5 Bins: 0-20%, 20-40%, ...)
        num_bins = 5
        bin_results = []
        
        for i in range(num_bins):
            # 定义桶的范围
            lower = i / num_bins
            upper = (i + 1) / num_bins
            
            # 找到落在该进度区间内的 token
            if i == num_bins - 1:
                # 最后一个桶包含 1.0
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos <= upper)
            else:
                in_bin = (flat_relative_pos >= lower) & (flat_relative_pos < upper)
            
            if in_bin.sum() > 0:
                # 计算该进度区间内，高熵 mask 的比例
                avg_ratio = flat_high_ent_mask[in_bin].mean().item()
                bin_results.append(f"{avg_ratio:.1%}")
            else:
                bin_results.append("-")
        
        print(f"[Ent-Dist-Rel] (0% -> 100%): {' -> '.join(bin_results)}")

        teacher_signal = torch.exp(teacher_ids_log_probs)
        
        weighted_teacher_signal = teacher_coef * teacher_signal * high_entropy_mask

        # 统计信息
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens

        # 整个序列的平均 Outcome Reward 强度
        outcome_magnitude = scores[eos_mask.bool()].abs().mean().item()
        
        # Teacher Magnitude: 高熵区Teacher 信号的平均强度
        valid_high_ent_count = high_entropy_mask.sum() + epsilon
        teacher_magnitude = weighted_teacher_signal.sum() / valid_high_ent_count
        
        # Teacher Prob: 模型在高熵区域对 Teacher Token 的平均置信度
        avg_teacher_prob = (teacher_signal * high_entropy_mask).sum() / valid_high_ent_count

        print(f"[TIPO-Mix] Raw_S: {raw_score_mean:.2f} | Ent: {avg_all_entropy:.3f} | Diff: {mismatch_ratio:.1%} | "
              f"T_Prob(HighEnt): {avg_teacher_prob:.3f} | "
              f"Mag(Out/Tea): {outcome_magnitude:.3f}/{teacher_magnitude.item():.3f}")

        scores = scores * high_entropy_mask + weighted_teacher_signal

    return scores, scores

# reward!=0样本采用grpo adv，reward=0样本采用stuforce teacher cep [👎]
def compute_tipo_neg_advantage(token_level_rewards: torch.Tensor,
                            entropys: torch.Tensor,
                            eos_mask: torch.Tensor,
                            index: torch.Tensor,
                            teacher_predict_ids: torch.Tensor,
                            student_predict_ids: torch.Tensor,
                            teacher_ids_log_probs: torch.Tensor,
                            epsilon: float = 1e-6,
                            use_std: bool = True):
    """
    Compute advantage for GRPO, with special handling for zero-reward samples.
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
    zero_reward_mask = (scores == 0)  # shape: (bs,)

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

        # ===== 融合优势与teacher监督 =====
        teacher_signal = -teacher_ids_log_probs
        
        zero_reward_sample_mask = zero_reward_mask.unsqueeze(-1).tile([1, response_length]) * eos_mask
        scores = torch.where(
            zero_reward_sample_mask.bool(),
            teacher_signal,
            scores
        )

    return scores, scores

# 所有tok grpo adv + 前20% stuforce teacher ce
def compute_tipo_top20ptok_advantage(token_level_rewards: torch.Tensor,
                        entropys: torch.Tensor,
                        eos_mask: torch.Tensor,
                        index: torch.Tensor,
                        teacher_predict_ids: torch.Tensor,
                        student_predict_ids: torch.Tensor,
                        teacher_ids_log_probs: torch.Tensor,  # (bs, seq_len)
                        epsilon: float = 1e-6,
                        teacher_coef: float = 0.002,
                        use_std: bool = True):
    """
    Compute advantage for GRPO with teacher supervision on top-20% valid tokens per sample.
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

        # 计算 CE Loss
        ce_loss = -teacher_ids_log_probs 
        teacher_signal = -ce_loss

        # ===== 为每个样本创建前20% valid token的掩码 =====
        top_20_mask = torch.zeros_like(eos_mask)
        for i in range(bsz):
            # 获取该样本的有效token位置
            valid_positions = torch.where(eos_mask[i] > 0)[0]
            num_valid = len(valid_positions)
            if num_valid > 0:
                # 计算前20%的数量
                top_20_count = max(1, int(num_valid * 0.2))
                # 获取前20%的有效token位置
                top_20_positions = valid_positions[:top_20_count]
                # 设置掩码
                top_20_mask[i, top_20_positions] = 1.0
        
        # ===== 统计信息打印 =====
        valid_all_entropys = entropys[eos_mask.bool()]
        avg_all_entropy = valid_all_entropys.mean().item() if valid_all_entropys.numel() > 0 else 0.0

        total_valid_tokens = eos_mask.sum() + epsilon
        
        # Teacher/Student 不一致比例
        is_diff = (teacher_predict_ids != student_predict_ids).float() * eos_mask
        mismatch_ratio = is_diff.sum() / total_valid_tokens
        
        # 不一致位置的熵统计
        diff_entropys = entropys[is_diff.bool()]
        if diff_entropys.numel() > 0:
            diff_ent_min = diff_entropys.min().item()
            diff_ent_mean = diff_entropys.mean().item()
            diff_ent_max = diff_entropys.max().item()
        else:
            diff_ent_min = diff_ent_mean = diff_ent_max = 0.0

        # 前20% token上的不一致比例
        is_diff_top20 = is_diff * top_20_mask
        mismatch_ratio_top20 = is_diff_top20.sum() / (top_20_mask.sum() + epsilon)
        
        # 前20% token的数量统计
        top_20_count_total = top_20_mask.sum().item()

        print(f"[TIPO-Entro] Ent_Mean: {avg_all_entropy:.4f} | Diff_Ratio: {mismatch_ratio:.1%} | "
              f"Top20%_Tokens: {top_20_count_total:.0f} | Diff_Ratio_Top20%: {mismatch_ratio_top20:.1%} | "
              f"Diff_Ent(Min/Avg/Max): {diff_ent_min:.3f}/{diff_ent_mean:.3f}/{diff_ent_max:.3f} | teacher_w: {teacher_coef}")

        # ===== 融合优势与teacher监督（仅在前20% valid token上） =====
        scores = scores + teacher_coef * teacher_signal * top_20_mask

    return scores, scores
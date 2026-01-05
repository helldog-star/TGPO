# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
import numpy as np
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

__all__ = ['MIXDataParallelPPOActor']

from verl.workers.actor.dp_actor import DataParallelPPOActor

class MIXDataParallelPPOActor(DataParallelPPOActor):
    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__(config, actor_module, actor_optimizer)
        self.use_adaptive_temperature = self.config.use_adaptive_temperature
        self.adaptive_temperature_target_entropy = self.config.adaptive_temperature_target_entropy
        if self.use_adaptive_temperature:
            self.log_alpha = torch.tensor(np.log(self.config.entropy_coeff), dtype=torch.float)
            self.log_alpha.requires_grad = True
            from torch import optim
            self.alpha_optimizer = optim.AdamW([self.log_alpha],
                                          lr=self.config.alpha_lr,
                                          betas=(0.9, 0.999),
                                          weight_decay=1e-2)
        else:
            self.alpha_optimizer = None
            
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'prefix_mask']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.config.use_off_policy_loss and self.config.off_policy_loss_impl == 'seq':
            select_keys.append('on_logprobs_mean')
            select_keys.append('on_logprobs_std')
        if self.config.use_off_policy_loss and self.config.use_off_policy_probs:
            select_keys.append('target_probs')
        if self.config.use_tipo_loss:
            teacher_coef = data.meta_info["teacher_coef"]
            select_keys.append('teacher_predict_ids')
        if self.config.use_kdrl_loss:
            teacher_coef = data.meta_info["teacher_coef"]
            select_keys.append('teacher_log_prob')

        batch = data.select(batch_keys=select_keys).batch

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()
                if self.alpha_optimizer is not None:
                    self.alpha_optimizer.zero_grad()

                for data in micro_batches:
                    print("MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    clip_ratio = self.config.clip_ratio
                    entropy_coeff = self.config.entropy_coeff

                    if self.config.use_tipo_loss:
                        entropy, log_prob, teacher_ids_log_probs = self._forward_teacher_ids_micro_batch(micro_batch=data, temperature=temperature)
                    else:
                        entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    if self.config.use_sft_multitask_loss:
                        assert self.config.use_off_policy_loss is False, 'Either use off-policy loss or sft multitask loss. You cannot set both to be True.'
                        from .mix_core_alg import compute_sft_pure_loss
                        off_policy_mask = data['prefix_mask'].any(-1) # [No]
                        off_policy_logprob = log_prob[off_policy_mask]
                        off_policy_eos_mask = response_mask[off_policy_mask]
                        
                        sft_loss = compute_sft_pure_loss(log_prob=off_policy_logprob,
                                                        eos_mask=off_policy_eos_mask)
                        if off_policy_logprob.numel() == 0:
                            sft_loss = torch.tensor(0.0)
                        
                        on_policy_mask = ~off_policy_mask
                        on_policy_logprob = log_prob[on_policy_mask]
                        on_policy_old_logprob = old_log_prob[on_policy_mask]
                        
                        # assert self.config.algorithm.adv_estimator == 'grpo_split'
                        # The on-policy advantages should not be computed together with the off-policy rewards
                        on_policy_advantages = advantages[on_policy_mask]
                        on_policy_eos_mask = response_mask[on_policy_mask]
                        
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                            old_log_prob=on_policy_old_logprob, log_prob=on_policy_logprob,
                            advantages=on_policy_advantages,
                            eos_mask=on_policy_eos_mask,
                            cliprange=clip_ratio,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        
                        pg_loss = sft_loss * self.config.sft_loss_coef + pg_loss

                    elif self.config.use_off_policy_loss:
                        from .mix_core_alg import compute_token_on_off_policy_loss
                        loss_fn = compute_token_on_off_policy_loss

                        ret_dict = loss_fn(old_log_prob=old_log_prob, 
                            log_prob=log_prob,
                            advantages=advantages,
                            eos_mask=response_mask,
                            cliprange=clip_ratio,
                            clip_upper_bound=self.config.clip_upper_bound,
                            prefix_mask=data['prefix_mask'],
                            off_cliprange=self.config.off_policy_cliprange,
                            off_normalize=self.config.off_policy_normalize,
                            off_max_clip=self.config.off_policy_max_clip if self.config.off_policy_max_clip != -1 else None,
                            off_min_clip=self.config.off_policy_min_clip if self.config.off_policy_min_clip != -1 else None,
                            all_max_clip=self.config.all_max_clip if self.config.all_max_clip != -1 else None,
                            off_policy_reshape=self.config.off_policy_reshape,
                            off_policy_reshape_weight=self.config.off_policy_reshape_weight,
                            off_policy_reshape_pow_exp=self.config.off_policy_reshape_pow_exp,
                            on_policy_reshape=self.config.on_policy_reshape,
                            on_policy_reshape_weight=self.config.on_policy_reshape_weight,
                            on_policy_reshape_pow_exp=self.config.on_policy_reshape_pow_exp,
                            target_probs=data['target_probs'] if 'target_probs' in data else None,
                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                            loss_remove_clip=self.config.loss_remove_clip
                        )
                        pg_loss = ret_dict['pg_loss']
                        off_pg_loss = ret_dict['off_pg_loss']
                        on_pg_loss = ret_dict['on_pg_loss']
                        off_pg_clipfrac = ret_dict['off_pg_clipfrac']
                        pg_clipfrac = ret_dict['on_pg_clipfrac']
                        ppo_kl = ret_dict['ppo_kl']
                        
                        data = {
                            'actor/off_pg_loss': off_pg_loss.detach().item(),
                            'actor/on_pg_loss': on_pg_loss.detach().item(),
                            'actor/off_pg_clipfrac': off_pg_clipfrac.detach().item(),
                        }
                        if 'off_policy_prob' in ret_dict:
                            data['actor/off_policy_prob'] = ret_dict['off_policy_prob'].detach().item()
                        if 'on_policy_prob' in ret_dict:
                            data['actor/on_policy_prob'] = ret_dict['on_policy_prob'].detach().item()
                        if 'off_ratio_mean' in ret_dict:
                            data['actor/off_ratio_mean'] = ret_dict['off_ratio_mean'].detach().item()
                        if 'off_ratio_max_clip_frac' in ret_dict:
                            data['actor/off_ratio_max_clip_frac'] = ret_dict['off_ratio_max_clip_frac'].detach().item()
                        if 'off_ratio_min_clip_frac' in ret_dict:
                            data['actor/off_ratio_min_clip_frac'] = ret_dict['off_ratio_min_clip_frac'].detach().item()
                        append_to_dict(metrics, data)

                    elif self.config.use_tipo_loss:

                        from .mix_core_alg import compute_token_on_tipo_loss
                        loss_fn = compute_token_on_tipo_loss
                        pg_loss, lm_loss, teacher_reg_loss, pg_clipfrac, ppo_kl = loss_fn(old_log_prob=old_log_prob, log_prob=log_prob,
                                                                                            advantages=advantages,
                                                                                            eos_mask=response_mask,
                                                                                            teacher_ids_log_probs=teacher_ids_log_probs,
                                                                                            teacher_coef=teacher_coef,
                                                                                            cliprange=clip_ratio,
                                                                                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                                                                                            loss_remove_clip=self.config.loss_remove_clip)
                        data = {
                            'actor/lm_loss': lm_loss.detach().item(),
                            'actor/teacher_reg_loss': teacher_reg_loss.detach().item(),
                            'actor/teacher_coef': teacher_coef
                        }
                        append_to_dict(metrics, data)

                    elif self.config.use_kdrl_loss:

                        from .mix_core_alg import compute_token_on_kdrl_loss
                        loss_fn = compute_token_on_kdrl_loss
                        teacher_log_prob = data['teacher_log_prob']
                        pg_loss, lm_loss, teacher_reg_loss, pg_clipfrac, ppo_kl = loss_fn(old_log_prob=old_log_prob, log_prob=log_prob,
                                                                                            advantages=advantages,
                                                                                            eos_mask=response_mask,
                                                                                            teacher_log_prob=teacher_log_prob,
                                                                                            teacher_coef=teacher_coef,
                                                                                            cliprange=clip_ratio,
                                                                                            loss_remove_token_mean=self.config.loss_remove_token_mean,
                                                                                            loss_remove_clip=self.config.loss_remove_clip)
                        data = {
                            'actor/lm_loss': lm_loss.detach().item(),
                            'actor/teacher_reg_loss': teacher_reg_loss.detach().item(),
                            'actor/teacher_coef': teacher_coef
                        }
                        append_to_dict(metrics, data)


                    else:
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob, log_prob=log_prob,
                                                                                advantages=advantages,
                                                                                eos_mask=response_mask,
                                                                                cliprange=clip_ratio,
                                                                                loss_remove_token_mean=self.config.loss_remove_token_mean,
                                                                                loss_remove_clip=self.config.loss_remove_clip)
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    if self.config.use_adaptive_temperature:
                        if self.config.use_adaptive_temperature_fixed is False:
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            entropy_coeff = self.log_alpha.exp()
                            if self.config.adaptive_temperature_clip > 0:
                                entropy_coeff = torch.clamp(entropy_coeff, max=self.config.adaptive_temperature_clip)
                            alpha_loss = verl_F.masked_mean(entropy - target_entropy, response_mask).detach() * entropy_coeff
                            alpha_loss = alpha_loss / self.gradient_accumulation
                            alpha_loss.backward()
                            
                            policy_loss = pg_loss - entropy_loss * entropy_coeff.detach().item()
                            metrics['actor/alpha_loss'] = alpha_loss.detach().item()
                            metrics['actor/entropy_coeff'] = entropy_coeff.detach().item()
                            metrics['actor/log_alpha'] = self.log_alpha.detach().item()
                        else: # fixed strategy for entropy coeff
                            target_entropy = self.config.adaptive_temperature_target_entropy
                            # cur_entropy = verl_F.masked_mean(entropy, response_mask)
                            entropy_coeff = (target_entropy / entropy_loss).detach().item() * self.config.entropy_coeff
                            policy_loss = pg_loss - entropy_loss * entropy_coeff
                            metrics['actor/entropy_coeff'] = entropy_coeff
                    else:
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef
                    if self.config.use_ppo_kl_loss:
                        policy_loss = policy_loss + ppo_kl.abs() * self.config.ppo_kl_loss_coef
                        metrics['actor/ppo_kl_loss'] = ppo_kl.abs().detach().item()
                        
                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                grad_norm = self._optimizer_step()
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.zero_grad()
        return metrics

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        if self.alpha_optimizer is not None:
            self.alpha_optimizer.step()
        return grad_norm

        
    def compute_log_prob_w_ids(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        predict_ids_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs, predict_ids = self._forward_micro_batch(micro_batch, temperature=temperature, return_ids=True)
            log_probs_lst.append(log_probs)
            predict_ids_lst.append(predict_ids)
            if calculate_entropy:
                entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        predict_ids = torch.concat(predict_ids_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            predict_ids = predict_ids[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys, predict_ids
    

    def _forward_micro_batch(self, micro_batch, temperature, return_ids=False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # 如果需要 return_ids，先在 unpad 状态下计算 argmax/topk，避免 OOM
                predict_ids_rmpad = None
                if return_ids:
                    # 获取预测的 token ID
                    predict_ids_rmpad = torch.argmax(logits_rmpad, dim=-1) # (total_nnz,)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    if return_ids and predict_ids_rmpad is not None:
                        predict_ids_rmpad = gather_outpus_and_unpad(predict_ids_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                full_predict_ids = None
                if return_ids:
                    full_predict_ids = pad_input(hidden_states=predict_ids_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    
                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                
                predict_ids = None
                if return_ids:
                    predict_ids = full_predict_ids.squeeze(-1)[:, -response_length - 1:-1]

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                predict_ids = None
                if return_ids:
                    predict_ids = torch.argmax(logits, dim=-1) # (bs, response_len)

            # return entropy, log_probs
            if return_ids:
                return entropy, log_probs, predict_ids
            else:
                return entropy, log_probs


    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys

    
    def compute_teacher_ids_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'teacher_predict_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        teacher_ids_log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                # entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
                entropy, log_probs, teacher_ids_log_probs = self._forward_teacher_ids_micro_batch(
                    micro_batch, temperature=temperature
                )
            log_probs_lst.append(log_probs)
            teacher_ids_log_probs_lst.append(teacher_ids_log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        teacher_ids_log_probs = torch.concat(teacher_ids_log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            teacher_ids_log_probs = teacher_ids_log_probs[revert_indices]
            if calculate_entropy:
                entropys = entropys[revert_indices]

        return log_probs, entropys, teacher_ids_log_probs


    def _forward_teacher_ids_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            teacher_predict_ids = micro_batch["teacher_predict_ids"]

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # 创建全长full_teacher_ids, 填充-100
                full_teacher_ids = torch.full_like(input_ids, -100)
                full_teacher_ids[:, -response_length:] = teacher_predict_ids
                teacher_ids_rmpad = index_first_axis(
                    rearrange(full_teacher_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1) # (1, total_nnz)


                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                teacher_ids_rmpad_rolled = torch.roll(teacher_ids_rmpad, shifts=-1, dims=1)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)
                    teacher_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(teacher_ids_rmpad_rolled, None, 
                                                                                    self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)
                teacher_ids_rmpad_rolled = teacher_ids_rmpad_rolled.squeeze(0)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                teacher_ids_log_probs = logprobs_from_logits(
                    logits=logits_rmpad,
                    labels=teacher_ids_rmpad_rolled
                )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    teacher_ids_log_probs = gather_outpus_and_unpad(
                        teacher_ids_log_probs, 
                        gather_dim=0, 
                        unpad_dim=0, 
                        padding_size=pad_size
                    )
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                full_teacher_ids_log_probs = pad_input(
                                            hidden_states=teacher_ids_log_probs.unsqueeze(-1), 
                                            indices=indices, 
                                            batch=batch_size, 
                                            seqlen=seqlen
                                        )

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                teacher_ids_log_probs = full_teacher_ids_log_probs.squeeze(-1)[:, -response_length-1:-1]

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                teacher_ids_log_probs = logprobs_from_logits(logits, teacher_predict_ids)
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs, teacher_ids_log_probs


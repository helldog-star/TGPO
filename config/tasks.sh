# 实验 / 算法配置：新增实验时只需改本文件
# 被 run_train_merge_eval.sh 与 train.sh source 后使用

# ---------- 1. 支持的 algo 列表（新增时在此追加，空格分隔） ----------
ALGOS="grpo kdrl luffy rkl tipo_adv tipo_reg"

# ---------- 2. 需要 teacher 的 algo（新增需 teacher 的实验时在此追加） ----------
NEED_TEACHER_ALGOS="kdrl rkl tipo_adv tipo_reg"

# ---------- 3. 工具函数 ----------
# 检查 algo 是否存在
algo_exists() { [[ " $ALGOS " == *" $1 "* ]]; }

# 检查是否需要 teacher（--teacher-model-path 须为有效路径）
algo_needs_teacher() { [[ " $NEED_TEACHER_ALGOS " == *" $1 "* ]]; }

# 默认 EXP_NAME：${algo}_qwen2d5_math_7b；若需例外可在此加 case
get_exp_name() { echo "${1}_qwen2d5_math_7b"; }

# 返回该 algo 的 hydra 差异化覆盖；内部会用到 $TEACHER_MODEL_PATH（train.sh 在调用前已设置）
get_algo_extra() {
    case "$1" in
        grpo)
            # GRPO: 基础 GRPO 算法，无 teacher，无 off-policy loss
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_off_policy_loss=False \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0"
            ;;
        kdrl)
            # KDRL: GRPO + teacher distillation (coef=0.002) + KDRL loss
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.teacher_ref.enable=True \
actor_rollout_ref.teacher_ref.teacher_coef=0.002 \
actor_rollout_ref.teacher_ref.model_path=$TEACHER_MODEL_PATH \
actor_rollout_ref.teacher_ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_kdrl_loss=True \
actor_rollout_ref.actor.use_off_policy_loss=False \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0"
            ;;
        luffy)
            # Luffy: GRPO + off-policy loss (reshape=p_div_p_0.1) + loss clipping + prefix_ratio=1.0
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_off_policy_loss=True \
actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=True \
actor_rollout_ref.actor.loss_remove_clip=True \
actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
actor_rollout_ref.rollout.max_prefix_ratio=1.0"
            ;;
        rkl)
            # RKL: RKL estimator + teacher (no coef/decay)
            echo "algorithm.adv_estimator=rkl \
actor_rollout_ref.teacher_ref.enable=True \
actor_rollout_ref.teacher_ref.model_path=$TEACHER_MODEL_PATH \
actor_rollout_ref.teacher_ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_off_policy_loss=False \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0"
            ;;
        tipo_adv)
            # TIPO-Adv: TIPO estimator + teacher (coef=0.02, decay=1e-4)
            echo "algorithm.adv_estimator=tipo \
actor_rollout_ref.teacher_ref.enable=True \
actor_rollout_ref.teacher_ref.min_teacher_coef=0.0 \
actor_rollout_ref.teacher_ref.teacher_coef=0.02 \
actor_rollout_ref.teacher_ref.decay_rate=0.0001 \
actor_rollout_ref.teacher_ref.model_path=$TEACHER_MODEL_PATH \
actor_rollout_ref.teacher_ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_off_policy_loss=False \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0"
            ;;
        tipo_reg)
            # TIPO-Reg: GRPO + teacher (coef=0.002, decay=1e-5) + TIPO loss
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.teacher_ref.enable=True \
actor_rollout_ref.teacher_ref.min_teacher_coef=0.0 \
actor_rollout_ref.teacher_ref.teacher_coef=0.002 \
actor_rollout_ref.teacher_ref.decay_rate=0.00001 \
actor_rollout_ref.teacher_ref.model_path=$TEACHER_MODEL_PATH \
actor_rollout_ref.teacher_ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_tipo_loss=True \
actor_rollout_ref.actor.use_off_policy_loss=False \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0"
            ;;
        *)
            echo ""
            ;;
    esac
}

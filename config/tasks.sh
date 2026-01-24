# 实验 / 算法配置：新增实验时只需改本文件
# 被 run_train_merge_eval.sh 与 train.sh source 后使用

# ---------- 1. 支持的 algo 列表（新增时在此追加，空格分隔） ----------
# 1.5B 版本使用 *_1d5b 后缀
ALGOS="grpo kdrl luffy rkl tipo_adv tipo_reg grpo_1d5b kdrl_1d5b luffy_1d5b rkl_1d5b tipo_adv_1d5b tipo_reg_1d5b"

# ---------- 2. 需要 teacher 的 algo（新增需 teacher 的实验时在此追加） ----------
NEED_TEACHER_ALGOS="kdrl rkl tipo_adv tipo_reg kdrl_1d5b rkl_1d5b tipo_adv_1d5b tipo_reg_1d5b"

# ---------- 3. 工具函数 ----------
# 检查 algo 是否存在
algo_exists() { [[ " $ALGOS " == *" $1 "* ]]; }

# 检查是否需要 teacher（--teacher-model-path 须为有效路径）
algo_needs_teacher() { [[ " $NEED_TEACHER_ALGOS " == *" $1 "* ]]; }

# 默认 EXP_NAME：
# - 7B: ${algo}_qwen2d5_math_7b
# - 1.5B: ${algo%_1d5b}_qwen2d5_math_1d5b
get_exp_name() {
    if [[ "$1" == *_1d5b ]]; then
        echo "${1%_1d5b}_qwen2d5_math_1d5b"
    else
        echo "${1}_qwen2d5_math_7b"
    fi
}

# 返回该 algo 的 hydra 差异化覆盖；内部会用到 $TEACHER_MODEL_PATH（train.sh 在调用前已设置）
get_algo_extra() {
    local _algo="$1"
    if [[ "$_algo" == *_1d5b ]]; then
        _algo="${_algo%_1d5b}"
    fi
    case "$_algo" in
        grpo)
            # GRPO: 基础 GRPO 算法，无 teacher，无 off-policy loss
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_off_policy_loss=True \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=True \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
algorithm.grpo_use_std=False"
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
actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
algorithm.grpo_use_std=False"
            ;;
        luffy)
            # Luffy: GRPO + off-policy loss (reshape=p_div_p_0.1) + loss clipping + prefix_ratio=1.0
            echo "algorithm.adv_estimator=grpo \
actor_rollout_ref.actor.use_off_policy_loss=True \
actor_rollout_ref.actor.off_policy_reshape=p_div_p_0.1 \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.actor.loss_remove_clip=True \
actor_rollout_ref.rollout.min_prefix_ratio=1.0 \
actor_rollout_ref.rollout.max_prefix_ratio=1.0 \
algorithm.grpo_use_std=False"
            ;;
        rkl)
            # RKL: RKL estimator + teacher (no coef/decay)
            echo "algorithm.adv_estimator=rkl \
actor_rollout_ref.teacher_ref.enable=True \
actor_rollout_ref.teacher_ref.model_path=$TEACHER_MODEL_PATH \
actor_rollout_ref.teacher_ref.fsdp_config.param_offload=True \
actor_rollout_ref.actor.use_off_policy_loss=True \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
algorithm.grpo_use_std=False"
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
actor_rollout_ref.actor.use_off_policy_loss=True \
actor_rollout_ref.actor.off_policy_normalize=False \
actor_rollout_ref.actor.off_policy_loss_impl=token \
actor_rollout_ref.actor.loss_remove_token_mean=False \
actor_rollout_ref.rollout.min_prefix_ratio=0.0 \
actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
algorithm.grpo_use_std=True"
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
actor_rollout_ref.rollout.max_prefix_ratio=0.0 \
algorithm.grpo_use_std=True"
            ;;
        *)
            echo ""
            ;;
    esac
}

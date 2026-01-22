# 环境配置：conda、环境变量等统一管理
# 被 train.sh、eval_scripts/*.sh 等脚本 source 后使用
# 支持通过环境变量覆盖（如 CONDA_SH_PATH, CONDA_ENV_NAME）

# ---------- 1. Conda 配置（可通过环境变量覆盖） ----------
CONDA_SH_PATH="${CONDA_SH_PATH:-/opt/conda/etc/profile.d/conda.sh}"  # Conda初始化脚本路径
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tgpo}"  # Conda环境名称

# ---------- 2. 激活 conda 环境 ----------
# 函数：setup_conda_env
# 用法：在脚本中调用 setup_conda_env 即可完成 conda 初始化与激活
setup_conda_env() {
    if [[ -n "$CONDA_SH_PATH" && -f "$CONDA_SH_PATH" ]]; then
        source "$CONDA_SH_PATH"
        conda activate "$CONDA_ENV_NAME" || {
            echo "错误: 无法激活 conda 环境 '$CONDA_ENV_NAME'"
            echo "提示: 可通过 CONDA_ENV_NAME 环境变量指定其他环境名"
            exit 1
        }
    else
        echo "警告: CONDA_SH_PATH 未设置或文件不存在: ${CONDA_SH_PATH:-未设置}"
        echo "提示: 可通过 CONDA_SH_PATH 环境变量指定 conda.sh 路径"
        echo "跳过 conda 激活，假设环境已配置"
    fi
}

# ---------- 3. 通用环境变量（训练/评估共用） ----------
# 这些变量在 setup_conda_env 后设置，可通过环境变量覆盖
setup_common_env() {
    export no_proxy="${no_proxy:-127.0.0.1,localhost}"
    export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
}

# ---------- 4. 完整环境设置（一步到位） ----------
# 函数：setup_all_env
# 用法：在脚本中调用 setup_all_env 即可完成所有环境设置
setup_all_env() {
    setup_conda_env
    setup_common_env
}

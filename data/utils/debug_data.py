from transformers import AutoTokenizer
from datasets import load_from_disk

# MODEL_LOCAL_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen3-30B-A3B-Thinking-2507-aligned"
MODEL_LOCAL_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-Math-7B-aligned"
# MODEL_LOCAL_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-Math-1.5B"
# MODEL_LOCAL_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-Math-7B-16k-think"
DATASET_PATH = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/datasets/Openr1-Math-46k-8192"

print(f"正在加载分词器: {MODEL_LOCAL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH, padding_side="left")

print(f"正在加载数据集: {DATASET_PATH}")
ds_full = load_from_disk(DATASET_PATH)

# 取第0条样本
sample = ds_full[0]

# 取多轮对话数据，假定你的数据字典结构如示例（prompt是list，每个元素是dict，包含role和content）
dialogue = sample['prompt']

# print("数据原始多轮对话内容：")
# for i, turn in enumerate(dialogue):
#     print(f"{turn['role']}: {turn['content']}\n")

# 使用apply_chat_template得到最终prompt字符串
prompt_text = tokenizer.apply_chat_template(
    dialogue,
    tokenize=False,                # 只看文本，便于人工debug
    add_generation_prompt=True     # 如果需要模型继续生成建议加这一项
)

print("\n经过chat_template处理后的prompt输入（用于模型）：")
print(prompt_text)



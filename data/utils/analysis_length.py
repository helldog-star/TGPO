import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# --- 配置 ---
model_path = '/mnt/nvme3/liuxinyu/models/Qwen3-30B-A3B-Thinking-2507'
data_path = '/mnt/nvme3/liuxinyu/TGPO/data/openr1.a3b_correct_35k.parquet'

# 1. 加载 Tokenizer
print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 2. 读取数据
print("正在读取数据...")
df = pd.read_parquet(data_path)

# 辅助函数：从数据格式中提取文本
def get_text(data, role):
    if isinstance(data, (list, np.ndarray)):
        for item in data:
            if isinstance(item, dict) and item.get('role') == role:
                return item.get('content', '')
    return ""

# 3. 计算 Token 数
print("正在计算 Token (可能需要一点时间)...")

# 提取文本
prompts = df['prompt'].apply(lambda x: get_text(x, 'user'))
responses = df['target'].apply(lambda x: get_text(x, 'assistant'))

# 计算长度
df['p_tokens'] = prompts.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
df['r_tokens'] = responses.apply(lambda x: len(tokenizer.encode(x, add_special_tokens=False)))
df['total'] = df['p_tokens'] + df['r_tokens']

# 4. 直接输出统计结果
print("\n" + "="*30)
print("Token 统计结果")
print("="*30)
# describe() 会自动输出 count, mean, std, min, 25%, 50%, 75%, max
print(df[['p_tokens', 'r_tokens', 'total']].describe().to_string(float_format="%.2f"))

print("\n" + "="*30)
print("超长检测 ( > 8192 )")
print("="*30)
over_limit = (df['total'] > 8192).sum()
print(f"超过 8192 Token 的样本数: {over_limit}")
print(f"占比: {over_limit / len(df) * 100:.2f}%")
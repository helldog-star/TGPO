# import pandas as pd
# import os

# input_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k.parquet"
# output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k_think.parquet"

# # 读取parquet
# df = pd.read_parquet(input_path)

# def add_think_prefix(target_list):
#     # target字段常见为list of dict，每个dict有content字段
#     assert len(target_list) == 1
#     if isinstance(target_list[0], dict) and 'content' in target_list[0] and not target_list[0]['content'].startswith('<think>\n'):
#         target_list[0]['content'] = '<think>\n' + target_list[0]['content']
#     return target_list

# # 对每行 target 批量处理
# df['target'] = df['target'].apply(add_think_prefix)

# # 保存新的 parquet 文件
# df.to_parquet(output_path, index=False)
# print(f"处理完毕，已保存到：{output_path}")


import pandas as pd
import os

input_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k.parquet"
output_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k_nothink.parquet"

# 读取parquet
df = pd.read_parquet(input_path)

def remove_think_prefix(target_list):
    """移除 <think>\n 前缀"""
    assert len(target_list) == 1
    if isinstance(target_list[0], dict) and 'content' in target_list[0]:
        content = target_list[0]['content']
        # 如果以 <think>\n 开头，则移除
        if content.startswith('<think>\n'):
            target_list[0]['content'] = content[len('<think>\n'):]
    return target_list

# 对每行 target 批量处理
df['target'] = df['target'].apply(remove_think_prefix)

# 保存新的 parquet 文件
df.to_parquet(output_path, index=False)
print(f"处理完毕，已保存到：{output_path}")
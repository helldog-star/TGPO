import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载两个模型的 tokenizer
student_tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen2.5-Math-7B")
teacher_tokenizer = AutoTokenizer.from_pretrained("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/models/Qwen3-30B-A3B-Thinking-2507")

print(f"Student tokenizer vocab_size: {len(student_tokenizer)}")
print(f"Teacher tokenizer vocab_size: {len(teacher_tokenizer)}")

# 查看额外的 token
student_vocab = set(student_tokenizer.get_vocab().keys())
teacher_vocab = set(teacher_tokenizer.get_vocab().keys())

extra_tokens = student_vocab - teacher_vocab
print(f"\nStudent 中比 Teacher 多的 token 数: {len(extra_tokens)}")
print(f"额外 token 示例: {list(extra_tokens)[:10]}")
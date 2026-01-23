from transformers import AutoTokenizer

# 模型路径
model_path = "/mnt/nvme3/liuxinyu/models/Qwen2.5-Math-1.5B-Instruct-aligned"

def check_chat_template():
    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 定义一个典型的对话示例
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How many r's are in 'strawberry'?"},
            {"role": "assistant", "content": "There are 3 r's in 'strawberry'."},
            {"role": "user", "content": "What is 2 + 2?"}
        ]

        # 1. 查看渲染后的字符串 (用于查看格式)
        rendered_chat = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. 查看 Token ID (用于检查特殊 token)
        tokenized_chat = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True
        )

        print("\n" + "="*30 + " 渲染后的 Chat Template " + "="*30)
        print(rendered_chat)
        print("="*80 + "\n")

        print(f"Token IDs 长度: {len(tokenized_chat)}")
        print(f"前 10 个 Token IDs: {tokenized_chat[:10]}")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    check_chat_template()
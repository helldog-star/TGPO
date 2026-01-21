import os
import time
import json
import argparse
import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI, APIError

# 以下由 parse_args() 及 __main__ 赋值，供后续函数使用
DATASET_PATH = None
MODEL_LOCAL_PATH = None
OUTPUT_PARQUET_PATH = None
TEMP_CACHE_JSON_PATH = None
VLLM_BASE_URL = None
MAX_WORKERS = None
BATCH_SIZE = None
tokenizer = None
client = None
SERVING_MODEL_NAME = None


def parse_args():
    parser = argparse.ArgumentParser(description="OpenR1-Math 数据准备：经 vLLM 推理后生成训练用 Parquet")
    parser.add_argument("--dataset-path", required=True, help="load_from_disk 用的数据集目录路径")
    parser.add_argument("--model-local-path", required=True, help="本地模型路径，用于加载 Tokenizer")
    parser.add_argument("--output-parquet-path", required=True, help="最终输出 Parquet 文件路径")
    parser.add_argument("--temp-cache-json-path", required=True, help="断点续传缓存 JSON 路径")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1", help="vLLM OpenAI 兼容接口地址")
    parser.add_argument("--max-workers", type=int, default=32, help="推理线程数")
    parser.add_argument("--batch-size", type=int, default=16, help="每批推理样本数")
    return parser.parse_args()

# --- 断点续传和缓存逻辑 ---
def load_cache():
    """尝试加载已完成的缓存结果"""
    if os.path.exists(TEMP_CACHE_JSON_PATH):
        print(f"✅ 发现缓存文件: {TEMP_CACHE_JSON_PATH}，尝试从断点继续...")
        try:
            with open(TEMP_CACHE_JSON_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print("❌ 缓存文件损坏，将从头开始。")
            return {}
    return {}

def save_cache(cache_data):
    """保存当前的缓存结果"""
    with open(TEMP_CACHE_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)

# --- VLLM 推理函数 (使用 OpenAI SDK) ---
def call_vllm_api(batch_prompts, indices):
    """
    使用 OpenAI SDK 调用 vLLM API 进行批量推理
    """
    # 1. 构造 prompt 文本 (Chat Template -> Text)
    # 因为我们要批量发送，使用 completions 接口比 chat.completions 更容易处理 list[str]
    prompt_texts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in batch_prompts
    ]

    try:
        # 2. 发送请求 (Completions API 支持 prompt 为列表)
        response = client.completions.create(
            model=SERVING_MODEL_NAME,
            prompt=prompt_texts,
            max_tokens=8192,
            temperature=0.7,
            # 如果你需要 deepseek_r1 的思考过程，它通常包含在生成的文本中
        )
        
        # 3. 解析结果
        # OpenAI SDK 返回的 choices 顺序通常与 prompt 顺序一致，但为了安全我们依靠索引
        processed_results = {}
        
        # vLLM 对 batch 请求的返回顺序通常是对应的，直接遍历即可
        for i, choice in enumerate(response.choices):
            original_index = indices[i]
            generated_text = choice.text.strip()
            
            # 格式化为 target 字段
            target_value = [{"content": generated_text, "role": "assistant"}]
            processed_results[str(original_index)] = target_value
            
        return processed_results

    except APIError as e:
        print(f"⚠️ API 报错 (索引 {indices[0]}-{indices[-1]}): {e}")
        return {}
    except Exception as e:
        print(f"⚠️ 未知错误 (索引 {indices[0]}-{indices[-1]}): {e}")
        return {}

# --- 主执行逻辑 ---
def main():
    # 确保输出目录存在
    d = os.path.dirname(OUTPUT_PARQUET_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    d = os.path.dirname(TEMP_CACHE_JSON_PATH)
    if d:
        os.makedirs(d, exist_ok=True)

    # 1. 加载数据集
    print(f"正在加载数据集: {DATASET_PATH}")
    ds_full = load_from_disk(DATASET_PATH)
    # # Debug
    # ds_full = ds_full.select(range(10))
    
    # 2. 加载缓存和确定待处理索引
    results_cache = load_cache()
    
    total_samples = len(ds_full)
    all_indices = list(range(total_samples))
    
    # 排除已完成的索引
    completed_indices = set(map(int, results_cache.keys()))
    pending_indices = [i for i in all_indices if i not in completed_indices]
    
    print(f"总样本数: {total_samples}")
    print(f"已完成样本数: {len(completed_indices)}")
    print(f"待处理样本数: {len(pending_indices)}")

    if not pending_indices:
        print("所有样本均已完成，跳过推理步骤。")
        finalize_results(ds_full, results_cache, total_samples)
        return

    # 3. 构造批量任务
    tasks = []
    for i in range(0, len(pending_indices), BATCH_SIZE):
        batch_indices = pending_indices[i : i + BATCH_SIZE]
        batch_prompts = [ds_full[j]["prompt"] for j in batch_indices]
        tasks.append((batch_prompts, batch_indices))
        
    print(f"准备执行 {len(tasks)} 个批次任务，线程数: {MAX_WORKERS}")

    # 4. 多线程执行任务
    futures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for batch_prompts, indices in tasks:
            future = executor.submit(call_vllm_api, batch_prompts, indices)
            futures.append(future)

        # 5. 实时处理结果和断点保存
        for i, future in enumerate(futures):
            try:
                batch_results = future.result()
                results_cache.update(batch_results)
                
                # 每处理 50 个批次或任务结束时保存一次缓存
                if (i + 1) % 50 == 0 or i == len(futures) - 1:
                    save_cache(results_cache)
                    elapsed = time.time() - start_time
                    progress = len(results_cache) / total_samples
                    speed = len(results_cache) / elapsed if elapsed > 0 else 0
                    print(
                        f"🔥 Progress: {len(results_cache)}/{total_samples} ({progress:.2%}) "
                        f"| Elapsed: {elapsed:.2f}s | Speed: {speed:.2f} samples/s"
                    )

            except Exception as e:
                print(f"🔥 捕获到线程执行错误: {e}")
                
    # 6. 最终保存
    finalize_results(ds_full, results_cache, total_samples)


def finalize_results(ds_full, results_cache, total_samples):
    """将缓存结果合并到Dataset并保存为Parquet"""
    
    print("\n--- 任务完成，开始最终数据合并 ---")
    
    # 将缓存的字典转换为列表，确保顺序和完整性
    all_targets = [None] * total_samples
    valid_count = 0
    for i in range(total_samples):
        key = str(i)
        if key in results_cache:
            all_targets[i] = results_cache[key]
            valid_count += 1
        else:
            # 如果存在未完成的样本，使用原始的target字段或设置为空
            all_targets[i] = ds_full[i]["target"] if "target" in ds_full[i] else [{"content": "ERROR_OR_PENDING", "role": "assistant"}]
            
    print(f"最终有效结果数: {valid_count}/{total_samples}")
    
    # 将 target 列表添加到原始数据集的副本中
    final_ds = ds_full.add_column("new_target", all_targets)
    final_ds = final_ds.remove_columns(["target"]).rename_column("new_target", "target")
    
    # 保存为 Parquet
    final_ds.to_parquet(OUTPUT_PARQUET_PATH)
    print(f"🎉 最终结果已保存到: {OUTPUT_PARQUET_PATH}")
    
    # 清理缓存文件
    if os.path.exists(TEMP_CACHE_JSON_PATH):
        os.remove(TEMP_CACHE_JSON_PATH)
        print("🗑️ 临时缓存文件已清除。")


if __name__ == "__main__":
    args = parse_args()
    DATASET_PATH = args.dataset_path
    MODEL_LOCAL_PATH = args.model_local_path
    OUTPUT_PARQUET_PATH = args.output_parquet_path
    TEMP_CACHE_JSON_PATH = args.temp_cache_json_path
    VLLM_BASE_URL = args.vllm_base_url
    MAX_WORKERS = args.max_workers
    BATCH_SIZE = args.batch_size

    print("正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH, padding_side="left")
    client = OpenAI(base_url=VLLM_BASE_URL, api_key="EMPTY")
    try:
        models_list = client.models.list()
        SERVING_MODEL_NAME = models_list.data[0].id
        print(f"✅ 连接成功，服务端模型名称: {SERVING_MODEL_NAME}")
    except Exception as e:
        print(f"❌ 无法连接到 vLLM 服务，请检查服务是否启动: {e}")
        exit(1)

    start_time = time.time()
    main()
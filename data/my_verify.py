import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from math_verify import parse, verify
from tqdm import tqdm

BATCH_SIZE = 1024

def parse_args():
    p = argparse.ArgumentParser(description="对采样结果 parquet 做数学答案验证，输出 _correct / _wrong")
    p.add_argument("--input", required=True, help="输入 parquet 路径（vLLM 采样结果）")
    return p.parse_args()

def process_batch(df_batch):
    """
    验证批次数据，返回验证结果列表和提取出的答案列表
    """
    validation_results = []
    extracted_preds = []
    
    # 提取数据
    try:
        responses = df_batch['target'].apply(
            lambda x: x[0]['content'] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else ""
        ).tolist()

        ground_truths = df_batch['reward_model'].apply(
            lambda x: x.get('ground_truth', "") if isinstance(x, dict) else ""
        ).tolist()

    except Exception as e:
        print(f"数据提取错误: {e}")
        # 如果提取失败，返回全 False 和空字符串，保持长度一致
        return [False] * len(df_batch), [""] * len(df_batch)

    # 验证逻辑
    for resp, gt in zip(responses, ground_truths):
        try:
            if not resp or not gt:
                validation_results.append(False)
                extracted_preds.append("")
                continue

            # 解析
            parsed_pred = parse(resp)
            parsed_gold = parse(f"${gt}$")
            
            # 记录提取出的答案以便后续分析
            extracted_preds.append(str(parsed_pred)) 

            # 验证
            is_correct = verify(parsed_gold, parsed_pred)
            validation_results.append(is_correct)
            
        except Exception as e:
            validation_results.append(False)
            extracted_preds.append("PARSE_ERROR")
            
    return validation_results, extracted_preds

def main():
    args = parse_args()
    file_path = args.input
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    correct_file = os.path.join(dir_name, f"{base_name}_correct.parquet")
    wrong_file = os.path.join(dir_name, f"{base_name}_wrong.parquet")

    print(f"🚀 开始处理文件: {file_path}")
    print(f"📂 正确样本将保存至: {correct_file}")
    print(f"📂 错误样本将保存至: {wrong_file}")

    try:
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        # 初始化统计
        total_count = 0
        correct_count = 0
        
        # 初始化 Parquet Writers
        writer_correct = None
        writer_wrong = None
        
        with tqdm(total=total_rows, unit="rows") as pbar:
            for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
                df_batch = batch.to_pandas()
                
                # 1. 验证并获取结果
                is_correct_list, extracted_answers = process_batch(df_batch)
                
                # 2. 将结果添加到 DataFrame 中（方便后续分析错误原因）
                df_batch['extracted_answer'] = extracted_answers
                df_batch['is_correct'] = is_correct_list
                
                # 3. 分割 DataFrame
                df_correct = df_batch[df_batch['is_correct'] == True]
                df_wrong = df_batch[df_batch['is_correct'] == False]
                
                # 4. 写入 Correct 文件
                if not df_correct.empty:
                    table_correct = pa.Table.from_pandas(df_correct)
                    if writer_correct is None:
                        writer_correct = pq.ParquetWriter(correct_file, table_correct.schema)
                    writer_correct.write_table(table_correct)

                # 5. 写入 Wrong 文件
                if not df_wrong.empty:
                    table_wrong = pa.Table.from_pandas(df_wrong)
                    if writer_wrong is None:
                        writer_wrong = pq.ParquetWriter(wrong_file, table_wrong.schema)
                    writer_wrong.write_table(table_wrong)
                
                # 6. 更新统计
                batch_correct_num = df_correct.shape[0]
                batch_total_num = df_batch.shape[0]
                
                total_count += batch_total_num
                correct_count += batch_correct_num
                
                pbar.update(batch_total_num)
                pbar.set_postfix({"Acc": f"{correct_count/total_count:.2%}" if total_count > 0 else "0%"})

        # 关闭 Writers
        if writer_correct: writer_correct.close()
        if writer_wrong: writer_wrong.close()

        # 最终报告
        print("\n" + "=" * 30)
        print("✅ 处理完成")
        print("=" * 30)
        print(f"总样本数 : {total_count}")
        print(f"正确样本 : {correct_count} -> 已保存至 {os.path.basename(correct_file)}")
        print(f"错误样本 : {total_count - correct_count} -> 已保存至 {os.path.basename(wrong_file)}")
        if total_count > 0:
            print(f"最终准确率: {correct_count / total_count:.4f} ({correct_count / total_count:.2%})")
        print("=" * 30)

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
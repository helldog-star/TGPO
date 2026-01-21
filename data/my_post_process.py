import argparse
import os

import pandas as pd

THINK_PREFIX = "<think>\n"


def remove_think_prefix(target_list):
    """移除 target 中 content 前的 <think>\\n 前缀"""
    if not target_list or not isinstance(target_list, list):
        return target_list
    for i, item in enumerate(target_list):
        if isinstance(item, dict) and "content" in item:
            c = item["content"]
            if isinstance(c, str) and c.startswith(THINK_PREFIX):
                target_list[i] = {**item, "content": c[len(THINK_PREFIX) :]}
    return target_list


def parse_args():
    p = argparse.ArgumentParser(description="去掉 target 中每条 content 前的 <think>\\n 前缀")
    p.add_argument("--input", required=True, help="输入 parquet 路径（一般为 verify 输出的 _correct.parquet）")
    p.add_argument(
        "--output",
        default=None,
        help="输出 parquet 路径；默认在输入同目录下，文件名为 {stem}_nothink.parquet",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        d = os.path.dirname(input_path)
        stem = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(d, f"{stem}_nothink.parquet")

    df = pd.read_parquet(input_path)
    df["target"] = df["target"].apply(remove_think_prefix)
    d = os.path.dirname(output_path)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"处理完毕，已保存到：{output_path}")


if __name__ == "__main__":
    main()
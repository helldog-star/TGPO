import argparse

import pandas as pd
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(description="将 OpenR1-Math 等数据集转为 parquet")
    p.add_argument(
        "--dataset",
        default="Elliott/Openr1-Math-46k-8192",
        help="数据集路径：HuggingFace  repo（如 Elliott/Openr1-Math-46k-8192）或本地目录",
    )
    p.add_argument("--split", default="train", help="split 名称")
    p.add_argument("--output", default="../data/openr1.parquet", help="输出 parquet 路径")
    return p.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    print(dataset[0])

    ret_dict = [item for item in dataset]
    train_df = pd.DataFrame(ret_dict)
    train_df.to_parquet(args.output)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
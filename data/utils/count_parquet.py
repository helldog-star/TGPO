import pandas as pd
row_count = pd.read_parquet('/mnt/nvme3/liuxinyu/TGPO/data/openr1.a3b_correct_35k.parquet', columns=[]).shape[0]
print(f"总行数: {row_count}")
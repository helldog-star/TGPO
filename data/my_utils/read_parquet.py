import pyarrow.parquet as pq

# file_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k.parquet'
# file_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.parquet'
# file_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/FMG/liuxinyu67/LUFFY/data/openr1.a3b_correct_35k_nothink.parquet'
file_path = "/Users/liuxinyu/Desktop/yuyucodebox/luffy/data/openr1.a3b_correct_35k.parquet"

try:
    # 创建 ParquetFile 对象，不立即加载数据
    parquet_file = pq.ParquetFile(file_path)
    
    # 只读取第一个行组（Row Group）或第一批次
    for batch in parquet_file.iter_batches(batch_size=1):
        first_row = batch.to_pandas().iloc[0]
        print(first_row.to_dict())
        break # 读到一条就退出
except Exception as e:
    print(f"读取失败: {e}")
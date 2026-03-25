import pandas as pd
import os
from config import Config

# 配置路径
INPUT_DIR = Config.GRAPHRAG_OUTPUT_DIR
OUTPUT_CSV = "dify_text_units.csv"

def convert_for_dify():
    path = os.path.join(INPUT_DIR, "text_units.parquet")
    if not os.path.exists(path):
        print("❌ 找不到 text_units.parquet")
        return

    # 读取 Parquet
    df = pd.read_parquet(path)
    
    # Dify 导入 CSV 支持两列：'问题' (可选) 和 '答案' (必须，即文本内容)
    # 或者自定义列。这里我们做成包含 ID 的格式，方便溯源。
    
    # 构造导出数据
    # 我们把 ID 拼接到文本前面，或者只保留文本。
    # 建议格式：
    # content (文本内容)
    # source_id (原本的 TextUnit ID)
    
    export_df = pd.DataFrame()
    export_df['content'] = df['text']
    export_df['source_id'] = df['id']
    
    # 导出 CSV
    export_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print(f"✅ 转换完成！请将 {OUTPUT_CSV} 上传到 Dify 知识库。")
    print(f"   共 {len(export_df)} 条文本块。")

if __name__ == "__main__":
    convert_for_dify()
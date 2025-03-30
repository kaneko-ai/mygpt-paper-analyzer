import os
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import json

# ✅ パス設定（本番環境向け）
DATA_PATH = "data/cd73_papers_scored.csv"
INDEX_PATH = "embeddings/cd73_faiss.index"
METADATA_PATH = "embeddings/cd73_metadata.csv"
TEXT_DIR = "texts/"
MODEL_NAME = "all-MiniLM-L6-v2"

# ✅ モデルとスプリッター初期化
model = SentenceTransformer(MODEL_NAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# ✅ スコア付きCSV読み込み
df = pd.read_csv(DATA_PATH)
df = df.sort_values(by="UnifiedScore", ascending=False).reset_index(drop=True)

# ✅ テキストチャンクとメタデータ準備
texts = []
metadata = []

for idx, row in df.iterrows():
    base_text = row.get("FullText") or row.get("Abstract") or ""
    chunks = splitter.split_text(base_text)
    for chunk in chunks:
        texts.append(chunk)
        metadata.append({"source": row["PMID"], "score": row["UnifiedScore"]})

# ✅ 埋め込みベクトル生成
doc_embeddings = model.encode(texts, show_progress_bar=True)

# ✅ FAISSインデックス作成
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# ✅ 保存処理（後のRAG検索で使用）
os.makedirs("embeddings", exist_ok=True)
pd.DataFrame(metadata).to_csv(METADATA_PATH, index=False)
faiss.write_index(index, INDEX_PATH)

# ✅ 出力完了
print(f"✅ チャンク数: {len(texts)} 件をFAISSに登録しました。")

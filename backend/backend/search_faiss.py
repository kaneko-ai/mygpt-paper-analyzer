# ✅ 完成版：FAISS類似チャンク検索 関数型モジュール（mygpt-paper-analyzer用）

import os
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ 設定（パスはプロジェクト構成に応じて変更）
INDEX_PATH = "embeddings/cd73_faiss.index"
METADATA_PATH = "embeddings/cd73_metadata.csv"
TEXTS_DIR = "texts/"
MODEL_NAME = "all-MiniLM-L6-v2"

# ✅ モデルの準備（埋め込みモデル）
model = SentenceTransformer(MODEL_NAME)

# ✅ チャンク再構築用スプリッター（LangChain）
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def search_chunks(query: str, k: int = 5) -> list[dict]:
    """
    質問に類似するチャンクをFAISSベクトル検索で返す。
    Parameters:
        query (str): 質問文（自然文）
        k (int): 上位k件の類似チャンクを返す（デフォルト：5）
    Returns:
        List[dict]: 類似チャンク情報（rank, similarity, source, chunk）を含む辞書のリスト
    """
    # 1. FAISSとメタ情報をロード
    index = faiss.read_index(INDEX_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # 2. テキスト読み込み＆チャンク分割（順番維持）
    all_chunks = []
    for fname in os.listdir(TEXTS_DIR):
        with open(os.path.join(TEXTS_DIR, fname), "r", encoding="utf-8") as f:
            full_text = f.read()
        chunks = splitter.split_text(full_text)
        all_chunks.extend(chunks)

    # 3. クエリをベクトル化し、FAISS検索
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k)

    # 4. 結果を整理して返す
    results = []
    for rank, (idx, dist) in enumerate(zip(I[0], D[0]), start=1):
        similarity = 1 / (1 + dist)
        results.append({
            "rank": rank,
            "similarity": round(similarity, 4),
            "source": metadata.iloc[idx]["source"],
            "chunk": all_chunks[idx].strip()
        })

    return results

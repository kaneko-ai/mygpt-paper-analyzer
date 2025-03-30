from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running!"}
# Your FastAPI code here
from fastapi import Request
from search_faiss import search_chunks  # ← 追加

@app.get("/faiss_search")
def faiss_search(query: str, k: int = 5):
    """
    クエリに基づいてFAISSベースの類似チャンクを返すエンドポイント
    例: /faiss_search?query=CD73の役割&k=5
    """
    results = search_chunks(query=query, k=k)
    return {"query": query, "results": results}

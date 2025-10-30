import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm  # 用于进度条

# === 初始化 ===
client = PersistentClient(path="./chroma")
collection = client.get_or_create_collection("medrag")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")  # ✅ 使用GPU

# === 批量读取数据 ===
texts, ids, metadatas = [], [], []

with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        texts.append(rec["Diseases"]+rec["text"])
        ids.append(rec["id"])
        metadatas.append({
            "Diseases": rec["Diseases"],
            "source": rec["source"],
            "url": rec["url"]
        })

# === 批量生成嵌入 ===
print(f"Encoding {len(texts)} documents...")
embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=True).tolist()

# === 批量写入 Chroma ===
print("Adding embeddings to Chroma...")
collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

print("✅ All documents processed and stored successfully!")

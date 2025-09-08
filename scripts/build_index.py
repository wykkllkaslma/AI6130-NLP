import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path

client = PersistentClient(path="./chroma")
collection = client.get_or_create_collection("medrag")
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        rec = json.loads(line)
        emb = embedder.encode(rec["text"]).tolist()
        collection.add(
            ids=[rec["id"]],
            embeddings=[emb],
            metadatas=[{
                "parent": rec["parent"],
                "source": rec["source"],
                "url": rec["url"]
            }],
            documents=[rec["text"]],
        )
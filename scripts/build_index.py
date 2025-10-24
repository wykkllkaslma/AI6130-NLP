import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Initialize ChromaDB client with persistent storage
client = PersistentClient(path="./chroma")

# Get or create collection for storing medical documents
collection = client.get_or_create_collection("medrag")

# Initialize sentence transformer model for text embedding
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# Read and process documents from JSONL file
with open("data/chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        # Parse JSON record from each line
        rec = json.loads(line)
        
        # Generate embedding vector for document text
        emb = embedder.encode(rec["text"]).tolist()
        
        # Add document to collection with metadata
        collection.add(
            ids=[rec["id"]],  # Unique identifier for each document
            embeddings=[emb],  # Embedding vector
            metadatas=[{
                "parent": rec["parent"],  # Parent document ID
                "source": rec["source"],  # Source of the document
                "url": rec["url"]         # URL reference
            }],
            documents=[rec["text"]],      # Original document text
        )
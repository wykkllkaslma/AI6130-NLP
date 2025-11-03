
import json
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from pathlib import Path
from tqdm import tqdm

# Initialize ChromaDB client with persistent storage
client = PersistentClient(path="./chroma")

# Get or create collection for storing medical documents
collection = client.get_or_create_collection("medrag")

# Initialize sentence transformer model for text embedding
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Building index from chunks.jsonl...")

# Batch processing for better performance
batch_size = 100
batch_ids = []
batch_texts = []
batch_metadatas = []

# Read and process documents from JSONL file
with open("data/chunks.jsonl", encoding="utf-8") as f:
    lines = f.readlines()
    
for line in tqdm(lines, desc="Processing chunks"):
    # Parse JSON record from each line
    rec = json.loads(line)
    
    # Add to batch
    batch_ids.append(rec["id"])
    batch_texts.append(rec["text"])
    batch_metadatas.append({
        "parent": rec["parent"],
        "source": rec["source"],
        "url": rec["url"]
    })
    
    # When batch is full, process it
    if len(batch_ids) >= batch_size:
        # Generate embeddings for batch
        embeddings = embedder.encode(batch_texts, show_progress_bar=False)
        
        # Add batch to collection
        collection.add(
            ids=batch_ids,
            embeddings=embeddings.tolist(),
            metadatas=batch_metadatas,
            documents=batch_texts
        )
        
        # Clear batch
        batch_ids = []
        batch_texts = []
        batch_metadatas = []

# Process remaining items
if batch_ids:
    embeddings = embedder.encode(batch_texts, show_progress_bar=False)
    collection.add(
        ids=batch_ids,
        embeddings=embeddings.tolist(),
        metadatas=batch_metadatas,
        documents=batch_texts
    )

print(f"Index built successfully! Total chunks: {len(lines)}")

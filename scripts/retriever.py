from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

client = PersistentClient(path="./chroma")

try:
    coll = client.get_collection("medrag")
except Exception:
    coll = client.create_collection("medrag")

embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def retrieve(query, k=20, topn=5):
    q_emb = embedder.encode(query).tolist()
    results = coll.query(query_embeddings=[q_emb], n_results=k)
    docs = list(zip(results["documents"][0], results["metadatas"][0]))
    pairs = [(query, d[0]) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:topn]

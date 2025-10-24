from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize ChromaDB client with persistent storage
client = PersistentClient(path="./chroma")

# Get or create collection for storing document embeddings
try:
    coll = client.get_collection("medrag")
except Exception:
    coll = client.create_collection("medrag")

# Initialize models for embedding generation and reranking
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")  # For generating document/query embeddings
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # For reranking results

def retrieve(query, k=20, topn=5):
    """
    Retrieve and rerank relevant documents for a given query
    
    Args:
        query (str): User's question or search query
        k (int): Number of initial candidates to retrieve
        topn (int): Number of final results to return after reranking
        
    Returns:
        list: Top-N documents with their metadata and relevance scores
    """
    # Generate embedding vector for query
    q_emb = embedder.encode(query).tolist()
    
    # Retrieve initial candidates using vector similarity
    results = coll.query(query_embeddings=[q_emb], n_results=k)
    
    # Combine documents with their metadata
    docs = list(zip(results["documents"][0], results["metadatas"][0]))
    
    # Prepare query-document pairs for reranking
    pairs = [(query, d[0]) for d in docs]
    
    # Rerank candidates using cross-encoder model
    scores = reranker.predict(pairs)
    
    # Sort documents by reranking scores
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    
    # Return top-N results
    return ranked[:topn]

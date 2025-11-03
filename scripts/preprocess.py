from pathlib import Path
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Define input and output paths
DATA = Path("./data/normalized")  # Directory containing normalized data files
OUT = Path("./data/chunks.jsonl") # Output file for text chunks

# Initialize tokenizer for text chunking
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunk_text(text, max_tokens=500, overlap=100):
    """
    Split text into overlapping chunks based on token count.

    Args:
        text (str): Input text to be chunked
        max_tokens (int): Maximum number of tokens per chunk
        overlap (int): Number of overlapping tokens between chunks

    Yields:
        str: Text chunks of specified maximum token length
    """
    tokens = tokenizer.encode(text)
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = tokens[start:end]
        yield tokenizer.decode(chunk)
        # Move start index by (max_tokens - overlap)
        start += max_tokens - overlap

def main():
    """
    Main function to process normalized documents into chunks
    """
    # Open output file for writing chunks
    with OUT.open("w", encoding="utf-8") as f_out:
        # Process each JSONL file in the data directory
        for path in DATA.glob("*.jsonl"):
            # Read and process each document
            for line in open(path, encoding="utf-8"):
                # Parse document JSON
                doc = json.loads(line)
                # Split document text into chunks
                for i, chunk in enumerate(chunk_text(doc["text"], max_tokens=500, overlap=0)):
                    # Write chunk with metadata to output file
                    f_out.write(json.dumps({
                        "id": f"{doc['id']}:{i}",      # Unique chunk ID
                        "parent": doc["id"],           # Original document ID
                        "text": chunk,                 # Chunk text content
                        "source": doc["source"],       # Data source
                        "url": doc["url"],            # Reference URL
                        "provenance": doc["provenance"], # Additional metadata
                    }, ensure_ascii=False) + "\n")

if __name__ == "__main__":

    main()

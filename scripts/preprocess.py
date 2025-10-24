from pathlib import Path
import json
from tqdm import tqdm
from transformers import AutoTokenizer

# Define input and output paths
DATA = Path("./data/normalized")  # Directory containing normalized data files
OUT = Path("./data/chunks.jsonl") # Output file for text chunks

# Initialize tokenizer for text chunking
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunk_text(text, max_tokens=500):
    """
    Split text into chunks based on token count
    
    Args:
        text (str): Input text to be chunked
        max_tokens (int): Maximum number of tokens per chunk
        
    Yields:
        str: Text chunks of specified maximum token length
    """
    # Encode text into tokens
    tokens = tokenizer.encode(text)
    # Generate chunks using sliding window
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i+max_tokens])

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
                for i, chunk in enumerate(chunk_text(doc["text"])):
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
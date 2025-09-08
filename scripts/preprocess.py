from pathlib import Path
import json
from tqdm import tqdm
from transformers import AutoTokenizer

DATA = Path("./data/normalized")
OUT = Path("./data/chunks.jsonl")
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")

def chunk_text(text, max_tokens=500):
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i+max_tokens])

def main():
    with OUT.open("w", encoding="utf-8") as f_out:
        for path in DATA.glob("*.jsonl"):
            for line in open(path, encoding="utf-8"):
                doc = json.loads(line)
                for i, chunk in enumerate(chunk_text(doc["text"])):
                    f_out.write(json.dumps({
                        "id": f"{doc['id']}:{i}",
                        "parent": doc["id"],
                        "text": chunk,
                        "source": doc["source"],
                        "url": doc["url"],
                        "provenance": doc["provenance"],
                    }, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
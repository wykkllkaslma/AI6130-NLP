import os, json
from pathlib import Path
import httpx
from lxml import etree
from tqdm import tqdm

# Define data directory and file paths
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
NORM = DATA_DIR / "normalized" / "openfda.jsonl"
OUT = DATA_DIR / "normalized" / "dailymed.jsonl"

# DailyMed API endpoint for fetching SPL (Structured Product Labeling) XML
DM_SPL = "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/{}.xml"  # v2 returns XML for set_id


def extract_text(xml_bytes: bytes):
    """
    Extract human-readable text from XML content
    
    Args:
        xml_bytes (bytes): Raw XML content
        
    Returns:
        str: Concatenated text from all text nodes
    """
    root = etree.fromstring(xml_bytes)
    # Very light extraction: concatenate all human-readable text nodes
    texts = root.xpath("//text()")
    return " ".join(t.strip() for t in texts if isinstance(t, str) and t.strip())


def main():
    """
    Main function to enrich drug data with DailyMed content
    """
    # Open output file for writing
    out = OUT.open("w", encoding="utf-8")
    # Initialize HTTP client with timeout
    client = httpx.Client(timeout=30)
    
    # Process each line from input JSONL file
    with open(NORM, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="DailyMed enrich"):
            # Parse JSON record
            doc = json.loads(line)
            # Extract set_id from provenance data
            set_id = (doc.get("provenance") or {}).get("set_id")
            if not set_id:
                continue
                
            # Fetch XML content from DailyMed API
            url = DM_SPL.format(set_id)
            try:
                r = client.get(url)
                r.raise_for_status()
            except Exception:
                continue
                
            # Extract and truncate text content
            txt = extract_text(r.content)[:2_000_000]
            
            # Write enriched document to output file
            out.write(json.dumps({
                "id": f"dailymed:{set_id}",
                "source": "dailymed",
                "title": doc.get("title"),
                "text": txt,
                "url": f"https://dailymed.nlm.nih.gov/dailymed/lookup.cfm?setid={set_id}",
                "date": doc.get("date"),
                "drug_names": doc.get("drug_names", []),
                "sections": {},
                "provenance": {"set_id": set_id},
            }, ensure_ascii=False) + "\n")
    out.close()

if __name__ == "__main__":
    main()
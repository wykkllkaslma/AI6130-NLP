import os, json, time
from pathlib import Path
import httpx
from urllib.parse import urlencode
from tqdm import tqdm

# OpenFDA API configuration
OPENFDA_ENDPOINT = "https://api.fda.gov/drug/label.json"
API_KEY = os.getenv("OPENFDA_API_KEY", "")

# Define data directories
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DIR = DATA_DIR / "raw" / "openfda"
NORM_DIR = DATA_DIR / "normalized"
RAW_DIR.mkdir(parents=True, exist_ok=True)
NORM_DIR.mkdir(parents=True, exist_ok=True)

# Define fields to retrieve from OpenFDA API
FIELDS = [
    "id", "effective_time", "set_id", "openfda.brand_name", "openfda.generic_name",
    "openfda.substance_name", "openfda.product_type", "indications_and_usage",
    "dosage_and_administration", "contraindications", "warnings", "adverse_reactions"
]


def fetch_openfda(query: str, limit=100, max_pages=3):
    """
    Fetch drug data from OpenFDA API with pagination
    
    Args:
        query (str): Search query for OpenFDA API
        limit (int): Number of results per page
        max_pages (int): Maximum number of pages to fetch
        
    Returns:
        list: Collection of drug label data
    """
    client = httpx.Client(timeout=30)
    results = []
    for page in range(max_pages):
        # Build query parameters
        params = {
            "search": query,  # e.g., 'openfda.generic_name:"ibuprofen"'
            "limit": limit,
            "skip": page * limit,
            "sort": "effective_time:desc",
        }
        if API_KEY:
            params["api_key"] = API_KEY
            
        # Make API request
        url = f"{OPENFDA_ENDPOINT}?{urlencode(params)}"
        r = client.get(url)
        if r.status_code == 404:
            break
        r.raise_for_status()
        
        # Process response data
        data = r.json()
        items = data.get("results", [])
        if not items:
            break
        results.extend(items)
        time.sleep(0.2)  # Rate limiting
    return results


def normalize(item):
    """
    Normalize OpenFDA drug label data into consistent format
    
    Args:
        item (dict): Raw drug label data from OpenFDA
        
    Returns:
        dict: Normalized drug data structure
    """
    of = item.get("openfda", {})
    # Extract and format text sections
    sections = {}
    for k in [
        "indications_and_usage", "dosage_and_administration", "contraindications",
        "warnings", "adverse_reactions"]:
        val = item.get(k)
        if isinstance(val, list):
            sections[k] = "\n\n".join(val)
        elif isinstance(val, str):
            sections[k] = val
            
    # Generate unique document ID
    doc_id = item.get("id") or item.get("set_id")
    
    # Return normalized document structure
    return {
        "id": f"openfda:{doc_id}",
        "source": "openfda",
        "title": ", ".join(of.get("brand_name", of.get("generic_name", []))) or of.get("generic_name", ["Unknown"])[0],
        "text": "\n\n".join(v for v in sections.values() if v),
        "url": None,  # Will be added later using DailyMed URL with set_id
        "date": item.get("effective_time"),
        "drug_names": list(set(of.get("brand_name", []) + of.get("generic_name", []) + of.get("substance_name", []))),
        "sections": sections,
        "provenance": {
            "set_id": item.get("set_id"),
            "product_type": of.get("product_type"),
            "raw": item,
        },
    }


def main():
    """
    Main function to fetch and process drug data from OpenFDA
    """
    # Ensure directory structure exists
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    # Define queries for different drug types
    queries = [
        'openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
        'openfda.product_type:"HUMAN OTC DRUG"',
    ]
    
    # Process queries and write normalized data
    out_path = NORM_DIR / "openfda.jsonl"
    with out_path.open("w", encoding="utf-8") as f_out:
        for q in queries:
            # Fetch data for each query
            items = fetch_openfda(q, limit=100, max_pages=10)  # ~1000 docs per query
            # Normalize and save each item
            for it in tqdm(items, desc=f"normalize {q}"):
                f_out.write(json.dumps(normalize(it), ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
import os, json, time
from pathlib import Path
import httpx
from lxml import etree
from urllib.parse import urlencode
from tqdm import tqdm

# Define data directory and output file path
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUT = DATA_DIR / "normalized" / "pubmed.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

# PubMed E-utilities API configuration
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "medrag-demo"
EMAIL = os.getenv("CONTACT_EMAIL", "your_email@example.com")
API_KEY = os.getenv("NCBI_API_KEY", "")


def esearch(term: str, retmax=100):
    """
    Search PubMed database using E-utilities
    
    Args:
        term (str): Search query term
        retmax (int): Maximum number of results to return
        
    Returns:
        list: List of PubMed IDs (PMIDs)
    """
    params = {
        "db": "pubmed", "term": term, "retmode": "json", "retmax": retmax,
        "tool": TOOL, "email": EMAIL,
    }
    if API_KEY:
        params["api_key"] = API_KEY
    url = f"{EUTILS}/esearch.fcgi?{urlencode(params)}"
    r = httpx.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def efetch(pmids):
    """
    Fetch full article data for given PMIDs
    
    Args:
        pmids (list): List of PubMed IDs to fetch
        
    Returns:
        bytes: XML content containing article data
    """
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": TOOL, "email": EMAIL}
    if API_KEY:
        params["api_key"] = API_KEY
    url = f"{EUTILS}/efetch.fcgi?{urlencode(params)}"
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_pubmed(xml_bytes):
    """
    Parse PubMed XML and extract article information
    
    Args:
        xml_bytes (bytes): Raw XML content from PubMed
        
    Yields:
        dict: Normalized article data structure
    """
    root = etree.fromstring(xml_bytes)
    for art in root.xpath("//PubmedArticle"):
        # Extract basic article metadata
        pmid = (art.xpath("string(.//PMID)") or "").strip()
        ti = (art.xpath("string(.//ArticleTitle)") or "").strip()
        ab = (art.xpath("string(.//Abstract)") or "").strip()
        journal = (art.xpath("string(.//Journal/Title)") or "").strip()
        year = (art.xpath("string(.//JournalIssue/PubDate/Year)") or "").strip()
        
        # Extract DOI if available
        doi = None
        for aid in art.xpath(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break
                
        # Yield normalized document structure
        yield {
            "id": f"pubmed:{pmid}",
            "source": "pubmed",
            "title": ti,
            "text": ab,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "date": year,
            "drug_names": [],
            "sections": {},
            "provenance": {"pmid": pmid, "journal": journal, "doi": doi},
        }


def main():
    """
    Main function to fetch and process articles from PubMed
    """
    # Create output directory if needed
    OUT.parent.mkdir(parents=True, exist_ok=True)
    
    # Example queries to seed corpus; expand per your formularies
    queries = [
        "ibuprofen AND dosage",
        "empagliflozin AND kidney disease",
        "amoxicillin AND dosing AND renal impairment",
    ]
    
    # Process each query and save results
    with OUT.open("w", encoding="utf-8") as f:
        for q in queries:
            # Search PubMed for articles
            ids = esearch(q, retmax=100)
            if not ids:
                continue
            # Fetch and parse full article data
            xml = efetch(ids)
            for rec in parse_pubmed(xml):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
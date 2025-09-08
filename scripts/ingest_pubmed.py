import os, json, time
from pathlib import Path
import httpx
from lxml import etree
from urllib.parse import urlencode
from tqdm import tqdm

DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
OUT = DATA_DIR / "normalized" / "pubmed.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "medrag-demo"
EMAIL = os.getenv("CONTACT_EMAIL", "your_email@example.com")
API_KEY = os.getenv("NCBI_API_KEY", "")


def esearch(term: str, retmax=100):
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
    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "tool": TOOL, "email": EMAIL}
    if API_KEY:
        params["api_key"] = API_KEY
    url = f"{EUTILS}/efetch.fcgi?{urlencode(params)}"
    r = httpx.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def parse_pubmed(xml_bytes):
    root = etree.fromstring(xml_bytes)
    for art in root.xpath("//PubmedArticle"):
        pmid = (art.xpath("string(.//PMID)") or "").strip()
        ti = (art.xpath("string(.//ArticleTitle)") or "").strip()
        ab = (art.xpath("string(.//Abstract)") or "").strip()
        journal = (art.xpath("string(.//Journal/Title)") or "").strip()
        year = (art.xpath("string(.//JournalIssue/PubDate/Year)") or "").strip()
        doi = None
        for aid in art.xpath(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break
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
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Example queries to seed corpus; expand per your formularies
    queries = [
        "ibuprofen AND dosage",
        "empagliflozin AND kidney disease",
        "amoxicillin AND dosing AND renal impairment",
    ]
    with OUT.open("w", encoding="utf-8") as f:
        for q in queries:
            ids = esearch(q, retmax=100)
            if not ids:
                continue
            xml = efetch(ids)
            for rec in parse_pubmed(xml):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
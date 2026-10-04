import os
import json
import time
import requests
import re
from typing import Dict, List, Any
from bs4 import BeautifulSoup

HEADERS = {
    'User-Agent': 'EnterpriseKnowledgeAssistant research@university.edu'
}

TARGET_COMPANIES = {
    'AAPL': {'name': 'Apple Inc.', 'industry': 'Technology'},
    'MSFT': {'name': 'Microsoft Corp', 'industry': 'Technology'},
    'NVDA': {'name': 'NVIDIA Corp', 'industry': 'Technology'},
    'WMT': {'name': 'Walmart Inc', 'industry': 'Retail'},
    'AMZN': {'name': 'Amazon.com Inc', 'industry': 'Retail'},
    'NKE': {'name': 'NIKE Inc', 'industry': 'Retail'},
    'GE': {'name': 'General Electric Co', 'industry': 'Industrials'},
    'CAT': {'name': 'Caterpillar Inc', 'industry': 'Industrials'},
    'XOM': {'name': 'Exxon Mobil Corp', 'industry': 'Energy'},
    'TSLA': {'name': 'Tesla Inc', 'industry': 'Automotive'}
}

TARGET_YEARS = [2021, 2022, 2023]
TARGET_FORMS = ['10-K', '10-Q']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw_filings')
METADATA_FILE = os.path.join(RAW_DATA_DIR, 'corpus_metadata.json')


def get_cik_map() -> Dict[str, str]:
    """Fetch mapping of tickers to CIKs from SEC EDGAR."""
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ticker_to_cik = {}
        for idx, entry in data.items():
            ticker = entry['ticker'].upper()
            cik = str(entry['cik_str']).zfill(10)
            ticker_to_cik[ticker] = cik
        return ticker_to_cik
    except Exception as e:
        print(f"[Warning] Failed to fetch live CIK map: {e}")
        # Hardcoded fallback map for reliability
        return {
            'AAPL': '0000320193',
            'MSFT': '0000789019',
            'NVDA': '0001045810',
            'WMT': '0000104169',
            'AMZN': '0001018724',
            'NKE': '0000320187',
            'GE': '0000040545',
            'CAT': '0000018255',
            'XOM': '0000034088',
            'TSLA': '0001318605'
        }


def fetch_company_submissions(cik: str) -> Dict[str, Any]:
    """Fetch company submissions metadata JSON from SEC EDGAR API."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    time.sleep(0.12)  # Respect SEC EDGAR rate limit (<10 requests/sec)
    resp = requests.get(url, headers=HEADERS, timeout=12)
    resp.raise_for_status()
    return resp.json()


def clean_html_content(raw_content: str) -> str:
    """Extract readable clean text from HTML while preserving section titles and basic structure."""
    if not raw_content or not raw_content.strip():
        return ""
    
    soup = BeautifulSoup(raw_content, 'html.parser')
    
    # Remove script and style elements
    for element in soup(["script", "style", "noscript", "header", "footer"]):
        element.decompose()
        
    # Extract text with linebreaks for formatting
    text = soup.get_text(separator="\n")
    
    # Normalize multiple newlines and spaces
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)
    return cleaned_text


def download_filing_document(cik: str, accession_number: str, primary_doc: str) -> str:
    """Download primary filing document text or html from SEC archives."""
    acc_no_clean = accession_number.replace('-', '')
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_clean}/{primary_doc}"
    time.sleep(0.12)
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.text


def acquire_corpus() -> List[Dict[str, Any]]:
    """Download raw filings and metadata for target companies and years."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    cik_map = get_cik_map()
    
    corpus_metadata = []
    
    print("=" * 60)
    print("PHASE 0: SEC EDGAR DATA ACQUISITION")
    print("=" * 60)
    
    for ticker, info in TARGET_COMPANIES.items():
        cik = cik_map.get(ticker)
        if not cik:
            print(f"[Error] CIK not found for {ticker}")
            continue
            
        print(f"\nProcessing {ticker} ({info['name']}) - CIK: {cik}")
        try:
            sub_data = fetch_company_submissions(cik)
            recent_filings = sub_data.get('filings', {}).get('recent', {})
            
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_docs = recent_filings.get('primaryDocument', [])
            report_dates = recent_filings.get('reportDate', [])
            
            fetched_for_company = 0
            
            for i in range(len(forms)):
                form = forms[i]
                if form not in TARGET_FORMS:
                    continue
                    
                filing_date = filing_dates[i]
                report_date = report_dates[i] if i < len(report_dates) else filing_date
                year = int(report_date.split('-')[0]) if report_date else int(filing_date.split('-')[0])
                
                if year not in TARGET_YEARS:
                    continue
                    
                accession_no = accession_numbers[i]
                primary_doc = primary_docs[i]
                
                # Limit to 1 10-K per year and up to 1 10-Q per year to keep corpus high quality & manageable
                doc_filename = f"{ticker}_{form.replace('-', '')}_{year}_{accession_no}.txt"
                doc_path = os.path.join(RAW_DATA_DIR, doc_filename)
                
                if os.path.exists(doc_path):
                    with open(doc_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text_content = f.read()
                else:
                    try:
                        raw_html = download_filing_document(cik, accession_no, primary_doc)
                        text_content = clean_html_content(raw_html)
                        with open(doc_path, 'w', encoding='utf-8') as f:
                            f.write(text_content)
                        print(f"  Downloaded: {form} for {year} ({doc_filename}) - {len(text_content)} chars")
                    except Exception as err:
                        print(f"  [Failed download] {ticker} {form} {year}: {err}")
                        continue
                        
                word_count = len(text_content.split())
                char_count = len(text_content)
                estimated_tokens = int(word_count * 1.3)
                
                meta_entry = {
                    'ticker': ticker,
                    'company_name': info['name'],
                    'industry': info['industry'],
                    'cik': cik,
                    'form_type': form,
                    'fiscal_year': year,
                    'filing_date': filing_date,
                    'report_date': report_date,
                    'accession_number': accession_no,
                    'primary_document': primary_doc,
                    'file_path': doc_path,
                    'file_name': doc_filename,
                    'word_count': word_count,
                    'token_count': estimated_tokens,
                    'char_count': char_count
                }
                corpus_metadata.append(meta_entry)
                fetched_for_company += 1
                
                # Max 3 filings per company (e.g. FY2021 10-K, FY2022 10-K, FY2023 10-K)
                if fetched_for_company >= 3:
                    break
                    
        except Exception as e:
            print(f"[Error] Could not process submissions for {ticker}: {e}")
            
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(corpus_metadata, f, indent=2)
        
    report_corpus_stats(corpus_metadata)
    return corpus_metadata


def report_corpus_stats(metadata: List[Dict[str, Any]]):
    """Generate and display summary corpus statistics."""
    total_filings = len(metadata)
    if total_filings == 0:
        print("\nNo filings downloaded.")
        return
        
    total_words = sum(m['word_count'] for m in metadata)
    total_tokens = sum(m['token_count'] for m in metadata)
    companies = sorted(list(set(m['ticker'] for m in metadata)))
    industries = sorted(list(set(m['industry'] for m in metadata)))
    forms = sorted(list(set(m['form_type'] for m in metadata)))
    
    min_size = min(m['word_count'] for m in metadata)
    max_size = max(m['word_count'] for m in metadata)
    avg_size = total_words // total_filings
    
    print("\n" + "=" * 60)
    print("PHASE 0: CORPUS STATISTICS REPORT")
    print("=" * 60)
    print(f"Total Filings Downloaded: {total_filings}")
    print(f"Unique Companies ({len(companies)}): {', '.join(companies)}")
    print(f"Industries Covered ({len(industries)}): {', '.join(industries)}")
    print(f"Filing Form Types: {', '.join(forms)}")
    print(f"Total Corpus Words: {total_words:,}")
    print(f"Total Estimated Corpus Tokens: {total_tokens:,}")
    print(f"Doc Size Range (words): {min_size:,} min | {avg_size:,} avg | {max_size:,} max")
    print(f"Metadata File Saved To: {METADATA_FILE}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    acquire_corpus()

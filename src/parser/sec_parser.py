import os
import re
import json
from typing import Dict, List, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw_filings')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_chunks')
METADATA_FILE = os.path.join(RAW_DATA_DIR, 'corpus_metadata.json')
CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, 'all_chunks.json')

# Standard SEC Section Patterns
ITEM_PATTERNS = [
    (r'(?i)\bITEM\s+1\.\s+BUSINESS\b', 'Item 1. Business'),
    (r'(?i)\bITEM\s+1A\.\s+RISK\s+FACTORS\b', 'Item 1A. Risk Factors'),
    (r'(?i)\bITEM\s+1B\.\s+UNRESOLVED\s+STAFF\s+COMMENTS\b', 'Item 1B. Unresolved Staff Comments'),
    (r'(?i)\bITEM\s+2\.\s+PROPERTIES\b', 'Item 2. Properties'),
    (r'(?i)\bITEM\s+3\.\s+LEGAL\s+PROCEEDINGS\b', 'Item 3. Legal Proceedings'),
    (r'(?i)\bITEM\s+5\.\s+MARKET\s+FOR\s+REGISTRANT', 'Item 5. Market Equity'),
    (r'(?i)\bITEM\s+7\.\s+MANAGEMENT(?:\'S)?\s+DISCUSSION\s+AND\s+ANALYSIS', 'Item 7. MD&A'),
    (r'(?i)\bITEM\s+7A\.\s+QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES', 'Item 7A. Market Risk Disclosures'),
    (r'(?i)\bITEM\s+8\.\s+FINANCIAL\s+STATEMENTS', 'Item 8. Financial Statements'),
    (r'(?i)\bITEM\s+9A\.\s+CONTROLS\s+AND\s+PROCEDURES', 'Item 9A. Controls & Procedures'),
    (r'(?i)\bITEM\s+15\.\s+EXHIBITS', 'Item 15. Exhibits'),
]


def detect_sections(text: str) -> List[Tuple[str, int, int]]:
    """Detect boundaries of major SEC filing items/sections."""
    matches = []
    for pattern, name in ITEM_PATTERNS:
        for m in re.finditer(pattern, text):
            matches.append((m.start(), name))
            
    if not matches:
        return [("General Filing Content", 0, len(text))]
        
    matches.sort(key=lambda x: x[0])
    
    # Remove duplicate matches within 500 characters (e.g. Table of Contents references)
    filtered_matches = []
    for pos, name in matches:
        if not filtered_matches:
            filtered_matches.append((pos, name))
        else:
            prev_pos, prev_name = filtered_matches[-1]
            if pos - prev_pos > 500:
                filtered_matches.append((pos, name))
                
    sections = []
    for idx, (pos, name) in enumerate(filtered_matches):
        start_idx = pos
        end_idx = filtered_matches[idx + 1][0] if idx + 1 < len(filtered_matches) else len(text)
        sections.append((name, start_idx, end_idx))
        
    return sections


def is_table_line(line: str) -> bool:
    """Determine if a single line of text represents a financial table row."""
    # Check for presence of numbers, dollar signs, percentages, or columnar tabs
    numbers = re.findall(r'\b\$?\d+(?:,\d{3})*(?:\.\d+)?%?\b', line)
    if len(numbers) >= 2:
        return True
    if '|' in line and len(numbers) >= 1:
        return True
    return False


def extract_tables_and_prose(section_text: str) -> List[Dict[str, Any]]:
    """Separate tabular text blocks from narrative prose blocks."""
    lines = section_text.splitlines()
    blocks = []
    
    current_block_type = None
    current_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        is_table = is_table_line(stripped)
        block_type = "table" if is_table else "prose"
        
        if current_block_type is None:
            current_block_type = block_type
            current_lines.append(stripped)
        elif current_block_type == block_type:
            current_lines.append(stripped)
        else:
            # If switching from table to prose or vice-versa, save previous block if non-trivial
            if len(current_lines) >= 2 or current_block_type == "prose":
                blocks.append({
                    'type': current_block_type,
                    'text': "\n".join(current_lines)
                })
            current_block_type = block_type
            current_lines = [stripped]
            
    if current_lines:
        blocks.append({
            'type': current_block_type,
            'text': "\n".join(current_lines)
        })
        
    return blocks


def format_table_block(table_text: str) -> str:
    """Format raw table lines into structured key-value markdown table format."""
    lines = table_text.splitlines()
    formatted_lines = []
    formatted_lines.append("| Financial Item / Metric | Values / Data |")
    formatted_lines.append("| --- | --- |")
    
    for line in lines:
        parts = re.split(r'\s{2,}|\t|\|', line)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            label = parts[0]
            val = " | ".join(parts[1:])
            formatted_lines.append(f"| {label} | {val} |")
        else:
            formatted_lines.append(f"| {line} | - |")
            
    return "\n".join(formatted_lines)


def chunk_text_sliding_window(text: str, max_words: int = 350, overlap_words: int = 50) -> List[str]:
    """Perform sliding window chunking over words."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
        
    chunks = []
    step = max_words - overlap_words
    for i in range(0, len(words), step):
        chunk_words = words[i:i + max_words]
        if len(chunk_words) > 30:  # Ignore tiny residual fragments
            chunks.append(" ".join(chunk_words))
            
    return chunks


def process_all_filings() -> List[Dict[str, Any]]:
    """Parse all downloaded filings into section-aware prose and table chunks with rich metadata."""
    if not os.path.exists(METADATA_FILE):
        print(f"[Error] Corpus metadata not found at {METADATA_FILE}. Run Phase 0 acquisition first.")
        return []
        
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        corpus_meta = json.load(f)
        
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    all_chunks = []
    
    print("=" * 60)
    print("PHASE 1: DOCUMENT PROCESSING & SECTION-AWARE CHUNKING")
    print("=" * 60)
    
    for meta in corpus_meta:
        file_path = meta['file_path']
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
            
        sections = detect_sections(raw_text)
        print(f"\nProcessing {meta['ticker']} {meta['form_type']} {meta['fiscal_year']} - {len(sections)} sections detected")
        
        chunk_seq = 0
        for section_name, start_idx, end_idx in sections:
            section_text = raw_text[start_idx:end_idx].strip()
            if not section_text or len(section_text) < 100:
                continue
                
            blocks = extract_tables_and_prose(section_text)
            
            for block in blocks:
                block_type = block['type']
                block_content = block['text']
                
                if block_type == 'table':
                    formatted_content = format_table_block(block_content)
                    sub_chunks = chunk_text_sliding_window(formatted_content, max_words=450, overlap_words=50)
                else:
                    sub_chunks = chunk_text_sliding_window(block_content, max_words=350, overlap_words=50)
                    
                for sub_chunk in sub_chunks:
                    chunk_seq += 1
                    chunk_id = f"{meta['ticker']}_{meta['form_type'].replace('-', '')}_{meta['fiscal_year']}_{section_name.replace(' ', '_').replace('.', '')}_{chunk_seq:04d}"
                    word_count = len(sub_chunk.split())
                    token_est = int(word_count * 1.3)
                    
                    source_loc = f"{meta['company_name']} ({meta['ticker']}) {meta['fiscal_year']} {meta['form_type']} | {section_name}"
                    
                    chunk_meta = {
                        'chunk_id': chunk_id,
                        'ticker': meta['ticker'],
                        'company_name': meta['company_name'],
                        'industry': meta['industry'],
                        'form_type': meta['form_type'],
                        'fiscal_year': meta['fiscal_year'],
                        'report_date': meta['report_date'],
                        'section': section_name,
                        'chunk_type': block_type,
                        'source_location': source_loc,
                        'text': sub_chunk,
                        'word_count': word_count,
                        'token_count': token_est
                    }
                    all_chunks.append(chunk_meta)
                    
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
        
    report_chunking_stats(all_chunks)
    return all_chunks


def report_chunking_stats(chunks: List[Dict[str, Any]]):
    """Display chunking distribution statistics."""
    total_chunks = len(chunks)
    if total_chunks == 0:
        return
        
    prose_chunks = sum(1 for c in chunks if c['chunk_type'] == 'prose')
    table_chunks = sum(1 for c in chunks if c['chunk_type'] == 'table')
    
    total_tokens = sum(c['token_count'] for c in chunks)
    avg_tokens = total_tokens // total_chunks
    
    sections = set(c['section'] for c in chunks)
    
    print("\n" + "=" * 60)
    print("PHASE 1: CHUNKING DISTRIBUTION REPORT")
    print("=" * 60)
    print(f"Total Structural Chunks Created: {total_chunks:,}")
    print(f"Prose Chunks: {prose_chunks:,} ({prose_chunks/total_chunks*100:.1f}%)")
    print(f"Table Chunks: {table_chunks:,} ({table_chunks/total_chunks*100:.1f}%)")
    print(f"Average Token Count per Chunk: {avg_tokens} tokens")
    print(f"Unique Filing Sections Preserved: {len(sections)}")
    print(f"Processed Chunks File Saved To: {CHUNKS_FILE}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    process_all_filings()

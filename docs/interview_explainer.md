# 1-Page Interview Explainer: Enterprise SEC Knowledge Assistant

> **The One-Sentence Framing**:
> *"An assistant that answers financial questions about public companies by retrieving and citing the exact passages from their SEC filings (10-K annual reports, 10-Q quarterly reports), instead of relying on the model's memory — because financial numbers must be traceable to a source, not guessed."*

---

## 3 Core Technical Upgrades Over Naive RAG (And Their "Why")

### 1. Hybrid Retrieval (BM25 + Dense) with Reciprocal Rank Fusion & Reranking
- **Why**: Dense embeddings alone miss exact 4-digit fiscal years, tickers, and table line-items; BM25 keyword search catches exact numbers but misses semantic paraphrasing. Combining both via Reciprocal Rank Fusion (RRF) and Cross-Encoder reranking ensures both high recall and high precision.

### 2. Explicit Query Routing & Comparative Multi-Pass Decomposition
- **Why**: Single-pass top-k retrieval naturally favors the single most dominant entity in a query, starving other entities in comparative questions ("Compare Apple and Microsoft 2023 revenue"). Decomposing comparative queries into sub-queries guarantees context retrieval for every company and year.

### 3. Post-Generation Citation Entailment Verification
- **Why**: LLMs can still hallucinate numbers or cite the wrong section even when context is present. Running a post-generation numerical entailment check directly verifies that every figure in the answer exists in the cited filing chunk, flagging unsupported claims before serving the answer.

---

## Headline Benchmark Results

| Metric | Baseline Naive RAG | Advanced Hybrid RAG | Delta / Improvement |
| --- | --- | --- | --- |
| **Retrieval Recall@5** | 71.43% | **94.29%** | **+22.86%** |
| **Mean Reciprocal Rank (MRR)** | 0.534 | **0.842** | **+0.308** |
| **Out-of-Scope Refusal Accuracy** | 0.00% | **95.00%** | **+95.00%** |
| **Citation Faithfulness Rate** | Unchecked | **97.14%** | **Verified** |
| **Human Agreement Rate (35-sample)** | N/A | **94.30%** | **Validated** |

---

## The Honest Caveat (What These Numbers Don't Prove)

> **Interview Talking Point**:
> *"While our evaluation demonstrates a ~23% recall improvement and 95% refusal accuracy, these results are based on a curated 120-question benchmark with a single annotator across 9 companies and 3 fiscal years. In production, unseen financial terminology, complex footnotes, or scanned legacy PDFs would require further domain-specific parser tuning."*

---

## Key Citations & Methodological References
1. **Reciprocal Rank Fusion (RRF)**: Cormack, Clarke, & Buettcher (2009). *Reciprocal Rank Fusion Outperforms Information Retrieval Combinations.*
2. **BM25Okapi**: Robertson et al. (1995). *Okapi at TREC-3.*
3. **Cross-Encoder Reranking**: Nogueira & Cho (2019). *Passage Re-ranking with BERT.*

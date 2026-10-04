# Phase 8 — Failure Analysis Log

This running log documents 5 concrete failure modes observed during evaluation of the SEC Knowledge Assistant, along with failure examples, root cause diagnoses, and architectural mitigations.

---

## Failure Category 1: Fiscal Year Confusion in Dense Retrieval
- **Example Question**: *"What was Apple's total net sales in fiscal year 2022?"*
- **Observed Behavior**: The naive baseline dense retriever returned top chunks from Apple's FY2023 10-K report (Item 7) instead of the FY2022 filing.
- **One-Sentence Hypothesis**: Dense embeddings map semantic similarity for "Apple net sales" strongly across all years, missing exact 4-digit fiscal year tokens (`2022` vs `2023`).
- **Mitigation Applied**: Added BM25 keyword matching and explicit metadata year-boosting (+0.15 score boost) in the hybrid reranking stage (`src/rag/hybrid_rag.py`).

---

## Failure Category 2: Table Formatting Corruption / Multi-Column Disambiguation
- **Example Question**: *"What was Microsoft's R&D expense in 2023?"*
- **Observed Behavior**: Unstructured text chunking split a multi-year financial statement table between columns, associating 2022 figures with 2023 metrics.
- **One-Sentence Hypothesis**: Arbitrary character or token splitting cuts across table rows, destroying row-column alignment required for financial figures.
- **Mitigation Applied**: Implemented table structure preservation in `src/parser/sec_parser.py` that formats tabular lines into Markdown tables (`| Metric | Value |`) before chunking, tagged with `chunk_type: "table"`.

---

## Failure Category 3: Single-Pass Half-Retrieval for Comparative Queries
- **Example Question**: *"Compare Apple and Microsoft total revenues in fiscal year 2023."*
- **Observed Behavior**: A single top-5 retrieval pass returned 4 chunks for Apple and only 1 chunk for Microsoft, leading to an incomplete comparison.
- **One-Sentence Hypothesis**: Dense retrieval top-k naturally favors the entity with stronger token density in the query embedding space.
- **Mitigation Applied**: Built explicit Query Decomposition in `src/rag/query_router.py` that splits comparative queries into sub-queries per company, retrieves candidates independently for each entity, and aggregates contexts before synthesis.

---

## Failure Category 4: False Refusal on Paraphrased Financial Questions
- **Example Question**: *"How much cash did Tesla generate from operations in 2023?"*
- **Observed Behavior**: The initial keyword router flagged the query as out-of-scope because "cash generated from operations" did not match exact keywords like "operating cash flow".
- **One-Sentence Hypothesis**: Overly rigid keyword blacklists trigger false refusal on valid financial questions phrased in plain English.
- **Mitigation Applied**: Expanded financial keyword taxonomy in `src/rag/query_router.py` and added semantic fallback check before triggering refusal.

---

## Failure Category 5: Correct Document but Non-Financial Section Retrieval
- **Example Question**: *"What was NVIDIA's net income in 2023?"*
- **Observed Behavior**: Naive vector search retrieved Item 1A (Risk Factors) passages discussing potential net income risks rather than Item 8 (Financial Statements).
- **One-Sentence Hypothesis**: Risk factor prose contains dense mentions of financial terms ("net income", "revenue losses") that match dense semantic embeddings without providing actual primary financial figures.
- **Mitigation Applied**: Reranker metadata boost for `chunk_type: "table"` and Item 8 / Item 7 section tags when numeric financial queries are detected.

import os
import re
import json
import time
from enum import Enum
from typing import Dict, List, Any, Tuple

KNOWN_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'WMT', 'AMZN', 'NKE', 'GE', 'CAT', 'XOM', 'TSLA']
COMPANY_KEYWORDS = {
    'apple': 'AAPL', 'microsoft': 'MSFT', 'nvidia': 'NVDA', 'walmart': 'WMT',
    'amazon': 'AMZN', 'nike': 'NKE', 'general electric': 'GE', 'caterpillar': 'CAT',
    'exxon': 'XOM', 'tesla': 'TSLA'
}

OUT_OF_SCOPE_KEYWORDS = [
    'weather', 'sports', 'recipe', 'movie', 'song', 'capital of', 'president',
    'cricket', 'football', 'actor', 'celebrity', 'joke', 'poem', 'who is elon musk\'s wife',
    'favorite color', 'ice cream', 'game'
]

FINANCIAL_KEYWORDS = [
    'revenue', 'sales', 'net income', 'profit', 'margin', 'expense', 'cost', 'r&d',
    'operating income', 'ebitda', 'cash flow', 'balance sheet', 'asset', 'liability',
    'debt', 'eps', 'earnings', 'risk', 'business', 'growth', 'guidance', 'sec', '10-k', '10-q'
]


class QueryCategory(Enum):
    SINGLE_FACT = "single_fact"
    COMPARATIVE = "comparative"
    OUT_OF_SCOPE = "out_of_scope"


class QueryRouter:
    """Phase 4: Lightweight Query Classification, Decomposition, and Routing System."""

    def __init__(self, hybrid_rag_engine=None):
        self.rag_engine = hybrid_rag_engine

    def classify_query(self, query: str) -> Tuple[QueryCategory, Dict[str, Any]]:
        """Classify user query into single_fact, comparative, or out_of_scope."""
        query_lower = query.lower()
        query_upper = query.upper()

        # 1. Out of scope check
        is_financial = any(kw in query_lower for kw in FINANCIAL_KEYWORDS)
        matches_company = any(ticker in query_upper for ticker in KNOWN_TICKERS) or any(name in query_lower for name in COMPANY_KEYWORDS)
        
        has_out_of_scope_kw = any(kw in query_lower for kw in OUT_OF_SCOPE_KEYWORDS)

        if has_out_of_scope_kw or (not is_financial and not matches_company):
            return QueryCategory.OUT_OF_SCOPE, {'reason': 'Non-financial or non-SEC query topic.'}

        # 2. Extract companies mentioned
        found_companies = set()
        for ticker in KNOWN_TICKERS:
            if re.search(rf'\b{ticker}\b', query_upper):
                found_companies.add(ticker)
        for name, ticker in COMPANY_KEYWORDS.items():
            if name in query_lower:
                found_companies.add(ticker)

        # 3. Extract years mentioned
        found_years = re.findall(r'\b(2021|2022|2023|2024)\b', query)

        # 4. Determine if Comparative
        is_multi_company = len(found_companies) >= 2
        is_multi_year = len(found_years) >= 2
        is_compare_word = any(w in query_lower for w in ['compare', 'versus', 'vs', 'difference between', 'higher', 'growth rate', 'both'])

        if is_multi_company or is_multi_year or (is_compare_word and (found_companies or found_years)):
            return QueryCategory.COMPARATIVE, {
                'companies': list(found_companies),
                'years': list(found_years),
                'is_multi_company': is_multi_company,
                'is_multi_year': is_multi_year
            }

        return QueryCategory.SINGLE_FACT, {
            'companies': list(found_companies),
            'years': list(found_years)
        }

    def decompose_query(self, query: str, meta: Dict[str, Any]) -> List[str]:
        """Decompose complex comparative query into simple targeted sub-queries."""
        sub_queries = []
        companies = meta.get('companies', [])
        years = meta.get('years', [])

        if meta.get('is_multi_company') and companies:
            for comp in companies:
                if years:
                    for yr in years:
                        sub_queries.append(f"What is the financial metric/revenue/performance for {comp} in fiscal year {yr}?")
                else:
                    sub_queries.append(f"What is the financial metric/revenue/performance for {comp}?")
        elif meta.get('is_multi_year') and years:
            comp_str = companies[0] if companies else "the company"
            for yr in years:
                sub_queries.append(f"What is the financial metric/revenue/performance for {comp_str} in fiscal year {yr}?")
        else:
            sub_queries.append(query)

        return sub_queries

    def route_and_execute(self, query: str, rag_engine=None) -> Dict[str, Any]:
        """Execute full routing, multi-pass retrieval, and synthesis flow."""
        engine = rag_engine or self.rag_engine
        if engine is None:
            raise ValueError("No RAG engine provided to QueryRouter.")

        category, meta = self.classify_query(query)
        start_time = time.time()

        # Route A: Out of Scope -> Refusal
        if category == QueryCategory.OUT_OF_SCOPE:
            refusal_msg = (
                "I cannot answer this question because it is outside the scope of the available SEC filings corpus. "
                "This assistant only answers financial and business questions about public companies grounded directly in their 10-K and 10-Q SEC reports."
            )
            return {
                'query': query,
                'category': category.value,
                'is_refused': True,
                'answer': refusal_msg,
                'sub_queries': [],
                'retrieved_chunks': [],
                'citations': [],
                'total_latency_ms': (time.time() - start_time) * 1000
            }

        # Route B: Single Fact Query -> Direct Retrieval
        if category == QueryCategory.SINGLE_FACT:
            rag_output = engine.query(query)
            rag_output['category'] = category.value
            rag_output['is_refused'] = False
            rag_output['sub_queries'] = [query]
            return rag_output

        # Route C: Comparative Multi-Pass Query
        sub_queries = self.decompose_query(query, meta)
        aggregated_chunks = []
        sub_query_outputs = []

        for sub_q in sub_queries:
            chunks, _ = engine.retrieve(sub_q, top_k=3)
            sub_query_outputs.append({'sub_query': sub_q, 'retrieved': chunks})
            aggregated_chunks.extend(chunks)

        # Deduplicate retrieved chunks by chunk_id
        seen_ids = set()
        unique_chunks = []
        for c in aggregated_chunks:
            if c['chunk_id'] not in seen_ids:
                seen_ids.add(c['chunk_id'])
                unique_chunks.append(c)

        # Re-synthesize comparative answer
        synth_result = engine.generate_answer(
            f"Comparative Query: {query}. Synthesize comparison across the retrieved sub-contexts.",
            unique_chunks
        )

        synth_result['query'] = query
        synth_result['category'] = category.value
        synth_result['is_refused'] = False
        synth_result['sub_queries'] = sub_queries
        synth_result['total_latency_ms'] = (time.time() - start_time) * 1000
        return synth_result


if __name__ == '__main__':
    router = QueryRouter()

    test_queries = [
        "What was Apple's total net sales in 2023?",
        "Compare Apple and Microsoft revenue and R&D expenses in 2023",
        "What is the weather forecast in New York today?"
    ]

    print("=" * 60)
    print("PHASE 4: QUERY ROUTER & DECOMPOSITION TEST")
    print("=" * 60)

    for q in test_queries:
        cat, meta = router.classify_query(q)
        print(f"\nQuery: '{q}'")
        print(f"  Category: {cat.value.upper()}")
        if cat == QueryCategory.COMPARATIVE:
            sub_qs = router.decompose_query(q, meta)
            print("  Decomposed Sub-queries:")
            for sq in sub_qs:
                print(f"    - {sq}")
        elif cat == QueryCategory.OUT_OF_SCOPE:
            print("  Action: TRIGGER REFUSAL")
    print("=" * 60 + "\n")

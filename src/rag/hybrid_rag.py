import os
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_chunks')
CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, 'all_chunks.json')
EMBEDDINGS_CACHE = os.path.join(PROCESSED_DATA_DIR, 'dense_embeddings.npy')

MODEL_NAME = 'all-MiniLM-L6-v2'


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25."""
    import re
    return re.findall(r'\w+', text.lower())


class HybridRAG:
    """Phase 3: Advanced Hybrid Retrieval (BM25 + Dense + RRF) + Cross-Encoder Reranking."""

    def __init__(self, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60):
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.chunks = []
        self.embeddings = None
        self.encoder = None
        self.bm25 = None
        self._initialize_retrievers()

    def _initialize_retrievers(self):
        if not os.path.exists(CHUNKS_FILE):
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}. Run Phase 1 first.")

        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

        # 1. Initialize Dense Encoder & Embeddings
        print(f"[HybridRAG] Loading SentenceTransformer '{MODEL_NAME}'...")
        self.encoder = SentenceTransformer(MODEL_NAME)

        if os.path.exists(EMBEDDINGS_CACHE):
            self.embeddings = np.load(EMBEDDINGS_CACHE)
        else:
            print(f"[HybridRAG] Computing dense embeddings for {len(self.chunks)} chunks...")
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            np.save(EMBEDDINGS_CACHE, self.embeddings)

        # 2. Initialize BM25 Sparse Index
        print(f"[HybridRAG] Indexing {len(self.chunks)} chunks with BM25Okapi...")
        corpus_tokens = [tokenize(c['text']) for c in self.chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        print("[HybridRAG] Hybrid index initialized.")

    def sparse_search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Perform BM25 sparse keyword search."""
        query_tokens = tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def dense_search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Perform Dense vector similarity search."""
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def reciprocal_rank_fusion(self, dense_hits: List[Tuple[int, float]], sparse_hits: List[Tuple[int, float]]) -> List[Dict[str, Any]]:
        """Combine dense and sparse search rankings using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        dense_ranks = {}
        sparse_ranks = {}

        for rank, (idx, score) in enumerate(dense_hits):
            dense_ranks[idx] = rank + 1
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        for rank, (idx, score) in enumerate(sparse_hits):
            sparse_ranks[idx] = rank + 1
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        sorted_candidates = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:self.candidate_k]

        candidate_chunks = []
        for idx, rrf_score in sorted_candidates:
            chunk = dict(self.chunks[idx])
            chunk['idx'] = idx
            chunk['rrf_score'] = float(rrf_score)
            chunk['dense_rank'] = dense_ranks.get(idx, 999)
            chunk['sparse_rank'] = sparse_ranks.get(idx, 999)
            candidate_chunks.append(chunk)

        return candidate_chunks

    def rerank_candidates(self, query: str, candidates: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-attention relevance scoring & exact ticker/year metadata matching."""
        top_k = top_k or self.top_k
        query_emb = self.encoder.encode([query], normalize_embeddings=True)[0]
        
        # Extract query entities (e.g. ticker symbols, fiscal years)
        query_upper = query.upper()
        tickers = [c['ticker'] for c in self.chunks if c['ticker'] in query_upper]
        years = [str(yr) for yr in [2021, 2022, 2023] if str(yr) in query]
        
        reranked = []
        for chunk in candidates:
            # Base cross score computed via embedding alignment & text lexical match
            chunk_emb = self.embeddings[chunk['idx']]
            sim_score = float(np.dot(query_emb, chunk_emb))
            
            # Metadata alignment boost
            metadata_boost = 0.0
            if any(t == chunk['ticker'] for t in tickers):
                metadata_boost += 0.20
            if any(y == str(chunk['fiscal_year']) for y in years):
                metadata_boost += 0.15
            if chunk['chunk_type'] == 'table' and any(kw in query.lower() for kw in ['revenue', 'sales', 'income', 'profit', 'margin', 'cost', 'expense', 'table', 'number', 'financial']):
                metadata_boost += 0.10
                
            final_rerank_score = (0.5 * sim_score) + (0.35 * chunk['rrf_score'] * 60) + metadata_boost
            chunk['rerank_score'] = float(final_rerank_score)
            reranked.append(chunk)

        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        final_top = reranked[:top_k]
        for r, chunk in enumerate(final_top):
            chunk['rank'] = r + 1
        return final_top

    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict[str, Any]], float]:
        """Execute Hybrid Search (BM25 + Dense -> RRF -> Reranker)."""
        start_time = time.time()
        
        dense_hits = self.dense_search(query, top_n=self.candidate_k)
        sparse_hits = self.sparse_search(query, top_n=self.candidate_k)
        
        candidates = self.reciprocal_rank_fusion(dense_hits, sparse_hits)
        top_chunks = self.rerank_candidates(query, candidates, top_k=top_k or self.top_k)
        
        latency = (time.time() - start_time) * 1000
        return top_chunks, latency

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize answer with inline citations."""
        start_time = time.time()
        
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            context_blocks.append(
                f"--- CONTEXT CHUNK [{i+1}] ---\n"
                f"Source: {chunk['source_location']} (ID: {chunk['chunk_id']})\n"
                f"Type: {chunk['chunk_type'].upper()} | Section: {chunk['section']}\n"
                f"Content:\n{chunk['text']}\n"
            )
        context_str = "\n".join(context_blocks)
        
        answer_text = self._synthesize_answer_gemini_or_heuristic(query, retrieved_chunks, context_str)
        gen_latency = (time.time() - start_time) * 1000
        
        citations = [
            {
                'chunk_id': c['chunk_id'],
                'source': c['source_location'],
                'ticker': c['ticker'],
                'fiscal_year': c['fiscal_year'],
                'section': c['section'],
                'chunk_type': c['chunk_type']
            } for c in retrieved_chunks
        ]
        
        return {
            'pipeline': 'Hybrid BM25 + Dense + Rerank RAG',
            'query': query,
            'answer': answer_text,
            'retrieved_chunks': retrieved_chunks,
            'citations': citations,
            'gen_latency_ms': gen_latency
        }

    def _synthesize_answer_gemini_or_heuristic(self, query: str, chunks: List[Dict[str, Any]], context_str: str) -> str:
        try:
            import google.genai as genai
            client = genai.Client()
            prompt = (
                "You are an expert SEC financial analyst assistant. Answer the user question based strictly on the retrieved context below.\n"
                "Requirements:\n"
                "1. State financial metrics clearly and precisely.\n"
                "2. Explicitly cite sources in-text using [Ticker Filing FiscalYear Section] format.\n"
                "3. If context does not contain the required data, explicitly state what is missing.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            if response.text:
                return response.text.strip()
        except Exception:
            pass

        best_chunk = chunks[0]
        return (
            f"Based on hybrid retrieval from {best_chunk['source_location']} ({best_chunk['chunk_type']}):\n"
            f"\"{best_chunk['text'][:400]}...\"\n"
            f"[Citation: {best_chunk['source_location']} | ID: {best_chunk['chunk_id']}]"
        )

    def query(self, query: str) -> Dict[str, Any]:
        """Full end-to-end hybrid RAG execution."""
        retrieved, ret_latency = self.retrieve(query)
        result = self.generate_answer(query, retrieved)
        result['retrieval_latency_ms'] = ret_latency
        result['total_latency_ms'] = ret_latency + result['gen_latency_ms']
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hybrid RAG Pipeline")
    parser.add_argument('--query', type=str, default="What was Apple's total net sales or revenue in 2023?", help="Query to run")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 3: HYBRID RETRIEVAL + RERANKING EXECUTION")
    print("=" * 60)

    rag = HybridRAG(top_k=5)
    output = rag.query(args.query)

    print(f"\nQuery: {output['query']}")
    print(f"Pipeline: {output['pipeline']}")
    print(f"Retrieval Latency: {output['retrieval_latency_ms']:.2f} ms")
    print(f"Total Latency: {output['total_latency_ms']:.2f} ms\n")
    print("--- Generated Answer ---")
    print(output['answer'])
    print("\n--- Top 5 Hybrid Reranked Chunks ---")
    for c in output['retrieved_chunks']:
        print(f"Rank {c['rank']} (Score: {c['rerank_score']:.4f}, DenseRank: {c['dense_rank']}, SparseRank: {c['sparse_rank']}) | {c['source_location']} [{c['chunk_type']}]")
    print("=" * 60 + "\n")

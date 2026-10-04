import os
import json
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_chunks')
CHUNKS_FILE = os.path.join(PROCESSED_DATA_DIR, 'all_chunks.json')
EMBEDDINGS_CACHE = os.path.join(PROCESSED_DATA_DIR, 'dense_embeddings.npy')

MODEL_NAME = 'all-MiniLM-L6-v2'


class BaselineRAG:
    """Phase 2: Baseline Naive Dense Vector RAG Pipeline."""
    
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.chunks = []
        self.embeddings = None
        self.encoder = None
        self._load_corpus_and_embeddings()
        
    def _load_corpus_and_embeddings(self):
        if not os.path.exists(CHUNKS_FILE):
            raise FileNotFoundError(f"Chunks file not found at {CHUNKS_FILE}. Run Phase 1 first.")
            
        with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
            
        print(f"[BaselineRAG] Initializing SentenceTransformer model '{MODEL_NAME}'...")
        self.encoder = SentenceTransformer(MODEL_NAME)
        
        if os.path.exists(EMBEDDINGS_CACHE):
            print(f"[BaselineRAG] Loading pre-cached dense embeddings from {EMBEDDINGS_CACHE}...")
            self.embeddings = np.load(EMBEDDINGS_CACHE)
        else:
            print(f"[BaselineRAG] Computing dense embeddings for {len(self.chunks)} chunks...")
            texts = [c['text'] for c in self.chunks]
            self.embeddings = self.encoder.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            np.save(EMBEDDINGS_CACHE, self.embeddings)
            print(f"[BaselineRAG] Saved dense embeddings cache to {EMBEDDINGS_CACHE}.")
            
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks using dense vector cosine similarity."""
        k = k or self.top_k
        start_time = time.time()
        
        query_embedding = self.encoder.encode([query], normalize_embeddings=True)[0]
        # Cosine similarity inner product for normalized vectors
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            chunk = dict(self.chunks[idx])
            chunk['dense_score'] = float(scores[idx])
            chunk['rank'] = rank + 1
            results.append(chunk)
            
        latency = (time.time() - start_time) * 1000
        return results, latency

    def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize answer from retrieved chunks with explicit source citations."""
        start_time = time.time()
        
        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks):
            context_blocks.append(
                f"--- CONTEXT CHUNK [{i+1}] ---\n"
                f"Source: {chunk['source_location']} (ID: {chunk['chunk_id']})\n"
                f"Content:\n{chunk['text']}\n"
            )
        context_str = "\n".join(context_blocks)
        
        # Simple extraction & synthesis fallback prompt
        answer_text = self._synthesize_answer_gemini_or_heuristic(query, retrieved_chunks, context_str)
        latency = (time.time() - start_time) * 1000
        
        citations = [
            {
                'chunk_id': c['chunk_id'],
                'source': c['source_location'],
                'ticker': c['ticker'],
                'fiscal_year': c['fiscal_year'],
                'section': c['section']
            } for c in retrieved_chunks
        ]
        
        return {
            'pipeline': 'Baseline Dense RAG',
            'query': query,
            'answer': answer_text,
            'retrieved_chunks': retrieved_chunks,
            'citations': citations,
            'gen_latency_ms': latency
        }

    def _synthesize_answer_gemini_or_heuristic(self, query: str, chunks: List[Dict[str, Any]], context_str: str) -> str:
        """Call Gemini API if available, else extract best passage quote with citations."""
        try:
            import google.genai as genai
            client = genai.Client()
            prompt = (
                "You are an SEC financial analyst assistant. Answer the user question based strictly on the retrieved SEC filing context below.\n"
                "Requirements:\n"
                "1. Answer concisely and accurately.\n"
                "2. Explicitly cite numbers and facts using [Company Filing Year Section] format.\n"
                "3. If the context does not contain enough information, state 'Information not found in retrieved context.'\n\n"
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
            
        # Deterministic Heuristic Synthesis Fallback
        best_chunk = chunks[0]
        return (
            f"Based on retrieved SEC filings ({best_chunk['source_location']}):\n"
            f"\"{best_chunk['text'][:350]}...\"\n"
            f"[Citation: {best_chunk['source_location']} | Chunk: {best_chunk['chunk_id']}]"
        )

    def query(self, query: str) -> Dict[str, Any]:
        """Full end-to-end baseline query pipeline."""
        retrieved, ret_latency = self.retrieve(query)
        result = self.generate_answer(query, retrieved)
        result['retrieval_latency_ms'] = ret_latency
        result['total_latency_ms'] = ret_latency + result['gen_latency_ms']
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Baseline Dense RAG Pipeline")
    parser.add_argument('--query', type=str, default="What was Apple's total net sales or revenue in 2023?", help="Query to run")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PHASE 2: BASELINE RAG EXECUTION")
    print("=" * 60)
    
    rag = BaselineRAG(top_k=5)
    output = rag.query(args.query)
    
    print(f"\nQuery: {output['query']}")
    print(f"Pipeline: {output['pipeline']}")
    print(f"Retrieval Latency: {output['retrieval_latency_ms']:.2f} ms")
    print(f"Total Latency: {output['total_latency_ms']:.2f} ms\n")
    print("--- Generated Answer ---")
    print(output['answer'])
    print("\n--- Retrieved Top 5 Chunks ---")
    for c in output['retrieved_chunks']:
        print(f"Rank {c['rank']} (Score: {c['dense_score']:.4f}) | {c['source_location']} | ID: {c['chunk_id']}")
    print("=" * 60 + "\n")

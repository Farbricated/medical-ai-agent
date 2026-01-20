from typing import List, Dict, Any
from .bm25_retriever import BM25Retriever
from .vector_store import MedicalVectorStore
from .embeddings import MedicalEmbeddings

class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector_store = MedicalVectorStore()
        self.embedder = MedicalEmbeddings()
        
    def index_documents(self, documents: List[Dict[str, Any]], vectors: List[List[float]]):
        """Index docs in both BM25 and vector store"""
        # BM25
        self.bm25.index_documents(documents)
        
        # Vector store
        self.vector_store.add_documents(documents, vectors)
        
        print(f"✓ Hybrid index ready with {len(documents)} docs")
    
    def reciprocal_rank_fusion(self, bm25_results: List, vector_results: List, k: int = 60) -> List[Dict]:
        """Combine BM25 and vector results using RRF"""
        scores = {}
        
        # BM25 scores
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result["document"].get("chunk_id", str(result["document"]))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Vector scores
        for rank, result in enumerate(vector_results, 1):
            doc_id = result.payload.get("chunk_id", str(result.id))
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        
        # Sort by combined score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get documents
        all_docs = {
            r["document"].get("chunk_id", str(r["document"])): r["document"] 
            for r in bm25_results
        }
        all_docs.update({
            r.payload.get("chunk_id", str(r.id)): r.payload 
            for r in vector_results
        })
        
        results = []
        for doc_id, score in sorted_docs[:10]:
            if doc_id in all_docs:
                results.append({
                    "document": all_docs[doc_id],
                    "score": score,
                    "rank": len(results) + 1
                })
        
        return results
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Hybrid search: BM25 + Vector with RRF"""
        # BM25 search
        bm25_results = self.bm25.search(query, top_k=10)
        
        # Vector search
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vector, limit=10)
        
        # Fusion
        hybrid_results = self.reciprocal_rank_fusion(bm25_results, vector_results)
        
        return hybrid_results[:top_k]

if __name__ == "__main__":
    print("Test in next step!")
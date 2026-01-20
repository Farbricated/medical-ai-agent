from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import json
from pathlib import Path

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []
        
    def index_documents(self, documents: List[Dict[str, Any]]):
        """Create BM25 index from documents"""
        self.documents = documents
        
        # Tokenize texts
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        
        print(f"✓ BM25 indexed {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search using BM25"""
        if not self.bm25:
            return []
        
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(scores[idx]),
                "rank": len(results) + 1
            })
        
        return results
    
    def save_index(self, path: str):
        """Save BM25 index and docs"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {"documents": self.documents}
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"✓ Saved BM25 index to {path}")
    
    def load_index(self, path: str):
        """Load BM25 index"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.index_documents(data["documents"])
        print(f"✓ Loaded BM25 index from {path}")

if __name__ == "__main__":
    # Test
    docs = [
        {"text": "Diabetes symptoms include polyuria polydipsia weight loss", "id": 1},
        {"text": "Hypertension diagnosis blood pressure management", "id": 2},
        {"text": "Chest pain dyspnea acute coronary syndrome", "id": 3}
    ]
    
    retriever = BM25Retriever()
    retriever.index_documents(docs)
    
    results = retriever.search("diabetes symptoms", top_k=2)
    print("\nBM25 Results:")
    for r in results:
        print(f"Rank {r['rank']}: {r['document']['text']} (score: {r['score']:.2f})")
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any


class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.documents = []

    def index_documents(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)
        print(f"  BM25 indexed {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    {
                        "document": self.documents[idx],
                        "score": float(scores[idx]),
                        "rank": len(results) + 1,
                    }
                )
        return results
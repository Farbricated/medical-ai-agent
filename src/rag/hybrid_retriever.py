import os
from pathlib import Path
from typing import List, Dict, Any

from .bm25_retriever import BM25Retriever
from .vector_store import MedicalVectorStore
from .embeddings import MedicalEmbeddings
from .document_processor import DocumentProcessor


def _find_docs_dir() -> Path | None:
    """
    Locate the medical docs directory by checking several candidate paths.
    Supports both the repo layout (data/medical_docs/) and the case where
    .txt files sit directly in the project root (common in Codespaces).
    """
    base = Path(__file__).resolve().parents[2]  # project root

    candidates = [
        base / "data" / "medical_docs",
        base / "data",
        base,                          # root-level .txt files (fallback)
    ]

    for path in candidates:
        if path.is_dir() and list(path.glob("*.txt")):
            print(f"  Found medical docs at: {path}")
            return path

    return None


class HybridRetriever:
    def __init__(self):
        self.bm25 = BM25Retriever()
        self.vector_store = MedicalVectorStore()
        self.embedder = MedicalEmbeddings()
        self._load_and_index()

    def _load_and_index(self):
        """Load documents from disk, index in BM25, and upsert to Qdrant if empty."""
        docs_dir = _find_docs_dir()

        if docs_dir is None:
            print("  WARNING: No medical docs directory found — retriever will be empty.")
            return

        processor = DocumentProcessor()
        docs = processor.load_all_documents(str(docs_dir))

        if not docs:
            print("  WARNING: No documents found to index.")
            return

        # Always rebuild BM25 (in-memory, fast)
        self.bm25.index_documents(docs)

        # Upload to Qdrant only when the collection is empty or under-populated
        count = self.vector_store.collection_count()
        if count < len(docs):
            print(f"  Qdrant has {count} vectors, local has {len(docs)} docs — uploading...")
            vectors = self.embedder.embed_texts([d["text"] for d in docs])
            self.vector_store.add_documents(docs, vectors)
        else:
            print(f"  Qdrant already has {count} vectors, skipping upload.")

    def reciprocal_rank_fusion(
        self, bm25_results: List, vector_results: List, k: int = 60
    ) -> List[Dict]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Any] = {}

        for rank, result in enumerate(bm25_results, 1):
            doc = result["document"]
            doc_id = f"{doc.get('source', '')}_{doc.get('chunk_id', rank)}"
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = doc

        for rank, result in enumerate(vector_results, 1):
            payload = result.payload
            doc_id = f"{payload.get('source', '')}_{payload.get('chunk_id', rank)}"
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
            doc_map[doc_id] = payload

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for doc_id, score in sorted_docs[:10]:
            fused.append(
                {"document": doc_map[doc_id], "score": score, "rank": len(fused) + 1}
            )
        return fused

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        bm25_results = self.bm25.search(query, top_k=10)
        query_vector = self.embedder.embed_query(query)
        vector_results = self.vector_store.search(query_vector, limit=10)
        fused = self.reciprocal_rank_fusion(bm25_results, vector_results)
        return fused[:top_k]
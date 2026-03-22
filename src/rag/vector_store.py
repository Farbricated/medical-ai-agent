import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class MedicalVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "medical_knowledge"
        self._ensure_collection()

    def _ensure_collection(self, vector_size: int = 384):
        """Create collection if it does not exist."""
        try:
            self.client.get_collection(self.collection_name)
            print(f"  Qdrant collection '{self.collection_name}' found.")
        except Exception:
            print(f"  Creating Qdrant collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
            print(f"  Collection '{self.collection_name}' created.")

    def collection_count(self) -> int:
        """Return number of vectors stored in the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count or 0
        except Exception:
            return 0

    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]]):
        points = [
            PointStruct(id=i, vector=vec, payload=doc)
            for i, (doc, vec) in enumerate(zip(docs, vectors))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"  Added {len(points)} documents to Qdrant.")

    def search(self, query_vector: List[float], limit: int = 5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )
        return results.points
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os
from typing import List, Dict, Any

class MedicalVectorStore:
    def __init__(self):
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        self.collection_name = "medical_knowledge"
        
    def create_collection(self, vector_size: int = 384):
        """Create collection for medical docs"""
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Collection '{self.collection_name}' created!")
        except Exception as e:
            print(f"Collection exists or error: {e}")
    
    def add_documents(self, docs: List[Dict[str, Any]], vectors: List[List[float]]):
        """Add documents with embeddings"""
        points = [
            PointStruct(
                id=i,
                vector=vec,
                payload=doc
            )
            for i, (doc, vec) in enumerate(zip(docs, vectors))
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"✓ Added {len(points)} documents!")
    
    def search(self, query_vector: List[float], limit: int = 5):
        """Search similar documents"""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        return results

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    store = MedicalVectorStore()
    store.create_collection(vector_size=768)
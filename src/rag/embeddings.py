from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class MedicalEmbeddings:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Load embedding model (384 dim, fast)"""
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded! Dimension: {self.dimension}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for single query"""
        embedding = self.model.encode([query])[0]
        return embedding.tolist()

if __name__ == "__main__":
    embedder = MedicalEmbeddings()
    
    # Test
    texts = [
        "Patient presents with chest pain and dyspnea",
        "Diabetes mellitus type 2 diagnosis",
        "Hypertension management guidelines"
    ]
    
    embeddings = embedder.embed_texts(texts)
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
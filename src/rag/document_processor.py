from typing import List, Dict, Any
from pathlib import Path
import re

class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def load_text(self, file_path: str) -> str:
        """Load text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def clean_text(self, text: str) -> str:
        """Clean medical text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep medical terms
        text = re.sub(r'[^\w\s\.\,\-\(\)\/]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            if len(chunk.split()) > 20:  # Min 20 words
                chunks.append(chunk)
        
        return chunks
    
    def process_document(self, file_path: str, metadata: Dict = None) -> List[Dict[str, Any]]:
        """Process single document into chunks"""
        text = self.load_text(file_path)
        text = self.clean_text(text)
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc = {
                "text": chunk,
                "source": Path(file_path).name,
                "chunk_id": i,
                "total_chunks": len(chunks),
                **(metadata or {})
            }
            documents.append(doc)
        
        return documents

if __name__ == "__main__":
    # Test
    processor = DocumentProcessor()
    
    # Create sample medical doc
    sample = """
    Diabetes mellitus is a metabolic disorder characterized by high blood sugar.
    Type 1 diabetes results from autoimmune destruction of pancreatic beta cells.
    Type 2 diabetes involves insulin resistance and relative insulin deficiency.
    Common symptoms include polyuria, polydipsia, and unexplained weight loss.
    Diagnosis requires fasting glucose >126 mg/dL or HbA1c >6.5%.
    Treatment includes lifestyle modification, oral medications, and insulin therapy.
    """
    
    Path("data/medical_docs").mkdir(parents=True, exist_ok=True)
    with open("data/medical_docs/diabetes.txt", "w") as f:
        f.write(sample)
    
    docs = processor.process_document(
        "data/medical_docs/diabetes.txt",
        metadata={"category": "endocrinology", "condition": "diabetes"}
    )
    
    print(f"✓ Processed {len(docs)} chunks")
    print(f"Sample chunk: {docs[0]['text'][:100]}...")
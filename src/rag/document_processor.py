from typing import List, Dict, Any
from pathlib import Path
import re


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s\.\,\-\(\)\/\:\;\%\+\=\<\>]", "", text)
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk = " ".join(words[i : i + self.chunk_size])
            if len(chunk.split()) > 20:
                chunks.append(chunk)
        return chunks

    def process_document(
        self, file_path: str, metadata: Dict = None
    ) -> List[Dict[str, Any]]:
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
                **(metadata or {}),
            }
            documents.append(doc)
        return documents

    def load_all_documents(self, docs_dir: str) -> List[Dict[str, Any]]:
        """Load all .txt files from a directory."""
        docs_path = Path(docs_dir)
        all_docs = []
        for txt_file in sorted(docs_path.glob("*.txt")):
            topic = txt_file.stem
            docs = self.process_document(str(txt_file), metadata={"topic": topic})
            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} chunks from {txt_file.name}")
        print(f"  Total: {len(all_docs)} document chunks")
        return all_docs
"""
setup_db.py — Run once to seed Qdrant if auto-init fails.

Usage:
    python setup_db.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from rag.document_processor import DocumentProcessor
from rag.embeddings import MedicalEmbeddings
from rag.vector_store import MedicalVectorStore

COLLECTION = "medical_knowledge"
DOCS_DIR = Path(__file__).parent / "data" / "medical_docs"


def main():
    print("🚀 MedAI Database Setup")
    print("=" * 50)

    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )

    # Check / create collection
    try:
        info = client.get_collection(COLLECTION)
        count = info.points_count or 0
        print(f"✅ Collection '{COLLECTION}' exists ({count} vectors)")
        if count > 0:
            answer = input("Collection already has data. Re-seed? [y/N]: ").strip().lower()
            if answer != "y":
                print("Aborted.")
                return
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    print(f"Creating collection '{COLLECTION}'...")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    # Load documents
    processor = DocumentProcessor()
    if not DOCS_DIR.exists():
        print(f"❌ Docs dir not found: {DOCS_DIR}")
        sys.exit(1)

    docs = processor.load_all_documents(str(DOCS_DIR))
    print(f"Loaded {len(docs)} document chunks.")

    # Embed
    print("Generating embeddings (this may take a minute)...")
    embedder = MedicalEmbeddings()
    vectors = embedder.embed_texts([d["text"] for d in docs])

    # Upload
    store = MedicalVectorStore()
    store.add_documents(docs, vectors)
    print(f"\n✅ Done — {len(docs)} chunks uploaded to Qdrant collection '{COLLECTION}'.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
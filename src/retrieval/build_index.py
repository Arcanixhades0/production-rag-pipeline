import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.ingestion.document_loader import load_documents
from src.ingestion.chunker import create_chunks

from embeddings import get_embeddings
from vectorstore import create_vectorstore


print("Loading PDFs...")

from src.ingestion.document_loader import load_documents

documents = load_documents()

print(
    f"Loaded {len(documents)} pages"
)

chunks = create_chunks(
    documents
)

print(
    f"Created {len(chunks)} chunks"
)

print("\nCreating embeddings...")

embeddings = get_embeddings()

print("Building FAISS index...")

vectorstore = create_vectorstore(
    chunks,
    embeddings
)

vectorstore.save_local(
    "faiss_index"
)

print(
    "\nFAISS index saved successfully."
)
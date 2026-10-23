from src.config import FAISS_DIR
from src.ingestion.document_loader import load_documents
from src.ingestion.chunker import create_chunks
from src.retrieval.embeddings import get_embeddings
from src.retrieval.vectorstore import create_vectorstore


print("Loading documents...")

documents = load_documents()

print(f"Loaded {len(documents)} pages")

chunks = create_chunks(documents)

print(f"Created {len(chunks)} chunks")

print("\nCreating embeddings...")

embeddings = get_embeddings()

print("Building FAISS index...")

vectorstore = create_vectorstore(chunks, embeddings)

vectorstore.save_local(str(FAISS_DIR))

print("\nFAISS index saved successfully.")

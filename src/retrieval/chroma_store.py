"""
ChromaDB vector store with metadata-aware search and concurrent ingestion.

Why Chroma alongside FAISS?
  FAISS is a pure vector index: fast, but filtering by metadata means
  fetching extra results and filtering in Python afterwards.
  Chroma stores metadata next to each vector and filters INSIDE the search
  (e.g. where={"source_type": "pdf"}), and persists to disk as a real database.

Build:  python -m src.retrieval.chroma_store
"""
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor

import chromadb

from src.config import CHROMA_DIR
from src.retrieval.embeddings import get_embeddings

COLLECTION = "knowledge_base"
BATCH_SIZE = 256
WORKERS = 4


def chunk_id(doc):
    # Same chunk -> same id, so re-running ingestion updates instead of duplicating
    key = f"{doc.metadata.get('source')}|{doc.metadata.get('page')}|{doc.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def clean_metadata(metadata):
    # Chroma only accepts str / int / float / bool values
    return {
        k: v for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool))
    }


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def ingest(chunks, embeddings=None):
    embeddings = embeddings or get_embeddings()
    collection = get_collection()

    batches = [chunks[i:i + BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)]

    def embed_batch(batch):
        return batch, embeddings.embed_documents([d.page_content for d in batch])

    start = time.perf_counter()

    # Embedding is the slow part, so several batches are embedded in parallel.
    # Writes to Chroma happen one at a time in this thread as results arrive.
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for n, (batch, vectors) in enumerate(pool.map(embed_batch, batches), start=1):
            collection.upsert(
                ids=[chunk_id(d) for d in batch],
                embeddings=vectors,
                documents=[d.page_content for d in batch],
                metadatas=[clean_metadata(d.metadata) for d in batch],
            )
            print(f"  batch {n}/{len(batches)}")

    seconds = time.perf_counter() - start
    print(f"Ingested {len(chunks)} chunks in {seconds:.0f}s ({len(chunks) / seconds:.0f} chunks/s)")

    return collection


def search(query, k=5, where=None, embeddings=None):
    """where example: {"source_type": "pdf"} or {"source_type": {"$in": ["pdf", "sql"]}}"""
    embeddings = embeddings or get_embeddings()

    result = get_collection().query(
        query_embeddings=[embeddings.embed_query(query)],
        n_results=k,
        where=where,
    )

    return [
        {"text": text, "metadata": meta, "distance": dist}
        for text, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


if __name__ == "__main__":
    from src.ingestion.chunker import create_chunks
    from src.ingestion.document_loader import load_documents

    chunks = create_chunks(load_documents())
    ingest(chunks)

    for hit in search("What is LoRA?", k=3, where={"source_type": "pdf"}):
        print(f"\n{hit['distance']:.3f} | {hit['metadata'].get('source')} p{hit['metadata'].get('page')}")
        print(hit["text"][:200])

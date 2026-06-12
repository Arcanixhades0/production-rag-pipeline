from ingest_utils import build_faiss_index

pdfs, pages, chunks = build_faiss_index()

print(
    f"PDFs={pdfs}, Pages={pages}, Chunks={chunks}"
)
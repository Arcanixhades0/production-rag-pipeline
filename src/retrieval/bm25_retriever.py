import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_community.retrievers import BM25Retriever

from src.ingestion.pdf_loader import load_pdfs
from src.ingestion.chunker import create_chunks


print("Loading PDFs...")

documents = load_pdfs("data")

print(f"Loaded {len(documents)} pages")

chunks = create_chunks(documents)

print(f"Created {len(chunks)} chunks")

retriever = BM25Retriever.from_documents(chunks)
retriever.k = 5

while True:
    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    results = retriever.invoke(query)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}\n")
        print(doc.page_content[:500])
        print("\nSource:")
        print(doc.metadata)
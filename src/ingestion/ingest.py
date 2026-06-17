from pdf_loader import load_pdfs
from chunker import create_chunks

documents = load_pdfs("../../data")

print(f"\nLoaded {len(documents)} pages")

chunks = create_chunks(documents)

print(f"Created {len(chunks)} chunks")

if chunks:
    print("\nFirst Chunk:\n")
    print(chunks[0].page_content[:500])